"""Light-weight renderer for scoring generated meshes: no OpenGL, no GPU, same code for both models.

A GLB is turned into a dense coloured point cloud (area-uniform surface samples; colour from the base-colour
texture, or vertex colours, or grey), then drawn with a z-buffer from fixed orthographic cameras. Shading is a
mild headlight term (0.7 + 0.3 * |n.z|) so geometry shows, while colour stays close to the albedo.
This is an approximation of a real renderer. Both models go through the same path, so the comparison is fair,
but absolute numbers should not be compared with papers. The view sheets saved by score_w5.py let you check it.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def load_mesh(path):
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)   # no automatic merge/repair: slow and memory-hungry on 1M-face meshes
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def _texture_image(mesh):
    vis = mesh.visual
    mat = getattr(vis, "material", None)
    if mat is None:
        return None
    for name in ("baseColorTexture", "image"):
        img = getattr(mat, name, None)
        if img is not None:
            return img
    return None


def _bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = tex.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = (1.0 - np.mod(uv[:, 1], 1.0)) * (h - 1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    t = tex.astype(np.float32)
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def sample_points(mesh, n: int = 600_000, seed: int = 0):
    """Returns points (n,3), colours uint8 (n,3), normals (n,3), textured flag."""
    import trimesh

    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    normals = mesh.face_normals[fidx]
    colours, textured = None, False
    vis = mesh.visual
    kind = getattr(vis, "kind", None)
    tex = _texture_image(mesh) if kind == "texture" else None
    uv = getattr(vis, "uv", None)
    if tex is not None and uv is not None and len(uv) == len(mesh.vertices):
        tri = mesh.triangles[fidx]
        bary = trimesh.triangles.points_to_barycentric(tri, pts)
        puv = (uv[mesh.faces[fidx]] * bary[:, :, None]).sum(axis=1)
        arr = np.asarray(tex.convert("RGB"))
        colours = np.clip(_bilinear(arr, puv), 0, 255)
        textured = True
    elif kind == "vertex":
        vc = np.asarray(vis.vertex_colors)[:, :3].astype(np.float32)
        tri = mesh.triangles[fidx]
        bary = trimesh.triangles.points_to_barycentric(tri, pts)
        colours = (vc[mesh.faces[fidx]] * bary[:, :, None]).sum(axis=1)
        textured = True
    if colours is None:
        colours = np.full((len(pts), 3), 180.0, dtype=np.float32)
    return np.asarray(pts, dtype=np.float32), colours.astype(np.uint8), np.asarray(normals, dtype=np.float32), textured


def orient(points: np.ndarray, normals: np.ndarray, up: str):
    """Rotate so the model's up axis is +y, then centre on the bounding-box centre."""
    if up == "z":
        r = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)   # z -> y
        points, normals = points @ r.T, normals @ r.T
    elif up != "y":
        raise ValueError(f"up must be 'y' or 'z', got {up!r}")
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    return points - centre, normals


def _rot(azim_deg: float, elev_deg: float) -> np.ndarray:
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]], dtype=np.float32)
    return rx @ ry


def render(points, colours, normals, azim_deg, elev_deg=15.0, size=512, radius=None, splat=3):
    """Orthographic z-buffer render on a white background. Returns (rgb uint8 HxWx3, mask bool HxW)."""
    from scipy import ndimage

    r = _rot(azim_deg, elev_deg)
    p = points @ r.T
    nz = np.abs((normals @ r.T)[:, 2])
    if radius is None:
        radius = float(np.linalg.norm(points, axis=1).max())
    scale = 0.46 * size / max(radius, 1e-8)
    x = np.round(size / 2 + p[:, 0] * scale).astype(np.int64)
    y = np.round(size / 2 - p[:, 1] * scale).astype(np.int64)
    depth = p[:, 2]
    shade = (0.7 + 0.3 * nz)[:, None]
    col = np.clip(colours.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    half = splat // 2
    pix_all, dep_all, idx_all = [], [], []
    base = np.arange(len(x))
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
            pix_all.append(yy[ok] * size + xx[ok])
            dep_all.append(depth[ok])
            idx_all.append(base[ok])
    pix, dep, idx = np.concatenate(pix_all), np.concatenate(dep_all), np.concatenate(idx_all)
    order = np.lexsort((-dep, pix))          # per pixel: the point nearest the camera (largest z) first
    pix, idx = pix[order], idx[order]
    first = np.ones(len(pix), dtype=bool)
    first[1:] = pix[1:] != pix[:-1]
    pix, idx = pix[first], idx[first]
    img = np.full((size * size, 3), 255, dtype=np.uint8)
    img[pix] = col[idx]
    mask = np.zeros(size * size, dtype=bool)
    mask[pix] = True
    mask = ndimage.binary_closing(mask.reshape(size, size), structure=np.ones((3, 3)), iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    return img.reshape(size, size, 3), mask


def crop_square(rgb: np.ndarray, mask: np.ndarray, out: int = 224, margin: float = 0.1):
    """Crop a square around the mask's bounding box (plus margin), resize to out x out. Background stays white."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.full((out, out, 3), 255, dtype=np.uint8), np.zeros((out, out), dtype=bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(round(max(y1 - y0, x1 - x0) * (1 + 2 * margin)))
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    top, left = int(round(cy - side / 2)), int(round(cx - side / 2))
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    mcanvas = np.zeros((side, side), dtype=bool)
    sy0, sx0 = max(top, 0), max(left, 0)
    sy1, sx1 = min(top + side, rgb.shape[0]), min(left + side, rgb.shape[1])
    canvas[sy0 - top: sy1 - top, sx0 - left: sx1 - left] = rgb[sy0:sy1, sx0:sx1]
    mcanvas[sy0 - top: sy1 - top, sx0 - left: sx1 - left] = mask[sy0:sy1, sx0:sx1]
    img = np.asarray(Image.fromarray(canvas).resize((out, out), Image.BICUBIC))
    m = np.asarray(Image.fromarray(mcanvas.astype(np.uint8) * 255).resize((out, out), Image.NEAREST)) > 127
    return img, m


def load_photo(rgba_path, size: int = 224, margin: float = 0.1):
    """Background-removed input (RGBA PNG) -> (white-background crop, mask) in the same crop convention as renders."""
    im = np.asarray(Image.open(rgba_path).convert("RGBA"))
    mask = im[..., 3] > 127
    rgb = im[..., :3].astype(np.float32)
    a = (im[..., 3:4].astype(np.float32)) / 255.0
    comp = np.clip(rgb * a + 255.0 * (1 - a), 0, 255).astype(np.uint8)
    return crop_square(comp, mask, size, margin)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def mesh_stats(mesh) -> dict:
    """Geometry facts, computed with plain sparse arrays (linear memory).

    Vertices are first merged by position, so UV-seam splits do not count as holes or extra pieces.
    watertight = every edge is shared by exactly two faces; n_components = connected pieces of the surface.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_faces, n_vertices = int(len(faces)), int(len(verts))
    if n_faces == 0:
        return {"n_faces": 0, "n_vertices": n_vertices, "watertight": 0.0, "n_components": 0}
    span = float(np.ptp(verts, axis=0).max()) or 1.0
    keys = np.round(verts / (span * 1e-6)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    f = inverse.reshape(-1)[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]   # drop collapsed faces
    if len(f) == 0:
        return {"n_faces": n_faces, "n_vertices": n_vertices, "watertight": 0.0, "n_components": 0}
    e = np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    nv = int(f.max()) + 1
    _, counts = np.unique(e[:, 0] * nv + e[:, 1], return_counts=True)
    used = np.unique(f)
    graph = coo_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(nv, nv))
    n_comp_all = connected_components(graph, directed=False)[0]
    n_comp = int(n_comp_all - (nv - len(used)))   # vertices no face uses are not pieces
    return {"n_faces": n_faces, "n_vertices": n_vertices, "watertight": float(bool(np.all(counts == 2))),
            "n_components": n_comp}
