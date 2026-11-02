# Week 5 results

## Agreement with the input photo (per item, mean over seeds)
| metric       |   n_items |   trellis2 |   trellis2_lo |   trellis2_hi |   hunyuan3d |   hunyuan3d_lo |   hunyuan3d_hi |   diff_A_minus_B |   diff_lo |   diff_hi |   A_wins |   B_wins | verdict             |
|:-------------|----------:|-----------:|--------------:|--------------:|------------:|---------------:|---------------:|-----------------:|----------:|----------:|---------:|---------:|:--------------------|
| dino_best    |        13 |      0.783 |         0.721 |         0.839 |       0.786 |          0.728 |          0.840 |           -0.003 |    -0.056 |     0.044 |        7 |        6 | no clear difference |
| dino_mean    |        13 |      0.660 |         0.597 |         0.709 |       0.654 |          0.605 |          0.703 |            0.005 |    -0.038 |     0.046 |        8 |        5 | no clear difference |
| clip_i_best  |        13 |      0.922 |         0.903 |         0.939 |       0.923 |          0.907 |          0.940 |           -0.002 |    -0.019 |     0.015 |        6 |        7 | no clear difference |
| clip_i_mean  |        13 |      0.889 |         0.867 |         0.909 |       0.886 |          0.863 |          0.905 |            0.004 |    -0.014 |     0.023 |        7 |        6 | no clear difference |
| sil_iou_best |        13 |      0.805 |         0.742 |         0.855 |       0.807 |          0.739 |          0.862 |           -0.002 |    -0.027 |     0.025 |        5 |        8 | no clear difference |

## Metric sanity check: matching photo vs a different item's photo
| metric      | model     |   matching_photo |   different_item_photo |   gap |
|:------------|:----------|-----------------:|-----------------------:|------:|
| dino_best   | trellis2  |            0.783 |                  0.104 | 0.679 |
| dino_best   | hunyuan3d |            0.786 |                  0.109 | 0.676 |
| clip_i_best | trellis2  |            0.922 |                  0.732 | 0.189 |
| clip_i_best | hunyuan3d |            0.923 |                  0.738 | 0.186 |

## Run cost per mesh
| metric                   | model     |   n |    mean |     min |     max |
|:-------------------------|:----------|----:|--------:|--------:|--------:|
| latency_s                | trellis2  |  13 | 317.701 | 167.365 | 912.452 |
| latency_s                | hunyuan3d |  13 | 116.096 |  87.403 | 209.533 |
| latency_shape_s          | trellis2  |  13 |  55.135 |  21.132 | 122.669 |
| latency_shape_s          | hunyuan3d |  13 |  30.836 |  30.250 |  32.579 |
| latency_texture_export_s | trellis2  |  13 | 262.565 | 101.959 | 891.319 |
| latency_texture_export_s | hunyuan3d |  13 |  84.999 |  56.912 | 176.600 |
| peak_vram_gb             | trellis2  |  13 |   7.749 |   2.850 |  15.349 |
| peak_vram_gb             | hunyuan3d |  13 |  20.618 |  20.212 |  20.720 |
| peak_vram_reserved_gb    | trellis2  |  13 |   8.853 |   3.561 |  18.730 |
| peak_vram_reserved_gb    | hunyuan3d |  13 |  23.182 |  22.484 |  23.639 |
| load_time_s              | trellis2  |   1 | 129.496 | 129.496 | 129.496 |
| load_time_s              | hunyuan3d |   1 |  82.739 |  82.739 |  82.739 |

## Geometry and reliability
| metric                  | model     |   n |       mean |        min |        max |
|:------------------------|:----------|----:|-----------:|-----------:|-----------:|
| n_faces                 | trellis2  |  13 | 965088.692 | 934811.000 | 989034.000 |
| n_faces                 | hunyuan3d |  13 |  40000.000 |  40000.000 |  40000.000 |
| watertight              | trellis2  |  13 |      0.000 |      0.000 |      0.000 |
| watertight              | hunyuan3d |  13 |      1.000 |      1.000 |      1.000 |
| n_components            | trellis2  |  13 |     96.308 |      1.000 |    520.000 |
| n_components            | hunyuan3d |  13 |      4.846 |      1.000 |     49.000 |
| textured                | trellis2  |  13 |      1.000 |      1.000 |      1.000 |
| textured                | hunyuan3d |  13 |      1.000 |      1.000 |      1.000 |
| glb_size_mb             | trellis2  |  13 |     69.446 |     60.343 |     86.765 |
| glb_size_mb             | hunyuan3d |  13 |      1.308 |      1.043 |      2.096 |
| generation_success_rate | trellis2  |  13 |      1.000 |      1.000 |      1.000 |
| generation_success_rate | hunyuan3d |  13 |      1.000 |      1.000 |      1.000 |

## Seed stability (mean within-item SD)
| metric       | model     |   mean_within_item_sd |
|:-------------|:----------|----------------------:|
| dino_best    | trellis2  |                 0.000 |
| dino_best    | hunyuan3d |                 0.000 |
| sil_iou_best | trellis2  |                 0.000 |
| sil_iou_best | hunyuan3d |                 0.000 |
