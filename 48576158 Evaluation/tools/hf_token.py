"""Resolve a Hugging Face token from whichever source is available.

Order: the HF_TOKEN environment variable, then a token file in Drive or the
home directory, then Colab Secrets, then an interactive prompt. Returns an
empty string if the user declines, so an ungated run still proceeds.

Colab Secrets (google.colab.userdata) are served by the Colab *web frontend*,
not by the runtime, so they time out whenever a notebook is driven from
VS Code or any other client. Every other source here works everywhere, which
also means someone else can run these notebooks without a secret of our name
configured in their own account.
"""

from __future__ import annotations

import os
import pathlib

TOKEN_FILES = ("/content/drive/MyDrive/.hf_token", "~/.hf_token")


def hf_token(prompt: bool = True) -> str:
    """Return a token, or "" if none is available and none is given."""
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        return tok

    for candidate in TOKEN_FILES:
        path = pathlib.Path(candidate).expanduser()
        try:
            if path.is_file():
                tok = path.read_text(encoding="utf-8").strip()
                if tok:
                    return tok
        except OSError:
            pass

    try:  # only reachable from the Colab web UI
        from google.colab import userdata

        tok = (userdata.get("HF_TOKEN") or "").strip()
        if tok:
            return tok
    except Exception:
        pass

    if prompt:
        from getpass import getpass

        tok = getpass("HF token (blank to skip gated models): ").strip()
    return tok


def setup(hf_home: str = "/content/hf_cache", login: bool = True) -> str:
    """Resolve the token, export it, keep the cache off Drive, and log in."""
    os.environ["HF_HOME"] = hf_home  # must precede any huggingface_hub import
    tok = hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
        if login:
            from huggingface_hub import login as hf_login

            hf_login(token=tok, add_to_git_credential=False)
    print("HF token:", "set" if tok else "NOT set - gated models will fail")
    return tok
