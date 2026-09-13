"""
Root entrypoint proxy forwarding to backend.api:app.
Maintains backward compatibility with uvicorn api:app and Dockerfile defaults.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_ROOT, "backend")
for _p in [_ROOT, _BACKEND]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.api import app, load_artifacts
import backend.api as _backend_api
for _attr in dir(_backend_api):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_backend_api, _attr)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
