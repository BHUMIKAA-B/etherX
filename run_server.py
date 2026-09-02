
import os
import multiprocessing

#  Settings 
HOST        = os.getenv("HOST",    "0.0.0.0")
PORT        = int(os.getenv("PORT", "8000"))
WORKERS     = int(os.getenv("WORKERS", str(min(4, multiprocessing.cpu_count()))))
LOG_LEVEL   = os.getenv("LOG_LEVEL", "info").lower()
RELOAD      = os.getenv("RELOAD", "false").lower() == "true"   # dev only

if __name__ == "__main__":
    try:
        
        import gunicorn.app.wsgiapp  # noqa: F401
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "gunicorn",
            "main:app",
            "--workers", str(WORKERS),
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--bind", f"{HOST}:{PORT}",
            "--timeout", "120",
            "--keep-alive", "5",
            "--log-level", LOG_LEVEL,
            "--access-logfile", "-",
            "--error-logfile", "-",
        ]
        print(f"Starting gunicorn with {WORKERS} worker(s) on {HOST}:{PORT}")
        subprocess.run(cmd)
    except ImportError:
        # gunicorn not installed (Windows dev) — fall back to uvicorn directly
        import uvicorn
        print(f"gunicorn not found — starting uvicorn (single worker) on {HOST}:{PORT}")
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            log_level=LOG_LEVEL,
            reload=RELOAD,
        )
