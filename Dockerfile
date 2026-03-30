# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 – Install Python dependencies
# Using a slim Debian-based image that ships libstdc++ needed by TensorFlow.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Copy only the dependency manifest first (better layer-caching)
COPY requirements.txt .

# Install into a dedicated prefix so we can copy to the final stage cleanly
RUN pip install --upgrade pip && \
    pip install --prefix=/runtime --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – Final runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python where the custom prefix libraries live
    PYTHONPATH=/runtime/lib/python3.10/site-packages \
    PATH="/runtime/bin:$PATH" \
    # Default port (can be overridden via `docker run -e PORT=8080`)
    PORT=5000

WORKDIR /app

# Bring in the installed packages from the builder stage
COPY --from=builder /runtime /runtime

# Copy application source code
COPY app.py        ./app.py
COPY templates/    ./templates/
COPY static/       ./static/

# Copy pre-trained model artifacts
COPY model/        ./model/

# Expose the port the app listens on
EXPOSE ${PORT}

# Health-check so Docker / orchestrators know when the container is ready.
# The /health endpoint returns {"status": "ok"} immediately.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Run with Gunicorn (production-grade WSGI server, already in requirements.txt)
# 2 workers × 2 threads each – sensible default for a CPU-bound ML workload.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 120 app:app"]
