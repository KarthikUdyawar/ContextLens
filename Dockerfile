# Use the Python 3.12 slim base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# uv binary, pinned to content-addressable digest — tag can move, digest can't.
# Get current digest: docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.5
#   --format '{{json .Manifest}}' | jq -r .digest
COPY --from=ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim@sha256:sha256:13a15bf8da80cc7cad97711a8a2e094756a9e79feb247d0b496a8cc647fd7d3b /uv /uvx /bin/

# Deps layer first — cache hit unless pyproject/lock change
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# App code layer — changes here don't bust the deps cache
COPY ./ .
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH"

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Probe /health; python stdlib avoids adding curl to the slim image just for this.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"


# Single worker per container: /health reflects this process's model state.
# Scale horizontally (multiple containers/replicas), not via --workers,
# or /health only ever reports one of N workers' readiness.
CMD ["uvicorn", "src.app.main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]
