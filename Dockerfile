# VHAGAR fire API + operations console, container image for a hosted deploy.
#
# Starts from a prebuilt, self-contained snapshot committed under serve/demo,
# so the service comes up instantly: no raw GOES parquet, no 45 s clustering.
# The console renders on Mapbox GL; set VHAGAR_MAPBOX_TOKEN in the host's
# environment and add the deployed origin to that token's allowed URLs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VHAGAR_FROZEN=1

WORKDIR /app

# Runtime deps: the serving stack plus scikit-learn for the /v1/danger models.
# The vhagar package is imported off ./src by the API (it inserts src on the
# path), so we only need its light core deps here, not the geo / torch extras.
RUN pip install \
    "fastapi" "uvicorn[standard]" "pandas>=2.2" "pyarrow>=16.0" "numpy>=1.26" \
    "scipy" "scikit-learn>=1.5" "pydantic>=2.7" "pyyaml>=6.0" "typer>=0.12" "rich>=13.7"

# Only what the service needs at runtime.
COPY src/ ./src/
COPY serve/ ./serve/
COPY vhagar_console.html ./vhagar_console.html
COPY brand/ ./brand/

EXPOSE 8000

# Render (and most hosts) inject $PORT. Default to 8000 for a plain `docker run`.
CMD ["sh", "-c", "uvicorn serve.vhagar_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
