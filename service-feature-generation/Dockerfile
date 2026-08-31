# ELASTIQ feature generation API service.
#
# Serves the batch pipeline's market-intelligence feature artifact
# (data/processed/) rather than running retrieval/extraction itself, but
# still needs src/config.py for its default data path and the data/
# directory as a sibling, so the build context must be the REPOSITORY
# ROOT, not this directory:
#
#   docker build -f service-feature-generation/Dockerfile -t elastiq-feature-api .
#   docker run --rm -p 8002:8002 \
#     -v "$(pwd)/data/processed:/srv/data/processed:ro" \
#     elastiq-feature-api
#
# The volume mount keeps served features in sync with whatever the batch
# pipeline most recently produced, without rebuilding the image. Omit it
# to use the data/ snapshot baked into the image at build time.

FROM python:3.11-slim AS base

WORKDIR /srv

COPY service-feature-generation/requirements.txt ./service-feature-generation/requirements.txt
RUN pip install --no-cache-dir -r service-feature-generation/requirements.txt

# src/config.py, for this service's default data path -- not the
# retrieval/feature-extraction code that produced the artifact (see the
# module docstrings in app/services/).
COPY src/ ./src/
COPY data/ ./data/

COPY service-feature-generation/app/ ./service-feature-generation/app/

WORKDIR /srv/service-feature-generation

EXPOSE 8002

# Most container hosts (Render, Railway, Cloud Run...) assign the listen
# port at runtime via $PORT and route to whatever that is; this defaults
# to 8002 so `docker run -p 8002:8002` above still works unchanged when
# $PORT isn't set (e.g. plain local `docker run`).
ENV PORT=8002

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8002)}/health', timeout=2)" || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
