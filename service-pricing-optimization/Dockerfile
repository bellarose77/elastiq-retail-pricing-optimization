# ELASTIQ pricing API service.
#
# This service reuses src/optimization from the monorepo rather than
# duplicating it (see app/services/pricing_service.py), so the build
# context must be the REPOSITORY ROOT, not this directory:
#
#   docker build -f service-pricing-optimization/Dockerfile -t elastiq-pricing-api .
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/data/processed:/srv/data/processed:ro" \
#     elastiq-pricing-api
#
# The volume mount keeps the served recommendations in sync with whatever
# the batch pipeline (python scripts/run_pipeline.py) most recently
# produced, without rebuilding the image. Omit it to use the data/
# snapshot baked into the image at build time instead.

FROM python:3.11-slim AS base

WORKDIR /srv

# Install this service's own dependencies (which include src/'s runtime
# dependencies -- see requirements.txt) before copying source, so the
# dependency layer is cached across source-only changes.
COPY service-pricing-optimization/requirements.txt ./service-pricing-optimization/requirements.txt
RUN pip install --no-cache-dir -r service-pricing-optimization/requirements.txt

# The shared pricing algorithm this service reuses, and the demo dataset
# it scores by default (see the volume-mount note above for production use).
COPY src/ ./src/
COPY data/ ./data/

# The service itself.
COPY service-pricing-optimization/app/ ./service-pricing-optimization/app/

WORKDIR /srv/service-pricing-optimization

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
