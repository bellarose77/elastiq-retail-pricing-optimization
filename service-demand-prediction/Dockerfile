# ELASTIQ demand prediction API service.
#
# Serves the batch pipeline's forecast artifact (data/processed/) rather
# than training or running the model itself, but still needs src/config.py
# for its default data path and the data/ directory as a sibling, so the
# build context must be the REPOSITORY ROOT, not this directory:
#
#   docker build -f service-demand-prediction/Dockerfile -t elastiq-demand-api .
#   docker run --rm -p 8001:8001 \
#     -v "$(pwd)/data/processed:/srv/data/processed:ro" \
#     elastiq-demand-api
#
# The volume mount keeps served forecasts in sync with whatever the batch
# pipeline most recently produced, without rebuilding the image. Omit it
# to use the data/ snapshot baked into the image at build time.

FROM python:3.11-slim AS base

WORKDIR /srv

COPY service-demand-prediction/requirements.txt ./service-demand-prediction/requirements.txt
RUN pip install --no-cache-dir -r service-demand-prediction/requirements.txt

# src/config.py, for this service's default data path -- not the ML code
# that produced the forecast (see the module docstrings in app/services/).
COPY src/ ./src/
COPY data/ ./data/

COPY service-demand-prediction/app/ ./service-demand-prediction/app/

WORKDIR /srv/service-demand-prediction

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
