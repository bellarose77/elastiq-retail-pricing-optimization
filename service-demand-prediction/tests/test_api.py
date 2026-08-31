"""API tests, run against the real pipeline artifact checked into the repo
(data/processed/xgboost_next_period_forecast.csv). No mocking: this
service's whole job is serving that file's actual numbers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_items_lists_real_forecasts():
    response = client.get("/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 50  # 10 products x 5 stores in the bundled dataset

    row = next(r for r in items if r["item_id"] == "P001_S001")
    assert row["product_id"] == "P001"
    assert row["store_id"] == "S001"
    assert row["category"] == "Beverages"
    assert row["forecast_quantity"] == pytest.approx(56.960033, abs=1e-4)
    assert row["prediction_interval_lower"] == pytest.approx(43.233387, abs=1e-4)
    assert row["prediction_interval_upper"] == pytest.approx(67.94833, abs=1e-4)


def test_forecasts_known_item():
    response = client.post("/forecasts", json={"item_ids": ["P001_S001"]})
    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == []
    assert len(body["forecasts"]) == 1
    assert body["forecasts"][0]["forecast_quantity"] == pytest.approx(56.960033, abs=1e-4)


def test_forecasts_unknown_item_is_reported_not_found():
    response = client.post(
        "/forecasts", json={"item_ids": ["P001_S001", "NOT_A_REAL_ITEM"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == ["NOT_A_REAL_ITEM"]
    assert len(body["forecasts"]) == 1


def test_forecasts_rejects_empty_request():
    response = client.post("/forecasts", json={"item_ids": []})
    assert response.status_code == 422


def test_product_forecasts_aggregates_across_stores():
    response = client.get("/products/forecasts")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 10
    p001 = next(p for p in products if p["itemId"] == "P001")
    assert p001["storeCount"] == 5
    assert p001["forecastQuantity"] > 0


def test_refresh_reloads_artifact():
    response = client.post("/refresh")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refreshed"
    assert "generated_from" in body
