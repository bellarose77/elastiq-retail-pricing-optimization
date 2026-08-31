"""API tests, run against the real pipeline artifacts checked into the repo
(data/processed/*.csv). No mocking: this service's whole job is to score
real evidence, so a test against synthetic stand-ins wouldn't catch a
schema drift between this service and src/optimization's actual output.
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


def test_items_lists_real_decision_units():
    response = client.get("/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 50  # 10 products x 5 stores in the bundled dataset
    assert {"P001_S001", "P010_S005"} <= {row["item_id"] for row in items}

    row = next(r for r in items if r["item_id"] == "P001_S001")
    assert row["product_id"] == "P001"
    assert row["store_id"] == "S001"
    assert row["elasticity_source"] == "product_iv"
    assert row["elasticity_is_causal"] is True


def test_recommendations_known_item_matches_batch_pipeline():
    response = client.post(
        "/recommendations", json={"item_ids": ["P001_S001"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == []
    assert len(body["recommendations"]) == 1

    rec = body["recommendations"][0]
    assert rec["item_id"] == "P001_S001"
    assert rec["status"] == "success"
    assert rec["recommendation_action"] == "increase_price"
    assert rec["recommended_price"] == pytest.approx(4.99, abs=1e-6)
    assert rec["price_change_rate"] == pytest.approx(0.08242950108459866, abs=1e-6)
    assert rec["elasticity_source"] == "product_iv"
    assert rec["elasticity_is_causal"] is True


def test_recommendations_unknown_item_is_reported_not_found():
    response = client.post(
        "/recommendations",
        json={"item_ids": ["P001_S001", "NOT_A_REAL_ITEM"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["not_found"] == ["NOT_A_REAL_ITEM"]
    assert len(body["recommendations"]) == 1


def test_recommendations_rejects_empty_request():
    response = client.post("/recommendations", json={"item_ids": []})
    assert response.status_code == 422


def test_products_matches_bundled_demo_data_shape():
    response = client.get("/products")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 10

    p001 = next(p for p in products if p["itemId"] == "P001")
    assert p001["productId"] == "P001"
    assert p001["category"] == "Beverages"
    assert p001["currentPrice"] == pytest.approx(4.38, abs=1e-6)
    assert p001["unitCost"] == pytest.approx(2.13, abs=1e-6)
    assert p001["elasticitySource"] == "product_iv"
    assert p001["elasticityIsCausal"] is True
    assert p001["dataAsOfDate"] == "2025-01-01"


def test_product_recommendations_aggregates_across_stores():
    response = client.get("/products/recommendations")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 10
    by_id = {p["itemId"]: p for p in products}

    # P003 prices successfully in every one of its 5 stores.
    p003 = by_id["P003"]
    assert p003["status"] == "success"
    assert p003["storeCount"] == 5
    assert p003["recommendedPrice"] > 0
    assert p003["expectedQuantity"] > 0

    # P001 has a mix of statuses across its stores (one store's grid
    # search finds no feasible price) -- the aggregate must say so
    # rather than silently picking one store's status.
    p001 = by_id["P001"]
    assert p001["status"] == "partial"
    assert p001["storeCount"] == 5


def test_refresh_reloads_artifacts():
    response = client.post("/refresh")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refreshed"
    assert "generated_from" in body
