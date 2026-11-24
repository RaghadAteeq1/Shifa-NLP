"""Smoke tests for the service. They use whatever model is available (baseline works after `make baseline`)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from shifa import config as C  # noqa: E402

pytestmark = pytest.mark.skipif(not (C.BASELINE.exists() or (C.TOPIC_ONNX_INT8 / "model.onnx").exists()),
                                reason="no topic model built yet (run `make baseline`)")


@pytest.fixture(scope="module")
def client():
    from shifa.serve import app
    with TestClient(app) as c:
        yield c


def test_health_and_canaries(client):
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["canaries_green"]


def test_canary_string_is_masked(client):
    r = client.post("/v1/analyze", json={"text": "CANARY MRN-000000 0500000000"}).json()
    assert "000000" not in r["masked"] and "[MRN]" in r["masked"] and "[PHONE]" in r["masked"]


def test_batch_matches_single_and_keeps_order(client):
    texts = ["ابي اجدد وصفة بنادول", "I was charged twice for my visit", "متى تطلع نتيجة تحليل الدم حقتي", "ابي الغي موعدي"]
    single = [client.post("/v1/classify", json={"text": t}).json()["label"] for t in texts]
    for bucket in (True, False):
        batch = client.post("/v1/classify:batch", json={"texts": texts, "bucket": bucket}).json()
        assert [x["label"] for x in batch["results"]] == single


def test_validation(client):
    assert client.post("/v1/classify", json={"text": ""}).status_code == 422
    assert client.post("/v1/classify", json={"text": "   "}).status_code == 422
    assert client.post("/v1/classify:batch", json={"texts": []}).status_code == 422


def test_batch_items_are_capped(client):
    from shifa.serve import MAX_BATCH_ITEMS, MAX_TEXT_CHARS
    assert client.post("/v1/classify:batch", json={"texts": ["ok", ""]}).status_code == 422
    assert client.post("/v1/classify:batch", json={"texts": ["ok", "x" * (MAX_TEXT_CHARS + 1)]}).status_code == 422
    assert client.post("/v1/classify:batch", json={"texts": ["ok"] * (MAX_BATCH_ITEMS + 1)}).status_code == 422
    assert client.post("/v1/classify:batch", json={"texts": ["x" * MAX_TEXT_CHARS] * 4}).status_code == 200


def test_oversized_body_rejected_before_parsing(client):
    from shifa.serve import MAX_BODY_BYTES
    body = '{"texts": ["' + "x" * (MAX_BODY_BYTES + 10) + '"]}'
    r = client.post("/v1/classify:batch", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_root_redirects_to_demo(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 302) and r.headers["location"] == "/demo"
    assert client.head("/", follow_redirects=False).status_code in (307, 302)
