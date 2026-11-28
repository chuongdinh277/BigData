"""Kiểm thử API: hợp đồng response, mã lỗi, vòng đời review + nhật ký, drift theo lô.

Dùng gói mô hình DEMO huấn luyện trên dữ liệu giả (tạo trong fixture) và SQLite tạm thời.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    env = {**os.environ, "HG_TASK": "p1", "PYTHONUTF8": "1"}
    gold, split = tmp / "gold.csv", tmp / "split.json"
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, check=True,  # noqa: E731
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("tools/make_synthetic_gold.py", "--task", "p1", "--rows", "2500", "--out", str(gold))
    run("-m", "ml.common.split", "--gold", str(gold), "--out", str(split))
    run("-m", "ml.trust.run", "--gold", str(gold), "--split", str(split), "--synthetic", "--quick", "--open-test",
        "--skip-extra", "--candidates", "LogisticRegression", "--out", str(tmp / "run"), "--docs-dir", str(tmp / "results" / "p1"))
    # gói được ghi vào artifacts/models/p1/synthetic-...; trỏ MODEL_DIR tới đó
    os.environ.update({"DATABASE_URL": f"sqlite:///{tmp / 'api.db'}", "MODEL_DIR": str(ROOT / "artifacts" / "models"),
                       "RESULTS_DIR": str(tmp / "results"), "TASKS": "p1", "MODEL_SOURCE": "local"})
    for mod in [m for m in sys.modules if m.startswith("backend")]:
        del sys.modules[mod]
    from fastapi.testclient import TestClient

    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def _case(client):
    return client.get("/api/v1/cases/p1").json()


def test_health_and_tasks(client):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and "p1" in h["models_loaded"]
    t = {x["task"]: x for x in client.get("/api/v1/tasks").json()}
    assert t["p1"]["enabled"] and not t["p2"]["enabled"]


def test_schema_has_labels(client):
    s = client.get("/api/v1/schema/p1").json()
    f = {x["name"]: x for x in s["fields"]}
    assert f["bun_max"]["label"].startswith("BUN") and f["bun_max"]["unit"] == "mg/dL"


def test_predict_contract_and_review_flow(client):
    case = next(c for c in _case(client) if c["case_id"].endswith("high"))
    r = client.post("/api/v1/predict/p1", json={"case_id": case["case_id"], "features": case["features"]})
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ("task", "model_version", "prediction_time", "probability", "calibrated_probability", "risk_level",
              "review", "input_quality", "explanation", "disclaimer"):
        assert k in body
    assert body["disclaimer"] == "Research prototype – not for clinical use"
    assert all("label" in f for f in body["explanation"]["top_features"])

    # ca thiếu dữ liệu phải vào hàng đợi vì độ tin cậy thấp
    sparse = next(c for c in _case(client) if c["case_id"].endswith("sparse"))
    r = client.post("/api/v1/predict/p1", json={"case_id": sparse["case_id"], "features": sparse["features"]}).json()
    assert r["review"]["flag"] and r["review"]["review_id"]
    rid = r["review"]["review_id"]

    # vòng đời: không nhảy cóc, phải có quyết định khi đóng, mỗi bước có nhật ký
    assert client.patch(f"/api/v1/reviews/{rid}", json={"status": "REVIEWED", "reviewer": "ha"}).status_code == 409
    assert client.patch(f"/api/v1/reviews/{rid}", json={"status": "IN_REVIEW", "reviewer": "ha"}).status_code == 200
    assert client.patch(f"/api/v1/reviews/{rid}", json={"status": "REVIEWED", "reviewer": "ha"}).status_code == 422
    done = client.patch(f"/api/v1/reviews/{rid}", json={"status": "REVIEWED", "reviewer": "ha", "decision": "AGREE",
                                                         "note": "Đồng ý"}).json()
    assert [e["new_status"] for e in done["audit"]] == ["PENDING", "IN_REVIEW", "REVIEWED"]
    assert client.patch(f"/api/v1/reviews/{rid}", json={"status": "IN_REVIEW", "reviewer": "ha"}).status_code == 409


def test_errors(client):
    assert client.post("/api/v1/predict/p1", json={"case_id": "x", "features": {"khong_co": 1}}).status_code == 422
    assert client.post("/api/v1/predict/p9", json={"case_id": "x", "features": {}}).status_code == 422
    assert client.post("/api/v1/predict/p2", json={"case_id": "x", "features": {}}).status_code in (404, 503)
    assert client.get("/api/v1/reviews/999999").status_code == 404


def test_history_reports_and_drift(client):
    assert len(client.get("/api/v1/predictions?task=p1").json()) >= 2
    rep = client.get("/api/v1/reports/p1/drift").json()
    assert rep["drift"]["periods"]
    cases = [c["features"] for c in _case(client)] * 10
    d = client.post("/api/v1/drift/p1", json={"records": cases}).json()
    assert "status" in d and d["features"]
