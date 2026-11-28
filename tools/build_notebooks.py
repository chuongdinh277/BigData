"""Sinh bộ notebook học/chạy theo từng phần của dự án (notebooks/0*.ipynb).

Mỗi notebook: (1) phần này trả lời câu chữ nào của đề, (2) chạy và đọc số liệu, (3) câu hỏi tự kiểm tra
trước buổi bảo vệ. Mặc định đọc kết quả DEMO (dữ liệu giả, chạy được trên laptop); trên EC2 đặt
REAL = True để đọc kết quả thật trong results/.

    python tools/build_notebooks.py          # sinh lại notebooks/
    make notebooks                           # chạy lại toàn bộ để kiểm tra
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = '''# --- Thiết lập chung ---------------------------------------------------------------
import json, os, sys
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

REAL = False          # True trên EC2 sau khi đã chạy `make trust TASK=...` với dữ liệu thật
TASK = "p1"           # p1 | p2 | p3

def load_results(task=TASK):
    p = ROOT / ("results/trust" if REAL else "artifacts/demo_results") / task / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}. Chạy `python tools/make_demo_bundles.py` (demo) hoặc `make trust TASK={task}` (EC2).")
    return json.loads(p.read_text(encoding="utf-8"))

VERIFIED = json.loads((ROOT / "results/data/verified_v1.json").read_text(encoding="utf-8"))
print("Nguồn:", "DỮ LIỆU THẬT" if REAL else "DEMO (dữ liệu giả — số liệu không có ý nghĩa y khoa)")'''


def nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    return n


BOOKS: dict[str, list[tuple[str, str]]] = {}

BOOKS["01_du_lieu_cohort_nhan.ipynb"] = [
    ("md", """# 01 · Dữ liệu, cohort và nhãn (phụ trách: Hùng · Hà)

**Đề:** *"Build a distributed feature-engineering pipeline for registry or EHR data"* và *"in-hospital mortality, 30-day readmission, or medium-term adverse outcomes"*.

Notebook này đọc **số liệu thật đã kiểm chứng** (`results/data/verified_v1.json`) để bạn nắm chắc từng con số: phễu cohort, ba bộ nhãn, bản chia tập.
Mã nguồn: `data_spark/cohort.py`, `data_spark/labels.py`, cấu hình `configs/hf_codes.yaml`, `configs/data.yaml`."""),
    ("code", SETUP),
    ("md", "## 1. Phễu cohort suy tim\nMỗi bước lọc loại bao nhiêu lượt nhập viện? Vì sao lại có 21 lượt bị loại ở S2?"),
    ("code", "pd.DataFrame(VERIFIED['cohort_funnel'])"),
    ("code", "VERIFIED['cohort_extra']"),
    ("md", "## 2. Ba bài toán và ca bị loại\nChú ý: quần thể mỗi bài khác nhau vì **mốc dự đoán** khác nhau (P1: giờ 24; P2/P3: lúc xuất viện)."),
    ("code", "lab = pd.DataFrame(VERIFIED['labels']); lab"),
    ("code", "pd.DataFrame({r['task']: r['excluded'] for r in VERIFIED['labels']}).T"),
    ("code", """# Kiểm tra chéo: số lượt = cohort − bị loại
cohort = 80746
for r in VERIFIED['labels']:
    ex = sum(v for v in r['excluded'].values() if isinstance(v, int))
    print(r['task'], 'cohort − loại =', cohort - ex, '| nhãn =', r['admissions'])
print('P1 còn <10 ca tử vong trước giờ 24 bị che nên lệch ≤ 9')"""),
    ("md", "## 3. 417 ca tử vong nằm ngoài P1\nGiải thích được điều này là một điểm cộng khi bảo vệ."),
    ("code", "VERIFIED['p1_deaths_outside_task']"),
    ("md", "## 4. Tỷ lệ dương theo giai đoạn (nguồn của drift nhãn)"),
    ("code", "yg = pd.DataFrame(VERIFIED['by_year_group_p1']); yg['share'] = yg['n'] / yg['n'].sum(); yg"),
    ("md", "## 5. Bản chia tập đã khóa"),
    ("code", "pd.DataFrame({k: v for k, v in VERIFIED['split_p1'].items() if isinstance(v, dict)}).T"),
    ("md", """## Câu hỏi tự kiểm tra
1. Vì sao chia theo `subject_id` chứ không theo `hadm_id`?
2. Nhãn P2 loại những ca nào, và vì sao loại "chết trong 30 ngày mà không tái nhập"?
3. `dod` của MIMIC có giới hạn gì với P3?
4. Vì sao cohort chọn bằng mã ICD lúc xuất viện lại là vấn đề với P1? Phân tích C2 trong báo cáo trả lời thế nào?

**Chạy thật trên EC2:** `make etl TASK=p1 VERSION=v1_1`, rồi `make etl TASK=p2`, `make etl TASK=p3`."""),
]

BOOKS["02_dac_trung_chat_luong.ipynb"] = [
    ("md", """# 02 · Đặc trưng, dữ liệu thiếu và kiểm tra chất lượng (phụ trách: Khoẻ · Chiến)

**Đề:** *"distributed feature-engineering"* + *"missing-data sensitivity"*.
Mã nguồn: `data_spark/features.py`, `data_spark/features_discharge.py`, `data_spark/quality.py`, `contracts/gold_schema_*.json`."""),
    ("code", SETUP),
    ("md", "## 1. Hợp đồng Gold: 114 đặc trưng P1, 146 đặc trưng P2/P3"),
    ("code", """for t in ("p1", "p2"):
    s = json.loads((ROOT / f"contracts/gold_schema_{t}.json").read_text(encoding="utf-8"))
    groups = {}
    for g, spec in s["feature_columns"].items():
        groups[g] = len(spec) if isinstance(spec, list) else len(spec["items"]) * len(spec["suffixes"]) + len(spec.get("extra", []))
    print(t, groups, "tổng", sum(groups.values()), "| cột cấm:", s["forbidden_columns"])"""),
    ("md", "## 2. Tỷ lệ thiếu thật (P1) — thiếu không ngẫu nhiên"),
    ("code", "pd.Series({k: v for k, v in VERIFIED['missing_rate_p1'].items() if not isinstance(v, list)}).sort_values(ascending=False).to_frame('tỷ lệ thiếu')"),
    ("md", "## 3. Lỗi đọc dữ liệu đã sửa ở Dataset v1.1\n`\"<0.01\"` là kết quả **bình thường dưới ngưỡng phát hiện**, không phải thiếu."),
    ("code", "pd.DataFrame(VERIFIED['data_issues_fixed_in_v1_1'])"),
    ("code", """# Hàm đọc giá trị bị chặn (Spark). Chạy được nếu máy có Java 17.
import re
from data_spark.features import CENSORED_RE
for v in ["<0.01", "< 0.03", ">60", "0.25", "NEG"]:
    m = re.match(CENSORED_RE, v)
    print(f"{v!r:10} → {float(m.group(2)) if m else 'giữ nguyên/thiếu'}")"""),
    ("md", "## 4. Chống rò rỉ: những gì bị cấm và vì sao"),
    ("code", """s1 = json.loads((ROOT / "contracts/gold_schema_p1.json").read_text(encoding="utf-8"))
print("Cấm ở P1:", s1["forbidden_columns"])
print("P1 dùng xét nghiệm trong [admittime − 6h, admittime + 24h], chỉ khi storetime ≤ mốc dự đoán.")"""),
    ("md", """## Câu hỏi tự kiểm tra
1. `charttime` và `storetime` khác nhau thế nào? Vì sao lọc theo `storetime`?
2. Vì sao ICD của lượt hiện tại bị cấm ở P1 nhưng được dùng ở P2/P3?
3. Cờ `_missing` có ý nghĩa gì với mô hình cây? Có rủi ro gì (gợi ý: phân tích C5)?
4. Tỷ lệ thiếu troponin trước và sau v1.1 là bao nhiêu? (đọc `results/data/features_result_full.md` sau khi chạy lại)

**Chạy thật:** `make etl TASK=p1 VERSION=v1_1` rồi `make quality TASK=p1 VERSION=v1_1`."""),
]

BOOKS["03_mo_hinh_so_sanh.ipynb"] = [
    ("md", """# 03 · Mô hình và so sánh (phụ trách: Hùng · Khoẻ · Chiến)

**Đề:** *"Train and compare suitable baseline prediction models"*, *"Evaluate ROC-AUC, PR-AUC, F1"*.
Mã nguồn: `ml/models/*.py`, `ml/baselines.py`, `ml/trust/run.py` (CV 5 fold, so sánh ghép cặp)."""),
    ("code", SETUP),
    ("md", "## 1. Kết quả GĐ4 thật trên validation (13 lần chạy)\nLạc quan vì Optuna chọn tham số trên chính validation → dùng CV ở mục 2 để chọn."),
    ("code", """g4 = pd.concat([pd.read_csv(p) for p in (ROOT / "results/models").glob("*.csv")], ignore_index=True)
g4["pr_auc_lift"] = g4["pr_auc"] / g4["positive_rate"]
g4.sort_values("pr_auc", ascending=False)[["case_id", "model", "member", "pr_auc", "pr_auc_lift", "roc_auc", "f1", "brier", "seconds"]]"""),
    ("md", "## 2. CV 5 fold theo bệnh nhân + so sánh có ý nghĩa thống kê"),
    ("code", """R = load_results()
pd.DataFrame({k: {m: v[m] for m in ("pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "brier_mean", "seconds")} for k, v in R["cv"].items()}).T"""),
    ("code", """mc = R["model_comparison"]
pd.DataFrame({n: {"ΔPR-AUC": v["pr_auc"]["diff"], "CI95": v["pr_auc"]["ci95"], "có ý nghĩa": v["pr_auc"]["significant"]} for n, v in mc["vs"].items()}).T"""),
    ("code", "print(R.get('recommendation', 'Mô hình tốt nhất là LR'))"),
    ("md", "## 3. So với mốc sàn và thang điểm lâm sàng"),
    ("code", "pd.DataFrame(R['baselines_dev'])"),
    ("md", "## 4. Learning curve — dữ liệu lớn có giúp không?"),
    ("code", "pd.DataFrame(R.get('learning_curve', []))"),
    ("md", """## Câu hỏi tự kiểm tra
1. Vì sao PR-AUC 0.32 ở P1 vẫn là tốt? (gợi ý: lift so với tỷ lệ nền 4.86%)
2. Vì sao đặt trọng số lớp làm Brier tăng từ 0.039 lên 0.083–0.158?
3. Luật chọn mô hình đặt trước là gì? Kết quả bootstrap ghép cặp nói gì?
4. GWTG-HF bản đầy đủ chỉ tính được trên bao nhiêu % ca, vì sao?

**Chạy thật:** `make train-lr train-xgb train-lgbm TASK=p1` (tùy chọn) rồi `make trust TASK=p1`."""),
]

BOOKS["04_hieu_chuan_review.ipynb"] = [
    ("md", """# 04 · Hiệu chuẩn, ngưỡng, lợi ích lâm sàng và quy trình review (phụ trách: Hà)

**Đề:** *"Evaluate … calibration"* + *"Introduce human review for high-risk or low-confidence predictions"*.
Mã nguồn: `trustworthy_ai/calibration.py`, `trustworthy_ai/utility.py`, `trustworthy_ai/review.py`, `backend/app/routers/reviews.py`."""),
    ("code", SETUP),
    ("code", "R = load_results(); ch = R['chosen_model']; print('Mô hình:', ch, '| hiệu chuẩn:', R['calibration'][ch]['method'])"),
    ("md", "## 1. So sánh phương pháp hiệu chuẩn (chọn theo Brier trên dự đoán ngoài fold)"),
    ("code", "pd.DataFrame(R['calibration'][ch]['comparison']).T"),
    ("md", "## 2. Ngưỡng LOW / MEDIUM / HIGH"),
    ("code", "{k: v for k, v in R['thresholds'].items() if not isinstance(v, dict)}, R['dev_risk_level_mortality']"),
    ("md", "## 3. Decision Curve Analysis và bảng điểm cắt"),
    ("code", "print(R['dca_dev']['summary']); pd.DataFrame(R['operating_dev'])"),
    ("md", "## 4. Luật review: khối lượng hàng đợi và hiệu quả"),
    ("code", "rv = R['review_dev']; print({k: rv[k] for k in ('flagged_share', 'deaths_captured_share', 'death_rate_in_queue', 'death_rate_not_flagged')}); pd.DataFrame(rv['by_reason']).T"),
    ("md", "## 5. Bằng chứng luật \"độ tin cậy thấp\" có tác dụng + risk–coverage"),
    ("code", "R['confidence_validation_dev']"),
    ("code", "pd.DataFrame(R['risk_coverage_dev'])"),
    ("md", "## 6. Quy trình trên hệ thống (gọi API thật bằng TestClient)"),
    ("code", """os.environ.setdefault("DATABASE_URL", "sqlite:///artifacts/notebook_demo.db")
from fastapi.testclient import TestClient
from backend.app.main import app
with TestClient(app) as c:
    case = next(x for x in c.get(f"/api/v1/cases/{TASK}").json() if x["case_id"].endswith("sparse"))
    r = c.post(f"/api/v1/predict/{TASK}", json={"case_id": case["case_id"], "features": case["features"]}).json()
    print("Cờ:", r["review"])
    rid = r["review"]["review_id"]
    print(c.patch(f"/api/v1/reviews/{rid}", json={"status": "REVIEWED", "reviewer": "ha"}).status_code, "← nhảy cóc bị chặn (409)")
    c.patch(f"/api/v1/reviews/{rid}", json={"status": "IN_REVIEW", "reviewer": "ha"})
    done = c.patch(f"/api/v1/reviews/{rid}", json={"status": "REVIEWED", "reviewer": "ha", "decision": "AGREE"}).json()
    print([(e["old_status"], e["new_status"], e["actor"]) for e in done["audit"]])"""),
    ("md", """## Câu hỏi tự kiểm tra
1. O/E = 0.8 nghĩa là gì? Slope < 1 nghĩa là gì?
2. Vì sao ngưỡng HIGH chọn theo F1 còn MEDIUM theo độ nhạy ≥ 80%?
3. Lợi ích ròng (net benefit) tính thế nào? Khi nào mô hình "vô ích"?
4. 7 lý do gắn cờ là gì? Lý do nào là "nguy cơ cao", lý do nào là "độ tin cậy thấp"?"""),
]

BOOKS["05_giai_thich_shap.ipynb"] = [
    ("md", """# 05 · Giải thích mô hình (phụ trách: Khoẻ)

**Đề:** *"Provide patient-level model explanations"* (có trong tên đề: *"with Explainability"*).
Mã nguồn: `trustworthy_ai/explain.py`, `trustworthy_ai/labels.py`, `trustworthy_ai/bundle.py`."""),
    ("code", SETUP),
    ("code", "R = load_results(); sh = R['shap']; print(sh['method'], '| sai số cộng tính:', sh['additivity_max_error'], '| ổn định:', sh['fold_stability'])"),
    ("md", "## 1. Toàn cục: top đặc trưng và theo nhóm thông tin"),
    ("code", "pd.DataFrame(sh['top_features']).head(15)"),
    ("code", "pd.DataFrame(sh['families']).head(10)"),
    ("md", "## 2. Hợp lý lâm sàng và độ trung thực"),
    ("code", "pd.DataFrame(sh['plausibility'])"),
    ("code", "pd.DataFrame(sh['deletion_curve'])"),
    ("md", "## 3. Giải thích cho MỘT bệnh nhân (ca demo, nạp gói mô hình như API)"),
    ("code", """from trustworthy_ai.bundle import HeartGuardModel
from trustworthy_ai.labels import describe
from backend.app.demo_cases import demo_cases
d = sorted((ROOT / "artifacts/models" / TASK).glob("*/SHA256SUMS"))[-1].parent
m = HeartGuardModel(d)
case = demo_cases(TASK)[2]
r = m.predict(case["features"], top_k=8)[0]
print(case["title"], "→ xác suất đã hiệu chuẩn", r["calibrated_probability"], r["risk_level"])
pd.DataFrame([{**f, **describe(f["feature"])} for f in r["explanation"]["top_features"]])"""),
    ("md", "## 4. Lý do của mô hình thay đổi theo thời gian"),
    ("code", "R['explanation_drift']"),
    ("md", """## Câu hỏi tự kiểm tra
1. SHAP ở thang nào? Vì sao không cộng thẳng ra phần trăm xác suất đã hiệu chuẩn?
2. Vì sao "thiếu sinh hiệu làm giảm nguy cơ" không có nghĩa là thiếu dữ liệu tốt cho sức khỏe?
3. Deletion curve chứng minh điều gì? Tỷ số bao nhiêu là tốt?
4. SHAP có chứng minh quan hệ nhân quả không?"""),
]

BOOKS["06_thieu_du_lieu_nhom_phu.ipynb"] = [
    ("md", """# 06 · Độ nhạy với dữ liệu thiếu và nhất quán giữa nhóm bệnh nhân (phụ trách: Chiến · Chưởng)

**Đề:** *"missing-data sensitivity"*, *"Assess performance consistency across relevant patient subgroups"*.
Mã nguồn: `trustworthy_ai/robustness.py`, `trustworthy_ai/subgroup.py`."""),
    ("code", SETUP),
    ("code", "R = load_results()"),
    ("md", "## 1. Che dần thông tin: PR-AUC và O/E giảm thế nào"),
    ("code", "rob = pd.DataFrame(R.get('robustness_test') or R['robustness_dev']); rob.pivot_table(index='scenario', columns='rate', values='pr_auc')"),
    ("code", "rob.pivot_table(index='scenario', columns='rate', values='oe_ratio')"),
    ("md", "## 2. Hiệu năng theo nhóm (có CI) và chênh lệch công bằng"),
    ("code", "sg = pd.DataFrame(R.get('subgroup_test') or R['subgroup_dev']); sg[['attribute','group','n','n_pos','roc_auc','pr_auc_lift','oe_ratio','recall']]"),
    ("code", "pd.DataFrame(R.get('fairness_test') or R['fairness_dev'])"),
    ("code", "pd.Series(R['subgroup_relevance']).to_frame('vì sao relevant')"),
    ("md", "## 3. Phân tích độ nhạy của thiết kế (C0–C6)"),
    ("code", "pd.DataFrame(R['sensitivity'])"),
    ("md", """## Câu hỏi tự kiểm tra
1. Vì sao so PR-AUC giữa các nhóm phải dùng lift?
2. Nhóm nào có O/E xa 1 nhất? Điều đó nguy hiểm thế nào?
3. Kịch bản thiếu nào làm mô hình giảm nhanh nhất? Liên hệ với luật gắn cờ HIGH_MISSINGNESS.
4. Equal opportunity (recall gap) là gì?"""),
]

BOOKS["07_drift_bao_tri.ipynb"] = [
    ("md", """# 07 · Drift dữ liệu, drift mô hình và chiến lược bảo trì (phụ trách: Chưởng)

**Đề:** *"Detect data drift and model drift"* (có trong tên đề: *"Drift Monitoring"*), prototype *"evaluated by … drift metrics"*.
Mã nguồn: `trustworthy_ai/drift.py`, `backend/app/routers/system.py` (`POST /drift/{task}`)."""),
    ("code", SETUP),
    ("code", "R = load_results(); dr = R['drift']; print('Tham chiếu:', dr['reference_period'], '| ngưỡng null:', dr['null_thresholds'])"),
    ("md", "## 1. Bảng drift metrics chính thức theo lô"),
    ("code", """pd.DataFrame([{"period": r["period"], "n": r["n"], "tỷ lệ dương": r["positive_rate"], "O/E": r["oe_ratio"],
               "% PSI>0.25": r["share_features_major"], "vượt null": r.get("n_features_over_null"),
               "AUC miền": (r.get("domain_classifier") or {}).get("auc"), "PSI điểm": r["score_psi"], "trạng thái": r["status"]}
              for r in dr["periods"]])"""),
    ("md", "## 2. Drift mô hình: huấn luyện giai đoạn cũ → đánh giá giai đoạn mới"),
    ("code", "pd.DataFrame(R['temporal_validation']['rows'])"),
    ("md", "## 3. Ba chiến lược bảo trì"),
    ("code", "mt = pd.DataFrame(R['maintenance']['rows']); mt.pivot_table(index='period', columns='strategy', values=['oe_ratio', 'pr_auc'])"),
    ("md", "## 4. Drift mã hóa ICD-9 → ICD-10"),
    ("code", "R['coding_drift'].get('largest_version_gaps'), R['coding_drift'].get('icd_version_mix_by_period')"),
    ("md", "## 5. Giám sát một lô mới (không cần nhãn) qua gói mô hình"),
    ("code", """from trustworthy_ai.bundle import HeartGuardModel
from backend.app.demo_cases import demo_cases
m = HeartGuardModel(sorted((ROOT / "artifacts/models" / TASK).glob("*/SHA256SUMS"))[-1].parent)
batch = [dict(c["features"]) for c in demo_cases(TASK)] * 15
for b in batch:
    for k in list(b):
        if k.startswith("creatinine_") and b[k] is not None and not k.endswith("_missing"): b[k] *= 1.5
rep = m.drift_report(batch); print(rep["status"], "| PSI điểm:", rep.get("score_psi")); pd.DataFrame(rep["features"]).head(8)"""),
    ("md", """## Câu hỏi tự kiểm tra
1. Drift dữ liệu khác drift mô hình thế nào? Cái nào đo được ngay, cái nào cần chờ nhãn (P2: 30 ngày, P3: 90 ngày)?
2. Vì sao ngưỡng PSI 0.25 quy ước chưa đủ? Ngưỡng null tính thế nào?
3. AUC của bộ phân loại miền 0.5 và 0.8 nghĩa là gì?
4. Khi tỷ lệ tử vong tăng từ 3.79% lên 8.01%, chiến lược nào khôi phục O/E rẻ nhất?"""),
]

BOOKS["08_benchmark_big_data.ipynb"] = [
    ("md", """# 08 · Bằng chứng Big Data: phễu dữ liệu, pandas vs Spark, khả năng mở rộng (phụ trách: Hùng)

**Đề:** *"distributed feature-engineering pipeline"*. Mã nguồn: `data_spark/scaling.py`, `deploy/spark/`.
Kết quả thật xuất hiện sau khi chạy `make bench` trên EC2."""),
    ("code", SETUP),
    ("md", "## 1. Benchmark cũ (v1) và vì sao phải làm lại"),
    ("code", "b = VERIFIED['benchmark_v1']; print(b['limitation']); pd.DataFrame(b['seconds'], index=['1%', '5%', '10%']).T"),
    ("md", "## 2. Kết quả mới (sau `make bench`)"),
    ("code", """bench = ROOT / "results/benchmark"
for name in ("funnel.json", "pandas_full_load.json"):
    p = bench / name
    print(name, "→", json.loads(p.read_text(encoding="utf-8")) if p.exists() else "chưa chạy")
if (bench / "scaling.csv").exists():
    s = pd.read_csv(bench / "scaling.csv"); s = s[s["ok"]]
    t = s.groupby(["mode", "cores"])["seconds"].median().unstack(0); t["speedup_strong"] = t["strong"].iloc[0] / t["strong"]
    display(t)
else:
    print("Chưa có scaling.csv — chạy: make bench-strong bench-weak bench-ablation bench-report")"""),
    ("md", """## Câu hỏi tự kiểm tra
1. Vì sao file `.csv.gz` chỉ đọc được bằng 1 task? Bước nào trong pipeline giải quyết?
2. Strong scaling và weak scaling khác nhau thế nào? E(n) giảm có phải là lỗi không (gợi ý: Amdahl)?
3. Broadcast join dùng ở đâu, tại sao? Ablation `no_auto_broadcast` cho thấy gì?
4. Chạy nhiều core trên MỘT máy khác cụm nhiều máy thế nào? (`deploy/spark/cluster.sh`)"""),
]

BOOKS["09_he_thong_api.ipynb"] = [
    ("md", """# 09 · Hệ thống đầu–cuối: gói mô hình → API → CSDL (phụ trách: Chưởng)

**Đề:** *"Experimental prototype system … supporting mortality/readmission/outcome prediction with explainability, calibration, subgroup analysis, and drift monitoring"*.
Mã nguồn: `trustworthy_ai/bundle.py`, `backend/app/`, `deploy/docker-compose.yml`."""),
    ("code", SETUP),
    ("code", """os.environ.setdefault("DATABASE_URL", "sqlite:///artifacts/notebook_demo.db")
from fastapi.testclient import TestClient
from backend.app.main import app
c = TestClient(app); c.__enter__()
pd.DataFrame(c.get("/api/v1/tasks").json())[["task", "name", "enabled", "model_version"]]"""),
    ("code", """rows = []
for t in ("p1", "p2", "p3"):
    for case in c.get(f"/api/v1/cases/{t}").json():
        r = c.post(f"/api/v1/predict/{t}", json={"case_id": case["case_id"], "features": case["features"]}).json()
        rows.append({"task": t, "ca": case["title"], "xác suất": r["calibrated_probability"], "mức": r["risk_level"],
                     "review": r["review"]["flag"], "lý do": ", ".join(r["review"]["reasons"])[:80]})
pd.DataFrame(rows)"""),
    ("code", "c.get('/api/v1/admin/system').json()"),
    ("code", "pd.DataFrame(c.get('/api/v1/admin/models').json())[['task', 'version', 'model', 'calibration', 'cv_pr_auc', 'test_pr_auc', 'serving']]"),
    ("md", """## Câu hỏi tự kiểm tra
1. Gói mô hình gồm những file nào? SHA256SUMS dùng để làm gì?
2. Vì sao API không bao giờ huấn luyện lại mô hình?
3. Một dự đoán trả về những trường nào để đáp ứng "trustworthy"?
4. Triển khai trên EC2 gồm những container nào, cổng nào mở ra ngoài?"""),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in BOOKS.items():
        nbf.write(nb(cells), OUT / name)
        print("đã ghi", OUT / name)


if __name__ == "__main__":
    main()
