"""Giai đoạn 2 — Khoẻ: tạo bảng đặc trưng Gold P1 (mốc dự đoán = admittime + 24h).

Một dòng / hadm_id. Cột và kiểu lấy từ contracts/gold_schema_p1.json (nguồn sự thật).
Chạy:
    python -m data_spark.features --fmt silver --version v1
    python -m data_spark.features --fmt silver --sample-frac 0.01      # thử 1% bệnh nhân
"""

from __future__ import annotations

import argparse
import json
import re
import time
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from data_spark.common.io import REPO_ROOT, icd_prefix_condition, load_yaml, read_table
from data_spark.common.session import get_spark

SCHEMA_PATH = REPO_ROOT / "contracts/gold_schema_p1.json"
PRED_HOURS = 24  # prediction_time = admittime + 24h
LAB_BEFORE_HOURS = 6  # xét nghiệm lấy từ admittime - 6h
OMR_LOOKBACK_DAYS = 365
LBS_TO_KG = 0.45359237  # omr_weight_last được quy đổi lb -> kg
SPARK_TYPES = {
    "long": "long",
    "int": "int",
    "double": "double",
    "string": "string",
    "timestamp": "timestamp",
}
OMR_RANGES = {
    "bmi": (10, 100),
    "weight_kg": (20, 400),
    "sbp": (40, 300),
    "egfr": (0, 200),
}


def _or(conds):
    return reduce(lambda a, b: a | b, conds) if conds else F.lit(False)


def _num(col):
    """Chuỗi số thuần -> double; chữ như '>60' -> NULL (không parse bừa)."""
    return F.when(col.rlike(r"^\s*-?\d+(\.\d+)?\s*$"), F.trim(col).cast("double"))


# --------------------------------------------------------------------------- kiểm tra itemid
def verify_items(spark, root, fmt, items) -> list[str]:
    """Đối chiếu itemid trong feature_items.yaml với d_labitems/d_items của dữ liệu thật."""
    problems = []
    dl = {
        r["itemid"]: r
        for r in read_table(spark, root, "d_labitems", fmt)
        .select("itemid", "label", "fluid")
        .collect()
    }
    for name, c in items["labs"].items():
        for iid in c["itemids"]:
            r = dl.get(iid)
            if r is None:
                problems.append(f"lab {name}: itemid {iid} không có trong d_labitems")
            elif not re.search(c["label_regex"], r["label"] or ""):
                problems.append(
                    f"lab {name}: itemid {iid} có label '{r['label']}' không khớp {c['label_regex']}"
                )
            elif c.get("fluid") and r["fluid"] != c["fluid"]:
                problems.append(
                    f"lab {name}: itemid {iid} fluid '{r['fluid']}' != {c['fluid']}"
                )
    di = {
        r["itemid"]: r["label"]
        for r in read_table(spark, root, "d_items", fmt)
        .select("itemid", "label")
        .collect()
    }
    for name, c in items["vitals"].items():
        for iid in c["itemids"]:
            if iid not in di:
                problems.append(f"vital {name}: itemid {iid} không có trong d_items")
            elif not re.search(c["label_regex"], di[iid] or ""):
                problems.append(
                    f"vital {name}: itemid {iid} có label '{di[iid]}' không khớp {c['label_regex']}"
                )
    return problems


# --------------------------------------------------------------------------- quần thể nền
def build_base(
    cohort: DataFrame,
    sample_frac=None,
    labels: DataFrame | None = None,
    label_col="label",
) -> DataFrame:
    b = cohort.withColumn(
        "prediction_time", F.expr(f"admittime + INTERVAL {PRED_HOURS} HOURS")
    )
    # Quy tắc P1: còn nằm viện và chưa tử vong tại mốc dự đoán
    b = b.filter(
        (F.col("admittime") < F.col("prediction_time"))
        & (F.col("prediction_time") < F.col("dischtime"))
        & (
            F.col("deathtime").isNull()
            | (F.col("deathtime") > F.col("prediction_time"))
        )
    )
    if sample_frac:
        b = b.filter(
            F.pmod(F.hash("subject_id"), F.lit(10000)) < int(sample_frac * 10000)
        )
    if labels is not None:  # nhãn của Hà: giới hạn đúng tập P1 và lấy nhãn từ đó
        lab = labels.select(
            "hadm_id", F.col(label_col).cast("int").alias("label_mortality")
        )
        b = b.join(lab, "hadm_id", "inner")
    else:
        b = b.withColumn("label_mortality", F.col("hospital_expire_flag").cast("int"))
    return b


# --------------------------------------------------------------------------- nhân khẩu học + subgroup
def demographics(base: DataFrame, icustays: DataFrame, dx: DataFrame) -> DataFrame:
    t = F.upper(F.trim(F.col("admission_type")))
    type_grp = (
        F.when(t.contains("OBSERVATION"), "OBSERVATION")
        .when(t.isin("EW EMER.", "DIRECT EMER.", "EMERGENCY"), "EMERGENCY")
        .when(t.contains("URGENT"), "URGENT")
        .when(t.isin("ELECTIVE", "SURGICAL SAME DAY ADMISSION"), "ELECTIVE")
        .otherwise("UNKNOWN")
    )
    loc = F.upper(F.trim(F.col("admission_location")))
    loc_grp = (
        F.when(loc.contains("EMERGENCY ROOM"), "EMERGENCY_ROOM")
        .when(loc.contains("TRANSFER"), "TRANSFER")
        .when(loc.contains("REFERRAL"), "REFERRAL")
        .when(loc.isNull() | loc.isin("", "INFORMATION NOT AVAILABLE"), "UNKNOWN")
        .otherwise("OTHER")
    )
    r = F.upper(F.coalesce(F.col("race"), F.lit("")))
    race_grp = (
        F.when(r.contains("HISPANIC") | r.contains("LATINO"), "HISPANIC")
        .when(r.contains("WHITE"), "WHITE")
        .when(r.contains("BLACK"), "BLACK")
        .when(r.contains("ASIAN"), "ASIAN")
        .when(
            r.isin("", "UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"),
            "UNKNOWN",
        )
        .otherwise("OTHER")
    )

    icu = (
        base.select("hadm_id", "admittime", "prediction_time")
        .join(icustays.select("hadm_id", "intime", "outtime"), "hadm_id")
        .filter(
            (F.col("intime") <= F.col("prediction_time"))
            & (F.col("outtime") >= F.col("admittime"))
        )
        .select("hadm_id")
        .distinct()
        .withColumn("in_icu_24h", F.lit(1))
    )

    ver = (
        dx.groupBy("hadm_id")
        .agg(
            F.countDistinct("icd_version").alias("nv"), F.min("icd_version").alias("v0")
        )
        .select(
            "hadm_id",
            F.when(F.col("nv") > 1, "MIXED")
            .when(F.col("v0") == 9, "ICD9")
            .otherwise("ICD10")
            .alias("icd_version_group"),
        )
    )

    return (
        base.withColumn("gender_male", F.when(F.col("gender") == "M", 1).otherwise(0))
        .withColumn("admission_type_group", type_grp)
        .withColumn("admission_location_group", loc_grp)
        .withColumn("came_from_ed", F.col("edregtime").isNotNull().cast("int"))
        .withColumn("race_group", race_grp)
        .join(icu, "hadm_id", "left")
        .withColumn("in_icu_24h", F.coalesce(F.col("in_icu_24h"), F.lit(0)))
        .join(ver, "hadm_id", "left")
        .withColumn(
            "icd_version_group", F.coalesce(F.col("icd_version_group"), F.lit("ICD10"))
        )
    )


# --------------------------------------------------------------------------- tiền sử
def history(
    base: DataFrame, adm_all: DataFrame, dx: DataFrame, hf_codes: dict, comorb: dict
) -> DataFrame:
    """Chỉ dùng admission có dischtime < admittime hiện tại (L7)."""
    hf_adm = (
        dx.filter(
            icd_prefix_condition(F.col("icd_code"), F.col("icd_version"), hf_codes)
        )
        .select(F.col("hadm_id").alias("p_hadm"))
        .distinct()
        .withColumn("is_hf", F.lit(1))
    )
    pairs = (
        base.select("hadm_id", "subject_id", "admittime")
        .join(
            adm_all.select(
                "subject_id",
                F.col("hadm_id").alias("p_hadm"),
                F.col("dischtime").alias("p_disch"),
            ),
            "subject_id",
        )
        .filter(F.col("p_disch").isNotNull() & (F.col("p_disch") < F.col("admittime")))
        .join(hf_adm, "p_hadm", "left")
    )
    gap_days = (F.unix_timestamp("admittime") - F.unix_timestamp("p_disch")) / 86400.0
    agg = pairs.groupBy("hadm_id").agg(
        F.count(F.lit(1)).alias("n_prior_all"),
        F.sum(F.when(gap_days <= 365, 1).otherwise(0)).alias("n_prior_adm_365d"),
        F.sum(F.coalesce(F.col("is_hf"), F.lit(0))).alias("n_prior_hf_adm"),
        F.min(gap_days).alias("days_since_last_discharge"),
    )
    pdx = pairs.select("hadm_id", "p_hadm").join(
        dx.select(F.col("hadm_id").alias("p_hadm"), "icd_code", "icd_version"), "p_hadm"
    )
    cm = pdx.groupBy("hadm_id").agg(
        *[
            F.max(
                icd_prefix_condition(
                    F.col("icd_code"), F.col("icd_version"), codes
                ).cast("int")
            ).alias(f"hx_{name}")
            for name, codes in comorb.items()
        ]
    )
    out = (
        base.select("hadm_id").join(agg, "hadm_id", "left").join(cm, "hadm_id", "left")
    )
    for c in ["n_prior_all", "n_prior_adm_365d", "n_prior_hf_adm"] + [
        f"hx_{n}" for n in comorb
    ]:
        out = out.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    return out.withColumn(
        "first_admission", (F.col("n_prior_all") == 0).cast("int")
    ).drop("n_prior_all")


def omr_features(base: DataFrame, omr: DataFrame) -> DataFrame:
    """Giá trị OMR gần nhất, chartdate < ngày nhập viện, trong 365 ngày trước."""
    b = base.select("hadm_id", "subject_id", F.to_date("admittime").alias("adm_date"))
    o = omr.withColumn("chartdate", F.to_date("chartdate")).select(
        "subject_id",
        "chartdate",
        *(["seq_num"] if "seq_num" in omr.columns else []),
        "result_name",
        "result_value",
    )
    j = b.join(o, "subject_id").filter(
        (F.col("chartdate") < F.col("adm_date"))
        & (F.col("chartdate") >= F.date_sub(F.col("adm_date"), OMR_LOOKBACK_DAYS))
    )
    val = F.col("result_value")
    j = j.withColumn(
        "metric",
        F.when(F.col("result_name") == "BMI (kg/m2)", "bmi")
        .when(F.col("result_name") == "Weight (Lbs)", "weight_kg")
        .when(F.col("result_name") == "Blood Pressure", "sbp")
        .when(F.col("result_name") == "eGFR", "egfr"),
    )
    j = j.filter(F.col("metric").isNotNull()).withColumn(
        "v",
        F.when(F.col("metric") == "sbp", _num(F.split(val, "/").getItem(0)))
        .when(F.col("metric") == "weight_kg", _num(val) * LBS_TO_KG)
        .otherwise(_num(val)),
    )
    j = j.filter(
        F.col("v").isNotNull()
        & _or(
            [
                (F.col("metric") == m) & F.col("v").between(lo, hi)
                for m, (lo, hi) in OMR_RANGES.items()
            ]
        )
    )
    order = [F.col("chartdate").desc()] + (
        [F.col("seq_num").desc()] if "seq_num" in j.columns else []
    )
    w = Window.partitionBy("hadm_id", "metric").orderBy(*order)
    last = j.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1)
    return last.groupBy("hadm_id").agg(
        *[
            F.max(F.when(F.col("metric") == m, F.col("v"))).alias(
                f"omr_{m.replace('weight_kg', 'weight')}_last"
            )
            for m in OMR_RANGES
        ]
    )


# --------------------------------------------------------------------------- xét nghiệm
def lab_window(
    labevents: DataFrame, base: DataFrame, strict_store: bool = True
) -> DataFrame:
    """Mọi labevents trong [admittime-6h, prediction_time]; nối theo subject_id (lab ED trước nhập viện có thể không có hadm_id)."""
    b = F.broadcast(
        base.select("subject_id", "hadm_id", "admittime", "prediction_time")
    )
    le = labevents.select(
        "subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"
    )
    j = le.join(b, "subject_id").filter(
        (F.col("charttime") >= F.expr(f"admittime - INTERVAL {LAB_BEFORE_HOURS} HOURS"))
        & (F.col("charttime") <= F.col("prediction_time"))
    )
    if (
        strict_store
    ):  # L5: kết quả phải có (storetime) tại mốc dự đoán; storetime NULL thì không kiểm được -> giữ
        j = j.filter(
            F.col("storetime").isNull()
            | (F.col("storetime") <= F.col("prediction_time"))
        )
    return j


def lab_features(lab_win: DataFrame, lab_cfg: dict, min_unit_share: float = 0.95):
    """Trả về (DataFrame <lab>_first/_min/_max theo hadm_id, thống kê loại bỏ)."""
    all_ids = [i for c in lab_cfg.values() for i in c["itemids"]]
    name_col = F.coalesce(
        *[
            F.when(F.col("itemid").isin(c["itemids"]), F.lit(n))
            for n, c in lab_cfg.items()
        ]
    )
    sel = (
        lab_win.filter(F.col("itemid").isin(all_ids))
        .withColumn("lab", name_col)
        .withColumn("unit", F.coalesce(F.col("valueuom"), F.lit("")))
    )
    # đơn vị chủ đạo của từng lab (quy tắc 3 của feature_spec)
    counts = (
        sel.filter(F.col("valuenum").isNotNull())
        .groupBy("lab", "unit")
        .count()
        .collect()
    )
    dominant, share_info = {}, {}
    for lab in lab_cfg:
        rows = [(r["unit"], r["count"]) for r in counts if r["lab"] == lab]
        tot = sum(n for _, n in rows)
        if tot:
            u, n = max(rows, key=lambda x: x[1])
            dominant[lab], share_info[lab] = u, (u, n / tot)
            if n / tot < min_unit_share:
                print(
                    f"[CẢNH BÁO] {lab}: đơn vị chủ đạo '{u}' chỉ chiếm {n / tot:.1%} < {min_unit_share:.0%}"
                )
    unit_ok = _or(
        [(F.col("lab") == lab) & (F.col("unit") == u) for lab, u in dominant.items()]
    )
    rng_ok = _or(
        [
            (F.col("lab") == n) & F.col("valuenum").between(*c["valid_range"])
            for n, c in lab_cfg.items()
        ]
    )
    tagged = (
        sel.withColumn("f_nonnum", F.col("valuenum").isNull().cast("int"))
        .withColumn("f_unit", (F.col("valuenum").isNotNull() & ~unit_ok).cast("int"))
        .withColumn(
            "f_range", (F.col("valuenum").isNotNull() & unit_ok & ~rng_ok).cast("int")
        )
    )
    stats_rows = (
        tagged.groupBy("lab")
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.sum("f_nonnum").alias("nonnum"),
            F.sum("f_unit").alias("bad_unit"),
            F.sum("f_range").alias("out_range"),
        )
        .collect()
    )
    stats = {
        r["lab"]: {
            "n": r["n"],
            "nonnum": r["nonnum"],
            "bad_unit": r["bad_unit"],
            "out_range": r["out_range"],
            "unit": share_info.get(r["lab"], ("", 0))[0],
            "unit_share": share_info.get(r["lab"], ("", 0))[1],
        }
        for r in stats_rows
    }
    good = tagged.filter((F.col("f_nonnum") + F.col("f_unit") + F.col("f_range")) == 0)
    aggs = []
    for n in lab_cfg:
        m = F.col("lab") == n
        aggs += [
            F.min(F.when(m, F.struct("charttime", "valuenum")))
            .getField("valuenum")
            .alias(f"{n}_first"),
            F.min(F.when(m, F.col("valuenum"))).alias(f"{n}_min"),
            F.max(F.when(m, F.col("valuenum"))).alias(f"{n}_max"),
        ]
    return good.groupBy("hadm_id").agg(*aggs), stats


# --------------------------------------------------------------------------- sinh hiệu ICU
def vitals_features(
    chartevents: DataFrame, base: DataFrame, vit_cfg: dict, strict_store: bool = True
):
    ids = [i for c in vit_cfg.values() for i in c["itemids"]]
    b = F.broadcast(base.select("hadm_id", "admittime", "prediction_time"))
    ce = chartevents.filter(F.col("itemid").isin(ids)).select(
        "hadm_id", "itemid", "charttime", "storetime", "valuenum"
    )
    j = ce.join(b, "hadm_id").filter(
        (F.col("charttime") >= F.col("admittime"))
        & (F.col("charttime") <= F.col("prediction_time"))
    )
    if strict_store:
        j = j.filter(
            F.col("storetime").isNull()
            | (F.col("storetime") <= F.col("prediction_time"))
        )
    j = j.withColumn(
        "vital",
        F.coalesce(
            *[
                F.when(F.col("itemid").isin(c["itemids"]), F.lit(n))
                for n, c in vit_cfg.items()
            ]
        ),
    )
    conv = F.col("valuenum")
    if "temperature" in vit_cfg:  # °F -> °C trước khi kiểm khoảng hợp lệ
        conv = F.when(
            F.col("vital") == "temperature", (F.col("valuenum") - 32.0) * 5.0 / 9.0
        ).otherwise(F.col("valuenum"))
    j = j.withColumn("v", conv).filter(F.col("v").isNotNull())
    in_rng = _or(
        [
            (F.col("vital") == n) & F.col("v").between(*c["valid_range"])
            for n, c in vit_cfg.items()
        ]
    )
    n_raw = j.count()
    j = j.filter(in_rng)
    aggs, cols = [], []
    for n, c in vit_cfg.items():
        for col in c["feature_columns"]:
            fn = (
                F.avg
                if col.endswith("_mean")
                else F.min if col.endswith("_min") else F.max
            )
            aggs.append(fn(F.when(F.col("vital") == n, F.col("v"))).alias(col))
            cols.append(col)
    out = j.groupBy("hadm_id").agg(*aggs)
    return out, cols, {"n_raw": n_raw}


# --------------------------------------------------------------------------- ghép + chuẩn hóa theo schema
def schema_columns(schema: dict) -> list[tuple[str, str]]:
    cols = []
    for sec in ("id_columns", "label_columns", "subgroup_only_columns"):
        cols += [(c["name"], c["type"]) for c in schema[sec]]
    for key, v in schema["feature_columns"].items():
        if key == "labs":
            for it in v["items"]:
                cols += [(it + suf, t) for suf, t in v["suffixes"].items()]
            cols += [(e["name"], e["type"]) for e in v["extra"]]
        else:
            cols += [(c["name"], c["type"]) for c in v]
    return cols


def assemble(
    base, demo, hist, omr, labs, nlab, vit, vit_cols, schema, lab_names
) -> DataFrame:
    df = (
        demo.join(hist, "hadm_id", "left")
        .join(omr, "hadm_id", "left")
        .join(labs, "hadm_id", "left")
        .join(nlab, "hadm_id", "left")
        .join(vit, "hadm_id", "left")
    )
    df = df.withColumn(
        "n_lab_tests_24h", F.coalesce(F.col("n_lab_tests_24h"), F.lit(0))
    )
    for n in lab_names:
        df = df.withColumn(f"{n}_missing", F.col(f"{n}_first").isNull().cast("int"))
    df = df.withColumn(
        "vitals_missing",
        reduce(lambda a, b: a & b, [F.col(c).isNull() for c in vit_cols]).cast("int"),
    )
    for c in ["n_prior_adm_365d", "n_prior_hf_adm", "first_admission"] + [
        c for c in df.columns if c.startswith("hx_")
    ]:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    spec = schema_columns(schema)
    missing = [n for n, _ in spec if n not in df.columns]
    assert not missing, f"Thiếu cột so với gold_schema_p1.json: {missing}"
    out = df.select(*[F.col(n).cast(SPARK_TYPES[t]).alias(n) for n, t in spec])
    forbidden = set(schema["forbidden_columns"]) & set(out.columns)
    assert not forbidden, f"Cột bị cấm xuất hiện: {forbidden}"
    return out


# --------------------------------------------------------------------------- báo cáo
def write_report(
    path: Path,
    out: DataFrame,
    schema: dict,
    lab_stats: dict,
    vit_stats: dict,
    n_base: int,
    args,
) -> None:
    spec = schema_columns(schema)
    feat_names = [
        n
        for n, t in spec
        if n
        not in {
            c["name"]
            for sec in ("id_columns", "label_columns", "subgroup_only_columns")
            for c in schema[sec]
        }
    ]
    n = out.count()
    agg = (
        out.agg(
            F.countDistinct("subject_id").alias("n_pat"),
            F.sum("label_mortality").alias("pos"),
            F.sum("in_icu_24h").alias("icu"),
            F.sum("vitals_missing").alias("vit_miss"),
            *[F.avg(F.col(c).isNull().cast("double")).alias(c) for c in feat_names],
        )
        .first()
        .asDict()
    )
    L = [
        "# Kết quả bảng đặc trưng P1 (Giai đoạn 2)",
        "",
        f"- Phiên bản dữ liệu: `{args.version}`; mẫu: {'toàn bộ' if not args.sample_frac else f'{args.sample_frac:.0%} bệnh nhân'}",
        f"- Số ca trong cohort sau quy tắc P1 (còn nằm viện, chưa tử vong tại 24h): {n_base:,}",
        f"- Số dòng Gold: {n:,}; số bệnh nhân: {agg['n_pat']:,}",
        f"- Ca dương (tử vong trong viện): {int(agg['pos'] or 0):,} ({(agg['pos'] or 0) / max(n, 1):.2%})",
        f"- Nằm ICU trong 24h đầu: {int(agg['icu'] or 0):,} ({(agg['icu'] or 0) / max(n, 1):.1%}); thiếu toàn bộ sinh hiệu: {int(agg['vit_miss'] or 0):,}",
        f"- Sinh hiệu: {vit_stats['n_raw']:,} phép đo trong cửa sổ trước khi lọc khoảng hợp lệ",
        "",
        "## Loại bỏ theo từng xét nghiệm (cửa sổ -6h → 24h)",
        "",
        "| Lab | Số phép đo | Không phải số | Sai đơn vị | Ngoài khoảng | Đơn vị chủ đạo | Tỷ lệ đơn vị |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for lab, s in lab_stats.items():
        L.append(
            f"| {lab} | {s['n']:,} | {s['nonnum']:,} | {s['bad_unit']:,} | {s['out_range']:,} | {s['unit'] or '—'} | {s['unit_share']:.1%} |"
        )
    L += [
        "",
        "## Tỷ lệ NULL của từng đặc trưng (toàn bộ)",
        "",
        "| Đặc trưng | % NULL |",
        "|---|---:|",
    ]
    L += [
        f"| `{c}` | {agg[c]:.1%} |"
        for c in feat_names
        if not c.endswith("_missing") and agg[c] > 0
    ]
    L += ["", "Chỉ chứa số liệu tổng hợp, không có dữ liệu cá nhân.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_yaml("configs/data.yaml")
    items = load_yaml("configs/feature_items.yaml")
    hfc = load_yaml("configs/hf_codes.yaml")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", choices=["silver", "bronze"], default="silver")
    ap.add_argument("--root", default=None)
    ap.add_argument("--cohort", default=None, help="Mặc định <paths.processed>/cohort")
    ap.add_argument(
        "--labels",
        default=None,
        help="Parquet nhãn P1 của Hà (cần cột hadm_id + --label-col)",
    )
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", default=None, help="Mặc định <paths.gold>/p1/<version>")
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Lấy mẫu theo subject_id (vd 0.01)",
    )
    ap.add_argument(
        "--no-storetime-filter",
        action="store_true",
        help="Bỏ lọc storetime <= prediction_time (L5)",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Bỏ qua đối chiếu itemid với d_labitems/d_items",
    )
    ap.add_argument(
        "--report", default=str(REPO_ROOT / "docs/stage2/features_result.md")
    )
    args = ap.parse_args()

    root = args.root or cfg["paths"][args.fmt]
    processed = (
        cfg["paths"]
        .get("processed", "s3a://heartguard-mimic-group11/processed")
        .rstrip("/")
    )
    gold = cfg["paths"].get("gold", "s3a://heartguard-mimic-group11/gold").rstrip("/")
    cohort_path = args.cohort or f"{processed}/cohort"
    out_path = args.out or (
        f"{gold}/p1/{args.version}"
        if not args.sample_frac
        else f"{gold}/p1/_sample_{args.sample_frac}"
    )
    strict = not args.no_storetime_filter

    t0 = time.time()
    spark = get_spark("hg-features-p1")
    rt = lambda name: read_table(spark, root, name, args.fmt)

    if not args.skip_verify:
        problems = verify_items(spark, root, args.fmt, items)
        if problems:
            print(
                "[LỖI] itemid chưa khớp từ điển MIMIC thật (báo Chiến sửa feature_items.yaml, hoặc dùng --skip-verify):"
            )
            print("\n".join("  - " + p for p in problems))
            raise SystemExit(1)
        print("[OK] Tất cả itemid khớp d_labitems/d_items")

    cohort = spark.read.parquet(cohort_path)
    labels = spark.read.parquet(args.labels) if args.labels else None
    base = build_base(cohort, args.sample_frac, labels, args.label_col).cache()
    n_base = base.count()
    print(f"[base] {n_base:,} ca P1")

    dx = rt("diagnoses_icd")
    demo = demographics(base, rt("icustays"), dx)
    hist = history(
        base, rt("admissions"), dx, hfc["heart_failure"], hfc["comorbidities"]
    )
    omr = omr_features(base, rt("omr"))

    lw = lab_window(rt("labevents"), base, strict).cache()
    nlab = lw.groupBy("hadm_id").agg(F.count(F.lit(1)).alias("n_lab_tests_24h"))
    labs, lab_stats = lab_features(
        lw, items["labs"], items["selection"].get("minimum_unit_share", 0.95)
    )
    vit, vit_cols, vit_stats = vitals_features(
        rt("chartevents"), base, items["vitals"], strict
    )

    out = assemble(
        base, demo, hist, omr, labs, nlab, vit, vit_cols, schema, list(items["labs"])
    ).cache()
    n = out.count()
    assert n == out.select("hadm_id").distinct().count(), "hadm_id bị trùng"
    assert out.filter(F.col("label_mortality").isNull()).count() == 0, "có nhãn NULL"
    assert (
        out.filter(F.col("prediction_time") <= F.col("admittime")).count() == 0
    ), "prediction_time <= admittime"
    print(f"[OK] {n:,} dòng, {len(out.columns)} cột")

    out.repartition(4).write.mode("overwrite").parquet(out_path)
    print(f"[OK] Đã ghi → {out_path}")
    write_report(Path(args.report), out, schema, lab_stats, vit_stats, n_base, args)
    print(f"[OK] Đã ghi báo cáo → {args.report}")
    print(f"Xong sau {time.time() - t0:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
