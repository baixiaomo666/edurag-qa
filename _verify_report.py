# -*- coding: utf-8 -*-
"""独立核验：《考研就业问答系统-C组实验结果.md》中的数字是否与落盘数据一致。
只读，不改任何实验产物。用法: python -X utf8 _verify_report.py
"""
import json
import os
import re
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, "考研就业问答系统-C组实验结果.md")

DIFFS = []
OKS = []


def chk(label, report_val, data_val, tol=0.0011):
    """tol：报告普遍保留 2—4 位小数，容忍四舍五入误差。"""
    if report_val is None:
        DIFFS.append(f"{label}: 报告中未找到")
        return
    if isinstance(report_val, str):
        DIFFS.append(f"{label}: 报告值无法解析 -> {report_val!r}")
        return
    if isinstance(data_val, (int, float)) and abs(report_val - data_val) <= tol:
        OKS.append(f"{label}: {report_val} == {data_val}")
    else:
        DIFFS.append(f"{label}: 报告 {report_val} vs 数据 {data_val}")


def num(s):
    if s is None:
        return None
    s = s.replace("**", "").replace(",", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def main():
    text = open(REPORT, encoding="utf-8").read()
    lines = text.split("\n")

    # ---------- C1 表 ----------
    c1 = json.load(open(os.path.join(ROOT, "eval_runs", "exp_c1.json"), encoding="utf-8"))
    groups = c1["groups"]
    order = ["direct", "auto", "hyde", "subquery"]
    keys = []
    for tag in order:
        hit = [k for k in groups if tag in k]
        if len(hit) != 1:
            DIFFS.append(f"C1 组名匹配失败: {tag} -> {hit}")
            return
        keys.append(hit[0])

    c1_rows = []
    for ln in lines:
        if not ln.strip().startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) != 10:
            continue
        head = cols[0].replace("**", "").strip().lower()
        if any(head.startswith(t) for t in order):
            c1_rows.append(cols)
    if len(c1_rows) != 4:
        DIFFS.append(f"C1 表解析到 {len(c1_rows)} 行，期望 4 行")
    else:
        for tag, key, row in zip(order, keys, c1_rows):
            s = groups[key]["summary"]
            chk(f"C1[{tag}].recall_term_mean", num(row[1]), s["recall_term_mean"])
            chk(f"C1[{tag}].recall_num_mean", num(row[2]), s["recall_num_mean"])
            chk(f"C1[{tag}].recall_term_full", num(row[3]) / 100.0, s["recall_term_full"])
            chk(f"C1[{tag}].rank_hit_rate", num(row[4]) / 100.0, s["rank_hit_rate"])
            chk(f"C1[{tag}].avg_ctx_chars", num(row[5]), s["avg_ctx_chars"], tol=0.51)
            chk(f"C1[{tag}].avg_n_ctx", num(row[6]), s["avg_n_ctx"], tol=0.006)
            chk(f"C1[{tag}].latency_median_ms", num(row[7]), s["latency_median_ms"], tol=0.51)
            lats = [r["latency_ms"] for r in groups[key]["rows"]]
            chk(f"C1[{tag}].latency_mean_ms", num(row[8]), sum(lats) / len(lats), tol=0.51)
            calls = num(row[9])
            chk(f"C1[{tag}].llm_calls_total", calls, s["llm_calls_total"], tol=0.001)
            tot = num(row[9].split("/")[-1]) if "/" in row[9] else None
            chk(f"C1[{tag}].llm_ms_total", tot, s["llm_ms_total"], tol=0.51)

    # C1 auto 的策略分布（报告用散文写法，如「直接检索 15 / 子查询 3 / 假设问题 2」）
    want = groups[keys[1]]["summary"]["strategy_dist"]
    miss = [
        f"{k} {v}"
        for k, v in want.items()
        if f"{k} {v}" not in text and f"{k.replace('检索', '')} {v}" not in text
    ]
    if miss:
        DIFFS.append(f"C1[auto].strategy_dist 缺 {miss}（期望 {want}）")
    else:
        OKS.append(f"C1[auto].strategy_dist 命中: {want}")

    # ---------- C2 表 ----------
    c2 = json.load(open(os.path.join(ROOT, "eval_runs", "exp_c2.json"), encoding="utf-8"))
    vers = c2["versions"]
    c2_rows = []
    for ln in lines:
        if not ln.strip().startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) != 10:
            continue
        head = cols[0].replace("**", "").strip().lower()
        if head in ("v1", "v2", "v3") and "n=" in cols[1]:
            c2_rows.append(cols)
    if len(c2_rows) != 3:
        DIFFS.append(f"C2 表解析到 {len(c2_rows)} 行，期望 3 行")
    else:
        for tag, row in zip(["v1", "v2", "v3"], c2_rows):
            s = vers[tag]["summary"]
            r = vers[tag]["ragas"]
            chk(f"C2[{tag}].latency_median_ms", num(row[5]), s["latency_median_ms"], tol=0.51)
            chk(f"C2[{tag}].answer_chars_mean", num(row[6]), s["answer_chars_mean"], tol=0.051)
            chk(f"C2[{tag}].avg_n_ctx", num(row[7]), s["avg_n_ctx"], tol=0.006)
            chk(f"C2[{tag}].avg_ctx_chars", num(row[8]), s["avg_ctx_chars"], tol=0.51)
            chk(f"C2[{tag}].recall_term_mean", num(row[9]), s["recall_term_mean"])
            if r:
                chk(f"C2[{tag}].faithfulness", num(row[1]), r.get("faithfulness"), tol=0.0011)
                chk(f"C2[{tag}].answer_relevancy", num(row[2]), r.get("answer_relevancy"), tol=0.0011)
                chk(f"C2[{tag}].context_precision", num(row[3]), r.get("context_precision"), tol=0.0011)
                chk(f"C2[{tag}].context_recall", num(row[4]), r.get("context_recall"), tol=0.0011)
            else:
                DIFFS.append(f"C2[{tag}] 的 summary 里没有 ragas 段")

    # ---------- 独立重算 RAGAS（直接从 CSV）----------
    import csv as _csv
    for tag in ["v1", "v2", "v3"]:
        p = os.path.join(ROOT, "eval_runs", f"exp_c2_{tag}_ragas.csv")
        if not os.path.exists(p):
            DIFFS.append(f"缺 CSV {p}")
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(_csv.DictReader(f))
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            vals = [float(x[metric]) for x in rows if x.get(metric) not in (None, "", "nan")]
            if not vals:
                DIFFS.append(f"CSV[{tag}].{metric} 全为空")
                continue
            mean = sum(vals) / len(vals)
            got = vers[tag]["ragas"].get(metric)
            chk(f"CSV复算[{tag}].{metric}", round(mean, 6), round(got, 6), tol=1e-6)
            nan = len(rows) - len(vals)
            if nan:
                OKS.append(f"CSV[{tag}].{metric} NaN={nan}")

    # ---------- 结论 ----------
    print("=" * 70)
    print(f"通过 {len(OKS)} 项 / 差异 {len(DIFFS)} 项")
    if DIFFS:
        print("-" * 70)
        for d in DIFFS:
            print("DIFF:", d)
    print("=" * 70)
    if not DIFFS:
        print("REPORT_VERIFY_OK")
    return 0 if not DIFFS else 1


if __name__ == "__main__":
    sys.exit(main())
