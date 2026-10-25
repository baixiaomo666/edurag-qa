# -*- coding: utf-8 -*-
"""
B 组实验 · 检索层消融（B1 Reranker / B2 上下文粒度 / B3 融合权重 / B4 上下文数量）
================================================================================
全部通过 vector_store.py 的 EXPERIMENT_* 环境变量驱动，不改一行检索逻辑。

指标（都在检索层，不调用 LLM，因此可以在几分钟内跑完全部组合）：
  recall_num  —— 参考答案里的数字，有多少比例出现在召回的上下文里。为 None 表示
                 该题参考答案不含数字，无法用这个口径衡量（不计入均值）
  ctx_chars   —— 送进 prompt 的上下文总字符数。在 recall 相同的前提下，它越小越好
                 （越短 = 越少无关内容 = 后面 faithfulness 越不容易被稀释）
  latency_ms  —— 单次检索耗时，含 BGE-M3 编码 + 双路 ANN + 融合 + 精排

指标口径的取舍（为什么不是 context_recall / context_precision）：
  真正的 context_recall 需要 LLM 判定，200+ 次调用不适合做参数扫描。
  参考答案里的数字是"硬事实"——只要答案说"录取 40 人"，上下文里就必须有 40。
  所以用数字召回率做 recall 的代理指标；它只可能高估、不会漏判。

用法：
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_b.py
"""
import collections
import json
import os
import re
import statistics
import tempfile
import time

# ---- 日志重定向必须在 import base.logger 之前 ----
from base.config import config

config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_b.log")

from rag_qa.core.vector_store import VectorStore              # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
os.makedirs(OUT_DIR, exist_ok=True)

QA_PATH = os.path.join(HERE, "rag_qa", "rag_assesment", "kaoyan_qa_100.json")
SAMPLES_PER_SUBJECT = {"kaoyan": 8, "cs": 7, "employment": 5}   # 共 20 条专业题

# ---------------------------------------------------------------- 实验分组
GROUPS = [
    ("B1 精排开（现状）",        {"EXPERIMENT_RERANKER": "1"}),
    ("B1 精排关",                {"EXPERIMENT_RERANKER": "0"}),
    ("B2 父块（现状）",          {"EXPERIMENT_CTX": "parent"}),
    ("B2 子块",                  {"EXPERIMENT_CTX": "child"}),
    ("B3 稠密0.7/稀疏1.0（现状）", {"DENSE_W": "0.7", "SPARSE_W": "1.0"}),
    ("B3 稠密1.0/稀疏0.7",       {"DENSE_W": "1.0", "SPARSE_W": "0.7"}),
    ("B3 稠密0.5/稀疏0.5",       {"DENSE_W": "0.5", "SPARSE_W": "0.5"}),
    ("B3 稠密0.3/稀疏1.0",       {"DENSE_W": "0.3", "SPARSE_W": "1.0"}),
    ("B3 纯稠密1.0/0.0",         {"DENSE_W": "1.0", "SPARSE_W": "0.0"}),
    ("B3 纯稀疏0.0/1.0",         {"DENSE_W": "0.0", "SPARSE_W": "1.0"}),
    ("B4 M=1（K=10）",           {"EXPERIMENT_M": "1", "EXPERIMENT_K": "10"}),
    ("B4 M=2（K=10）",           {"EXPERIMENT_M": "2", "EXPERIMENT_K": "10"}),
    ("B4 M=3（K=10）",           {"EXPERIMENT_M": "3", "EXPERIMENT_K": "10"}),
    ("B4 M=5（K=10）",           {"EXPERIMENT_M": "5", "EXPERIMENT_K": "10"}),
]

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    return set(NUM_RE.findall(text or ""))


def build_samples():
    """按学科抽题，并把参考答案从评测集里对齐过来"""
    qa = {item["question"]: item.get("ground_truth", "") for item in
          json.load(open(QA_PATH, encoding="utf-8"))}
    bm = BM25Search()
    qs, subs = list(bm.original_questions), list(bm.subjects or [])
    bucket = collections.defaultdict(list)
    for q, s in zip(qs, subs):
        if q in qa:                       # 只取评测集里有参考答案的题
            bucket[s].append({"question": q, "ground_truth": qa[q], "subject": s})
    samples = []
    for s, n in SAMPLES_PER_SUBJECT.items():
        samples.extend(bucket.get(s, [])[:n])
    return samples


def run_group(vs, samples, env):
    """在指定环境变量下把 20 条题跑一遍"""
    for k, v in env.items():
        os.environ[k] = v
    rows = []
    try:
        for s in samples:
            t0 = time.perf_counter()
            docs = vs.hybrid_search_with_reranker(s["question"])
            ms = (time.perf_counter() - t0) * 1000
            ctx = "\n".join((d.page_content or "") for d in docs)
            gt_nums = numbers(s["ground_truth"])
            hit = len(gt_nums & numbers(ctx)) / len(gt_nums) if gt_nums else None
            rows.append({
                "question": s["question"],
                "subject": s["subject"],
                "n_ctx": len(docs),
                "ctx_chars": len(ctx),
                "recall_num": hit,
                "latency_ms": round(ms, 1),
            })
    finally:
        for k in env:
            os.environ.pop(k, None)
    return rows


def summarize(rows):
    recalls = [r["recall_num"] for r in rows if r["recall_num"] is not None]
    return {
        "n_questions": len(rows),
        "n_verifiable": len(recalls),
        "recall_num_mean": round(statistics.mean(recalls), 4) if recalls else None,
        "recall_num_median": round(statistics.median(recalls), 4) if recalls else None,
        "recall_num_full_rate": round(sum(1 for x in recalls if x == 1.0) / len(recalls), 4) if recalls else None,
        "avg_ctx_chars": round(statistics.mean([r["ctx_chars"] for r in rows]), 0),
        "avg_n_ctx": round(statistics.mean([r["n_ctx"] for r in rows]), 2),
        "latency_median_ms": round(statistics.median([r["latency_ms"] for r in rows]), 0),
        "latency_mean_ms": round(statistics.mean([r["latency_ms"] for r in rows]), 0),
    }


def main():
    samples = build_samples()
    print("=" * 100)
    print(f"B 组实验 · 检索层消融   样本 {len(samples)} 条专业题"
          f"（{dict(collections.Counter(s['subject'] for s in samples))}）")
    print("=" * 100)

    t0 = time.perf_counter()
    vs = VectorStore()
    print(f"VectorStore 构造耗时 {time.perf_counter() - t0:.1f} s")
    print(f"共 {len(GROUPS)} 组，每组 {len(samples)} 次检索\n")

    results = {}
    for i, (label, env) in enumerate(GROUPS, 1):
        rows = run_group(vs, samples, env)
        sm = summarize(rows)
        results[label] = {"env": env, "summary": sm, "rows": rows}
        print(f"[{i:>2}/{len(GROUPS)}] {label:<28} "
              f"recall {sm['recall_num_mean'] if sm['recall_num_mean'] is not None else float('nan'):.4f}  "
              f"满召回率 {sm['recall_num_full_rate']:.2%}  "
              f"ctx {sm['avg_ctx_chars']:>6.0f} 字  "
              f"{sm['latency_median_ms']:>6.0f} ms")

    # ---------------- 汇总表 ----------------
    print("\n" + "=" * 100)
    print("汇总（recall_num = 参考答案数字出现在上下文中的比例；ctx = 送入 LLM 的上下文字符数）")
    print("=" * 100)
    print(f"{'分组':<30}{'可验证':<8}{'均值':<10}{'中位':<10}{'满召回率':<12}{'平均字数':<12}{'中位延迟':<10}")
    print("-" * 100)
    for label, _ in GROUPS:
        s = results[label]["summary"]
        r = s["recall_num_mean"]
        med = s["recall_num_median"]
        full = s["recall_num_full_rate"]
        print(f"{label:<30}{s['n_verifiable']:<8}"
              f"{(f'{r:.4f}' if r is not None else '-'):<10}"
              f"{(f'{med:.4f}' if med is not None else '-'):<10}"
              f"{(f'{full:.2%}' if full is not None else '-'):<12}"
              f"{s['avg_ctx_chars']:<12.0f}"
              f"{s['latency_median_ms']:<10.0f}")
    print("-" * 100)

    # ---------------- 逐题差异（只在关键对照之间） ----------------
    def diff(label_a, label_b):
        print(f"\n逐题差异：{label_b} 相对 {label_a}（只看 recall_num 发生变化的题）")
        ra = {r["question"]: r for r in results[label_a]["rows"]}
        rb = {r["question"]: r for r in results[label_b]["rows"]}
        changed = 0
        for q, a in ra.items():
            b = rb[q]
            if a["recall_num"] is None or b["recall_num"] is None:
                continue
            if abs(a["recall_num"] - b["recall_num"]) > 1e-9:
                changed += 1
                print(f"  [{a['recall_num']:.2f} → {b['recall_num']:.2f}] {q}")
                print(f"        {label_a}: {a['n_ctx']} 条 / {a['ctx_chars']} 字 / {a['latency_ms']:.0f} ms")
                print(f"        {label_b}: {b['n_ctx']} 条 / {b['ctx_chars']} 字 / {b['latency_ms']:.0f} ms")
        if changed == 0:
            print("  （无变化）")

    diff("B1 精排开（现状）", "B1 精排关")
    diff("B2 父块（现状）", "B2 子块")
    diff("B4 M=1（K=10）", "B4 M=5（K=10）")

    payload = {
        "sample_size": len(samples),
        "samples_by_subject": dict(collections.Counter(s["subject"] for s in samples)),
        "note": "recall_num 是 context_recall 的代理指标（参考答案数字命中率），只可能高估",
        "groups": {k: v for k, v in results.items()},
    }
    path = os.path.join(OUT_DIR, "exp_b.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    main()
