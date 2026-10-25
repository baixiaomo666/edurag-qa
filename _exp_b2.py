# -*- coding: utf-8 -*-
"""
B 组实验（第二轮）· 检索层消融 —— 改用更细的指标
================================================================================
第一轮用「参考答案数字命中率」做 recall 代理，14 组里 11 组都是 1.0000，指标饱和：
知识库只有 41 个块，答案几乎必然落在 Top-2 父块内，看不出参数差异。

本轮改为三个互补指标，其中 recall_term 是主要判据：

  recall_num   参考答案里的数字覆盖率（粗，但硬）
  recall_term  参考答案的实词覆盖率（jieba 分词去停用词后）—— 主要判据
  rank_hit     召回的上下文里，是否出现了参考答案的「核心词」
               （核心词 = 参考答案中长度≥3 且非数字的实词；只要有一个命中即算命中）
  ctx_chars    送进 prompt 的上下文总字符数
  latency_ms   单次检索耗时

用法：
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_b2.py
"""
import collections
import json
import os
import re
import statistics
import tempfile
import time

from base.config import config

config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_b2.log")

from rag_qa.core.vector_store import VectorStore              # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search          # noqa: E402
from mysql_qa.util.preprocess import process_text              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
os.makedirs(OUT_DIR, exist_ok=True)
QA_PATH = os.path.join(HERE, "rag_qa", "rag_assesment", "kaoyan_qa_100.json")
SAMPLES_PER_SUBJECT = {"kaoyan": 8, "cs": 7, "employment": 5}

GROUPS = [
    ("B1 精排开（现状）",          {"EXPERIMENT_RERANKER": "1"}),
    ("B1 精排关",                  {"EXPERIMENT_RERANKER": "0"}),
    ("B2 父块（现状）",            {"EXPERIMENT_CTX": "parent"}),
    ("B2 子块",                    {"EXPERIMENT_CTX": "child"}),
    ("B3 稠密0.7/稀疏1.0（现状）",  {"DENSE_W": "0.7", "SPARSE_W": "1.0"}),
    ("B3 纯稠密1.0/0.0",           {"DENSE_W": "1.0", "SPARSE_W": "0.0"}),
    ("B3 纯稀疏0.0/1.0",           {"DENSE_W": "0.0", "SPARSE_W": "1.0"}),
    ("B4 M=1（K=10）",             {"EXPERIMENT_M": "1", "EXPERIMENT_K": "10"}),
    ("B4 M=2（K=10）",             {"EXPERIMENT_M": "2", "EXPERIMENT_K": "10"}),
    ("B4 M=3（K=10）",             {"EXPERIMENT_M": "3", "EXPERIMENT_K": "10"}),
    ("B4 M=5（K=10）",             {"EXPERIMENT_M": "5", "EXPERIMENT_K": "10"}),
]

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    return set(NUM_RE.findall(text or ""))


def core_terms(text):
    """参考答案的核心词：长度≥3 的非数字实词（process_text 已去停用词）"""
    return {t for t in process_text(text or "") if len(t) >= 3 and not t.isdigit()}


def build_samples():
    qa = {item["question"]: item.get("ground_truth", "") for item in
          json.load(open(QA_PATH, encoding="utf-8"))}
    bm = BM25Search()
    bucket = collections.defaultdict(list)
    for q, s in zip(bm.original_questions, bm.subjects or []):
        if q in qa:
            bucket[s].append({"question": q, "ground_truth": qa[q], "subject": s})
    samples = []
    for s, n in SAMPLES_PER_SUBJECT.items():
        samples.extend(bucket.get(s, [])[:n])
    return samples


def run_group(vs, samples, env):
    for k, v in env.items():
        os.environ[k] = v
    rows = []
    try:
        for s in samples:
            t0 = time.perf_counter()
            docs = vs.hybrid_search_with_reranker(s["question"])
            ms = (time.perf_counter() - t0) * 1000
            ctx = "\n".join((d.page_content or "") for d in docs)

            gt_nums, ctx_nums = numbers(s["ground_truth"]), numbers(ctx)
            gt_terms = set(process_text(s["ground_truth"] or ""))
            ctx_terms = set(process_text(ctx))
            gt_core = core_terms(s["ground_truth"])

            rows.append({
                "question": s["question"],
                "subject": s["subject"],
                "n_ctx": len(docs),
                "ctx_chars": len(ctx),
                "recall_num": (len(gt_nums & ctx_nums) / len(gt_nums)) if gt_nums else None,
                "recall_term": (len(gt_terms & ctx_terms) / len(gt_terms)) if gt_terms else None,
                "rank_hit": (1.0 if (gt_core & ctx_terms) else 0.0) if gt_core else None,
                "latency_ms": round(ms, 1),
            })
    finally:
        for k in env:
            os.environ.pop(k, None)
    return rows


def mean_of(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(vals) if vals else None


def summarize(rows):
    return {
        "n_questions": len(rows),
        "recall_num_mean": mean_of(rows, "recall_num"),
        "recall_term_mean": mean_of(rows, "recall_term"),
        "recall_term_full": (sum(1 for r in rows if r["recall_term"] == 1.0)
                             / max(1, sum(1 for r in rows if r["recall_term"] is not None))),
        "rank_hit_rate": mean_of(rows, "rank_hit"),
        "avg_ctx_chars": statistics.mean([r["ctx_chars"] for r in rows]),
        "avg_n_ctx": statistics.mean([r["n_ctx"] for r in rows]),
        "latency_median_ms": statistics.median([r["latency_ms"] for r in rows]),
    }


def fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"


def main():
    samples = build_samples()
    print("=" * 104)
    print(f"B 组实验（第二轮·细化指标）  样本 {len(samples)} 条专业题 "
          f"{dict(collections.Counter(s['subject'] for s in samples))}")
    print("=" * 104)

    vs = VectorStore()
    print(f"共 {len(GROUPS)} 组 × {len(samples)} 次检索\n")

    results = {}
    for i, (label, env) in enumerate(GROUPS, 1):
        rows = run_group(vs, samples, env)
        sm = summarize(rows)
        results[label] = {"env": env, "summary": sm, "rows": rows}
        print(f"[{i:>2}/{len(GROUPS)}] {label:<26} "
              f"词召回 {fmt(sm['recall_term_mean'])}  满召回 {sm['recall_term_full']:>6.1%}  "
              f"核心词命中 {fmt(sm['rank_hit_rate'], '.2%')}  "
              f"ctx {sm['avg_ctx_chars']:>6.0f} 字  {sm['latency_median_ms']:>6.0f} ms")

    print("\n" + "=" * 104)
    print("汇总")
    print("=" * 104)
    hdr = (f"{'分组':<28}{'词召回':<10}{'中位词召回':<12}{'满召回率':<11}"
           f"{'核心词命中':<12}{'平均字数':<10}{'中位延迟':<9}")
    print(hdr)
    print("-" * 104)
    for label, _ in GROUPS:
        rows = results[label]["rows"]
        s = results[label]["summary"]
        med = None
        vals = [r["recall_term"] for r in rows if r["recall_term"] is not None]
        if vals:
            med = statistics.median(vals)
        print(f"{label:<28}{fmt(s['recall_term_mean']):<10}{fmt(med):<12}"
              f"{s['recall_term_full']:<11.1%}{fmt(s['rank_hit_rate'], '.2%'):<12}"
              f"{s['avg_ctx_chars']:<10.0f}{s['latency_median_ms']:<9.0f}")
    print("-" * 104)

    # ---- 逐题差异 ----
    def diff(a_label, b_label, threshold=0.02):
        print(f"\n逐题差异：{b_label}  相对  {a_label}（阈值 >{threshold}）")
        ra = {r["question"]: r for r in results[a_label]["rows"]}
        rb = {r["question"]: r for r in results[b_label]["rows"]}
        n = 0
        for q, a in ra.items():
            b = rb[q]
            if a["recall_term"] is None or b["recall_term"] is None:
                continue
            if abs(a["recall_term"] - b["recall_term"]) > threshold:
                n += 1
                print(f"  [{a['recall_term']:.3f} → {b['recall_term']:.3f}] {q}")
        if n == 0:
            print("  （无显著变化）")

    diff("B1 精排开（现状）", "B1 精排关")
    diff("B2 父块（现状）", "B2 子块")
    diff("B3 稠密0.7/稀疏1.0（现状）", "B3 纯稠密1.0/0.0")
    diff("B3 稠密0.7/稀疏1.0（现状）", "B3 纯稀疏0.0/1.0")
    diff("B4 M=1（K=10）", "B4 M=3（K=10）")

    path = os.path.join(OUT_DIR, "exp_b2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sample_size": len(samples),
                   "samples_by_subject": dict(collections.Counter(s["subject"] for s in samples)),
                   "groups": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    main()
