# -*- coding: utf-8 -*-
"""
B2 实验（生成层）· 上下文粒度对「答案质量」的影响
================================================================================
第二轮检索层实验已经证明：EXPERIMENT_CTX=child 时送进 prompt 的上下文从
约 1786 字降到约 516 字（-71%），而检索层的词召回几乎不变。

那么问题变成：**上下文砍掉 71% 之后，生成的答案质量会掉吗？**

这正是父子分块设计要回答的核心问题：
  * 若答案质量不降 → 说明父块里大量内容是「陪跑」，子块反而更聚焦
  * 若答案质量下降   → 说明父块提供的完整语境确实是必要的

本脚本用 RAGSystem（纯 RAG 链路，不含 FAQ）对同一批题跑两遍，
指标：
  term_recall  参考答案的实词在【生成答案】里的覆盖率  ← 主要判据
  num_recall   参考答案的数字在【生成答案】里的覆盖率
  faithfulness 用 RAGAS 打分（--ragas 开启，默认关闭，因为慢）

用法：
  # 只跑生成（约 5 分钟）
  python -X utf8 _exp_b2_llm.py
  # 同时算 RAGAS（再 +5 分钟）
  python -X utf8 _exp_b2_llm.py --ragas
"""
import argparse
import collections
import json
import os
import re
import statistics
import tempfile
import time

from base.config import config

config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_b2_llm.log")

from rag_qa.core.rag_system import RAGSystem                      # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search              # noqa: E402
from mysql_qa.util.preprocess import process_text                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
os.makedirs(OUT_DIR, exist_ok=True)
QA_PATH = os.path.join(HERE, "rag_qa", "rag_assesment", "kaoyan_qa_100.json")
SAMPLES_PER_SUBJECT = {"kaoyan": 8, "cs": 7, "employment": 5}

MODES = [("父块（现状）", "parent"), ("子块", "child")]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(t):
    return set(NUM_RE.findall(t or ""))


def terms(t):
    return set(process_text(t or ""))


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


def run_mode(rag, samples, mode):
    os.environ["EXPERIMENT_CTX"] = mode
    rows = []
    try:
        for i, s in enumerate(samples, 1):
            trace = {}
            t0 = time.perf_counter()
            ans = "".join(rag.generate_answer(s["question"], None, [], trace))
            ms = (time.perf_counter() - t0) * 1000
            ctxs = trace.get("contexts") or []
            ctx = "\n".join(ctxs)

            gt_t, ans_t = terms(s["ground_truth"]), terms(ans)
            gt_n, ans_n = numbers(s["ground_truth"]), numbers(ans)
            rows.append({
                "question": s["question"],
                "subject": s["subject"],
                "category": trace.get("category"),
                "n_ctx": len(ctxs),
                "ctx_chars": len(ctx),
                "answer_chars": len(ans),
                "term_recall": (len(gt_t & ans_t) / len(gt_t)) if gt_t else None,
                "num_recall": (len(gt_n & ans_n) / len(gt_n)) if gt_n else None,
                "latency_ms": round(ms, 1),
                "answer": ans,
                "contexts": ctxs,
                "ground_truth": s["ground_truth"],
            })
            print(f"    [{i:>2}/{len(samples)}] {s['question']}")
            print(f"          ctx={len(ctxs)} 条/{len(ctx)} 字  "
                  f"答案 {len(ans)} 字  用时 {ms:.0f} ms  "
                  f"词召回 {rows[-1]['term_recall'] if rows[-1]['term_recall'] is not None else float('nan'):.3f}")
    finally:
        os.environ.pop("EXPERIMENT_CTX", None)
    return rows


def mean_of(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(vals) if vals else None


def summarize(rows):
    return {
        "n": len(rows),
        "term_recall_mean": mean_of(rows, "term_recall"),
        "num_recall_mean": mean_of(rows, "num_recall"),
        "avg_ctx_chars": statistics.mean([r["ctx_chars"] for r in rows]),
        "avg_answer_chars": statistics.mean([r["answer_chars"] for r in rows]),
        "latency_median_ms": statistics.median([r["latency_ms"] for r in rows]),
        "category_dist": dict(collections.Counter(str(r["category"]) for r in rows)),
    }


def run_ragas(rows, label):
    """对一组结果跑 RAGAS 四指标（可选，慢）"""
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.chat_models import init_chat_model
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    except ImportError as e:
        print(f"    RAGAS 依赖缺失，跳过：{e}")
        return None

    usable = [r for r in rows if r["contexts"]]
    if not usable:
        print("    没有带上下文的样本，跳过 RAGAS")
        return None
    ds = Dataset.from_dict({
        "question": [r["question"] for r in usable],
        "answer": [r["answer"] for r in usable],
        "contexts": [r["contexts"] for r in usable],
        "ground_truth": [r["ground_truth"] for r in usable],
    })
    llm = init_chat_model(model=config.LLM_MODEL, model_provider="openai",
                          api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    emb = HuggingFaceEmbeddings(model_name=r"D:\Program\Develop\bge-m3")
    res = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                                context_precision, context_recall],
                   llm=llm, embeddings=emb)
    df = res.to_pandas()
    out = {m: (round(float(df[m].mean()), 4) if m in df.columns else None)
           for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    out["n"] = len(usable)
    df.to_csv(os.path.join(OUT_DIR, f"exp_b2_llm_{label}.csv"),
              index=False, encoding="utf-8-sig")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragas", action="store_true", help="额外跑 RAGAS 四指标（慢）")
    args = ap.parse_args()

    samples = build_samples()
    print("=" * 96)
    print(f"B2 生成层实验 · 父块 vs 子块   样本 {len(samples)} 条专业题")
    print("=" * 96)

    rag = RAGSystem()
    results = {}
    for label, mode in MODES:
        print(f"\n>>> 模式：{label}（EXPERIMENT_CTX={mode}）")
        results[label] = run_mode(rag, samples, mode)

    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    print(f"{'模式':<16}{'答案词召回':<12}{'答案数字召回':<14}{'上下文均字数':<14}{'答案均字数':<12}{'中位延迟':<10}")
    print("-" * 96)
    summaries = {}
    for label, _ in MODES:
        s = summarize(results[label])
        summaries[label] = s
        print(f"{label:<16}"
              f"{(format(s['term_recall_mean'], '.4f') if s['term_recall_mean'] is not None else '-'):<12}"
              f"{(format(s['num_recall_mean'], '.4f') if s['num_recall_mean'] is not None else '-'):<14}"
              f"{s['avg_ctx_chars']:<14.0f}{s['avg_answer_chars']:<12.0f}"
              f"{s['latency_median_ms']:<10.0f}")
    print("-" * 96)
    for label, _ in MODES:
        print(f"  {label} 的 BERT 分类分布：{summaries[label]['category_dist']}")

    # 逐题对比
    print("\n逐题对比（词召回变化 > 0.05 的题）")
    ra = {r["question"]: r for r in results[MODES[0][0]]}
    rb = {r["question"]: r for r in results[MODES[1][0]]}
    n = 0
    for q, a in ra.items():
        b = rb[q]
        if a["term_recall"] is None or b["term_recall"] is None:
            continue
        if abs(a["term_recall"] - b["term_recall"]) > 0.05:
            n += 1
            print(f"  [父块 {a['term_recall']:.3f} → 子块 {b['term_recall']:.3f}] {q}")
    if n == 0:
        print("  （无显著变化）")

    ragas_out = {}
    if args.ragas:
        print("\n" + "=" * 96)
        print("RAGAS 四指标")
        print("=" * 96)
        for label, _ in MODES:
            print(f"  {label} ...")
            key = "parent" if "父块" in label else "child"
            ragas_out[label] = run_ragas(results[label], key)
            print(f"    {ragas_out[label]}")

    path = os.path.join(OUT_DIR, "exp_b2_llm.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sample_size": len(samples),
                   "summaries": summaries,
                   "ragas": ragas_out,
                   "rows": {k: v for k, v in results.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    main()
