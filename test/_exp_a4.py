# -*- coding: utf-8 -*-
"""
A 组实验 A4 —— 完整链路 vs 纯 RAG（FAQ 层到底带来了什么）
================================================================
对同一批问题跑两条链路，对比延迟与答案：
  链路甲（完整）：IntegratedQASystem.query()  —— 问候语 → FAQ → 降级 RAG
  链路乙（纯 RAG）：RAGSystem.generate_answer() —— 直接 BERT 分类 → 检索/直答

要回答的问题：
  1. FAQ 层把题库内问题的响应速度提升了多少倍？
  2. 纯 RAG 链路能不能自己答对题库内的问题（即 FAQ 层是否只是"更快"）？
  3. 题库外的题目，两条链路是否等价（都应走 RAG）？

设计：
  * 复用同一份模型实例（qa.rag 就是 IntegratedQASystem 内部的 RAGSystem），
    否则会加载两份 BGE-M3 + bge-reranker，内存翻倍且结论被污染
  * 样本按学科均匀抽取（common / kaoyan / cs / employment 各 5 条）
  * 纯 RAG 链路不传 history，保证与完整链路可比

用法：
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_a4.py
"""
import asyncio
import collections
import json
import os
import tempfile
import time

# ---- 日志重定向必须在 import base.logger 之前 ----
from base.config import config

config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_a4.log")

from faq_milvus import IntegratedQASystem                          # noqa: E402
from mysql_qa.retrieval.bm25_search import BM25Search               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
os.makedirs(OUT_DIR, exist_ok=True)

PER_SUBJECT = 5  # 每个学科抽几条


def build_samples():
    """从真实题库里按学科均匀抽题（题库 = FAQ 层能答的问题）"""
    bm = BM25Search()
    qs = list(bm.original_questions)
    subs = list(bm.subjects or ["common"] * len(qs))
    by_subject = collections.defaultdict(list)
    for q, s in zip(qs, subs):
        by_subject[s].append(q)
    samples = []
    for s in ["common", "kaoyan", "cs", "employment"]:
        for q in by_subject.get(s, [])[:PER_SUBJECT]:
            samples.append({"question": q, "subject": s})
    # 再补 2 条题库外的问题（验证两条链路的降级行为是否一致）
    samples.append({"question": "量子计算在金融领域有哪些应用？", "subject": "outside"})
    samples.append({"question": "红烧肉怎么做最好吃？", "subject": "outside"})
    return samples


async def run_full(qa, question):
    trace = {}
    t0 = time.perf_counter()
    parts = []
    async for text, _is_complete in qa.query(question, None, None, trace):
        parts.append(text)
    return "".join(parts), trace, (time.perf_counter() - t0) * 1000


def run_rag(rag, question):
    trace = {}
    t0 = time.perf_counter()
    ans = "".join(rag.generate_answer(question, None, [], trace))
    return ans, trace, (time.perf_counter() - t0) * 1000


async def main():
    samples = build_samples()
    print("=" * 78)
    print(f"实验 A4 · 完整链路 vs 纯 RAG    样本 {len(samples)} 条"
          f"（每学科 {PER_SUBJECT} 条 + 题库外 2 条）")
    print("=" * 78)

    t0 = time.perf_counter()
    qa = IntegratedQASystem()
    rag = qa.rag  # 复用同一实例，避免加载两份向量/精排模型
    print(f"系统构造耗时 {time.perf_counter() - t0:.1f} s\n")

    rows = []
    for i, item in enumerate(samples, 1):
        q = item["question"]
        full_ans, full_trace, full_ms = await run_full(qa, q)
        rag_ans, rag_trace, rag_ms = run_rag(rag, q)

        row = {
            "index": i,
            "question": q,
            "subject": item["subject"],
            "full": {
                "layer": full_trace.get("layer"),
                "faq_score": full_trace.get("faq_score"),
                "matched_question": full_trace.get("matched_question"),
                "category": full_trace.get("category"),
                "n_contexts": len(full_trace.get("contexts") or []),
                "latency_ms": round(full_ms, 1),
                "answer": full_ans,
            },
            "rag_only": {
                "category": rag_trace.get("category"),
                "n_contexts": len(rag_trace.get("contexts") or []),
                "latency_ms": round(rag_ms, 1),
                "answer": rag_ans,
            },
            "speedup": round(rag_ms / full_ms, 2) if full_ms > 0 else None,
        }
        rows.append(row)
        print(f"[{i:>2}/{len(samples)}] [{item['subject']:<10}] {q}")
        print(f"        完整链路 layer={row['full']['layer']:<8} {full_ms:>8.0f} ms"
              f"  ctx={row['full']['n_contexts']}")
        print(f"        纯RAG    category={row['rag_only']['category']}  {rag_ms:>8.0f} ms"
              f"  ctx={row['rag_only']['n_contexts']}")
        print(f"        加速比 {row['speedup']}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)

    faq_rows = [r for r in rows if r["full"]["layer"] == "faq"]
    rag_rows = [r for r in rows if r["full"]["layer"] == "rag"]
    fb_rows = [r for r in rows if r["full"]["layer"] == "fallback"]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"完整链路分流：FAQ {len(faq_rows)} 条 / RAG {len(rag_rows)} 条 / 兜底 {len(fb_rows)} 条")
    print(f"完整链路平均延迟 {avg([r['full']['latency_ms'] for r in rows]):.0f} ms")
    print(f"纯 RAG  平均延迟 {avg([r['rag_only']['latency_ms'] for r in rows]):.0f} ms")
    if faq_rows:
        print(f"\nFAQ 命中那 {len(faq_rows)} 条：完整链路均值 "
              f"{avg([r['full']['latency_ms'] for r in faq_rows]):.0f} ms  vs  "
              f"纯 RAG 均值 {avg([r['rag_only']['latency_ms'] for r in faq_rows]):.0f} ms"
              f"  → 快 {avg([r['rag_only']['latency_ms'] for r in faq_rows]) / avg([r['full']['latency_ms'] for r in faq_rows]):.1f} 倍")

    # 纯 RAG 对题库内问题答得如何（用字符重合粗判，不做语义评测）
    print("\n纯 RAG 链路对「题库内问题」的分类结果：")
    cat = collections.Counter(r["rag_only"]["category"] for r in rows)
    for k, v in cat.items():
        print(f"  {k}: {v} 条")
    print("（被 BERT 判为[通用知识]的问题，纯 RAG 链路根本不检索，直接让 LLM 自由发挥）")

    payload = {
        "sample_size": len(samples),
        "per_subject": PER_SUBJECT,
        "summary": {
            "layer_distribution": dict(collections.Counter(r["full"]["layer"] for r in rows)),
            "avg_full_ms": round(avg([r["full"]["latency_ms"] for r in rows]), 1),
            "avg_rag_only_ms": round(avg([r["rag_only"]["latency_ms"] for r in rows]), 1),
            "faq_hit_count": len(faq_rows),
            "faq_hit_avg_full_ms": round(avg([r["full"]["latency_ms"] for r in faq_rows]), 1),
            "faq_hit_avg_rag_ms": round(avg([r["rag_only"]["latency_ms"] for r in faq_rows]), 1),
        },
        "rows": rows,
    }
    path = os.path.join(OUT_DIR, "exp_a4.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    asyncio.run(main())
