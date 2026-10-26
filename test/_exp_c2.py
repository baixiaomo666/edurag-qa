# -*- coding: utf-8 -*-
"""
C 组实验（二）· Prompt 版本消融 v1 / v2 / v3
================================================================================
待验证的命题
--------------------------------------------------------------------------------
项目里有三份 prompt 模板并存（`rag_qa/prompts/prompt_v1.py`、`prompt_v2.py`、
`prompt_v3.py`），历史上只比过"字数/可读性"这类主观印象，没有 RAGAS 量化。
B 组留下的靶子很明确：`faithfulness 0.797` 是四个指标里最低的那个，v2 的严格
约束（只依据上下文、没有就说不知道）正是冲着它去的。本实验固定知识库、固定
检索参数，只换 prompt 模板，回答：

  Q1  三代 prompt 的 RAGAS 四指标孰优？差距是否超出噪声？
  Q2  v2/v3 提 faithfulness 的代价是什么？是否压低了 answer_relevancy
      （过度保守 → 答非所问）——这个权衡本身就是简历素材。
  Q3  版本间差异是"全面变好"还是"此消彼长"？

测量口径（三点必须在报告里同时声明，否则结论会被误读）
--------------------------------------------------------------------------------
1. **走完整链路**：被测对象是 IntegratedQASystem（FAQ 关键词层 + RAG 检索层），
   与 C1 和 eval_pipeline 同源；不是只调 VectorStore 或只调 prompt。
2. **FAQ 层被强制失效**：20 条采样题全部是 MySQL 题库原题，默认阈值 0.68 下
   20/20 被 FAQ 层秒回复制标准答案，RAGAS 有效样本恒为 0（FAQ 层答案是题库原文
   复制、无 contexts），无法比较 prompt。故进程内 config.FAQ_THRESHOLD=1.01 让
   FAQ 层直通（详见 _c_common.force_faq_sink）。
   → 本实验测的是"**FAQ 不命中时**，RAG 层三代 prompt 的差异"，
     不等于"线上以 FAQ 为主体的端到端表现"。这一句必须写进报告。
3. **生成层真实执行**：三组都真跑 LLM 生成（否则没有 answer 可评），
   因此端到端延迟可直接横向比较，代价差异是真实的。

产出（关键：必须能直接喂 eval_ragas.py）
--------------------------------------------------------------------------------
  eval_runs/exp_c2_v1.json / _v2.json / _v3.json
      每个版本的 payload 与 `eval_pipeline.py:236-258` 的字段结构逐字段一致
      （tag / meta / total_questions / faq_hits / rag_with_context /
        skipped_no_context / layer_distribution / avg_latency_ms /
        avg_latency_by_layer / records / faq_records / skipped），
      因此可以直接跑：eval_ragas.py eval_runs/exp_c2_v1.json ... 做横向对比。
  eval_runs/exp_c2.json
      C 组自己的汇总（每版本 summary + RAGAS 四指标 + 逐题明细），便于出报告。
  每版本的 RAGAS 明细 CSV：eval_runs/exp_c2_<v>_ragas.csv（落盘在 payload 旁）

为什么逐题明细要在两个文件里各存一份
--------------------------------------------------------------------------------
payload 那份是给 eval_ragas.py 读的（字段名不能动）；汇总那份带 subject /
recall_term 等 C 组附加字段，是写报告时按学科下钻用的。两者由同一次运行产生，
不会不同步。

用法
--------------------------------------------------------------------------------
  # ① 先预检（默认 2 条 × 3 组，约 1 分钟）：确认链路、RAGAS 都能通
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_c2.py --preflight
  # ② 正式跑（3 组 × 20 题 = 60 次生成 + RAGAS 裁判调用）
  & "...\\python.exe" -X utf8 _exp_c2.py
  # ③ 用项目自带的评测器对三份 payload 横向对比（可选，独立复核）
  & "...\\python.exe" -X utf8 -m rag_qa.rag_assesment.eval_ragas eval_runs/exp_c2_v1.json eval_runs/exp_c2_v2.json eval_runs/exp_c2_v3.json
"""
import argparse
import asyncio
import collections
import difflib
import json
import os
import statistics
import time

import _c_common as C

from base.config import config                                   # noqa: E402
from faq_milvus import IntegratedQASystem                        # noqa: E402
from rag_qa.rag_assesment.eval_pipeline import (                 # noqa: E402
    _norm,                 # FAQ 层相似度口径，复用以免出现第二套定义
    _number_coverage,      # FAQ 层数字覆盖口径，同上
    run_one,               # 消费 qa.query 的唯一权威口径（见 docstring 说明）
)

VERSIONS = ["v1", "v2", "v3"]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# 与 eval_ragas.py:75 同一个本地模型路径。不 import 那个模块（它有模块级
# sys.path 操作等副作用），改为按值同步，并在此注明来源以便日后核对。
EVAL_EMBEDDING_MODEL = r"D:\Program\Develop\bge-m3"


async def run_version(qa, samples, version):
    """跑一个 prompt 版本：逐题走完整链路，同时采集答案与上下文。

    为什么不用 eval_pipeline.collect()：它直接落盘成固定 tag 的文件，且不接受
    "按版本跑三轮"的循环；但它逐条采集的核心是 run_one，这里原样复用，
    以保证口径与 eval_pipeline / C1 完全一致（同一 query 调用、同一计时方式）。
    """
    # 开关名是 EXPERIMENT_PROMPT（rag_qa/core/prompts.py:18 读取），
    # 不是 EXPERIMENT_PROMPT_VERSION —— 写错会让三组全跑 v1，实验静默失效。
    os.environ["EXPERIMENT_PROMPT"] = version
    try:
        rows = []
        for i, s in enumerate(samples, 1):
            answer, trace, cost_ms = await run_one(qa, s["question"], None)
            contexts = trace.get("contexts") or []
            m = C.context_metrics(s["ground_truth"], contexts)
            row = {
                "index": i,
                "question": s["question"],
                "answer": answer,
                "ground_truth": s["ground_truth"],
                "layer": trace.get("layer"),
                "category": trace.get("category"),
                "latency_ms": round(cost_ms, 1),
                "faq_score": trace.get("faq_score"),
                "matched_question": trace.get("matched_question"),
                "reject_reason": trace.get("reject_reason"),
                # ---- C 组附加字段（仅汇总文件使用，不影响 payload 兼容性） ----
                "subject": s["subject"],
                "prompt_version": version,
                "answer_chars": len(answer or ""),
                **m,
            }
            if contexts:
                row["contexts"] = contexts
                row["n_contexts"] = len(contexts)
                row["context_chars"] = sum(len(c) for c in contexts)
            rows.append(row)
            print(f"    [{i:>2}/{len(samples)}] {version}  ctx={row['n_ctx']} 条/"
                  f"{row['ctx_chars']:>5} 字  答 {row['answer_chars']:>4} 字  "
                  f"{cost_ms:>6.0f} ms  {s['question'][:26]}")
        return rows
    finally:
        os.environ.pop("EXPERIMENT_PROMPT", None)


def build_payload(rows, tag, version):
    """把一轮记录组装成与 eval_pipeline.py:236-258 逐字段兼容的 payload。

    为什么必须自己组装而不是只存原始记录：eval_ragas.py 读的是
    run["records"] / run["faq_records"] / run["skipped"] 三个列表 + tag 与
    layer_distribution，字段名错一个就静默拿到 0 样本。这里按层分流的口径
    与 eval_pipeline.py:170-204 完全相同：
        layer == "faq"   → faq_records（答案来自题库，不进 RAGAS）
        有 contexts      → records（RAGAS 打分对象）
        其余（RAG 直答） → skipped（context_* 无定义，强行算会污染均值）
    C2 强制 FAQ 下沉，正常情况下 faq_records 必为空；仍完整生成该字段，
    以便将来换阈值重跑时结构不用改。
    """
    faq_records, rag_records, skipped = [], [], []
    for r in rows:
        base = {k: r[k] for k in (
            "index", "question", "answer", "ground_truth", "layer", "category",
            "latency_ms", "faq_score", "matched_question", "reject_reason",
        )}
        if r["layer"] == "faq":
            answer = r["answer"] or ""
            base["similarity"] = round(
                difflib.SequenceMatcher(
                    None, _norm(r["ground_truth"]), _norm(answer)).ratio(), 4
            )
            base["contains_gt"] = _norm(r["ground_truth"]) in _norm(answer)
            base["number_coverage"] = _number_coverage(r["ground_truth"], answer)
            faq_records.append(base)
        elif r.get("contexts"):
            base["contexts"] = r["contexts"]
            base["n_contexts"] = r["n_contexts"]
            base["context_chars"] = r["context_chars"]
            rag_records.append(base)
        else:
            skipped.append(base)

    def avg(rs, key):
        return round(sum(x[key] for x in rs) / len(rs), 1) if rs else 0.0

    both = faq_records + rag_records
    meta = {
        "tag": tag,
        "questions_path": C.QA_PATH,
        "system": "IntegratedQASystem（FAQ 关键词层 + RAG 检索层完整链路）",
        "source_filter": None,
        # 这里是本次运行真实生效的阈值（1.01），如实记录：结果文件必须自带
        # "当时的 FAQ 层是关着的"这一信息，否则后人会按 FAQ 在线来解读。
        "faq_threshold": config.FAQ_THRESHOLD,
        "faq_forced_off": True,
        "prompt_version": version,
        "llm_model": config.LLM_MODEL,
        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
        "child_chunk_size": config.CHILD_CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_k": config.RETRIEVAL_K,
        "candidate_m": config.CANDIDATE_M,
        "code_constants": {
            "weighted_ranker": "WeightedRanker(0.7, 1.0) @ rag_qa/core/vector_store.py:247",
            "context_source": "parent_content（父块回溯）@ rag_qa/core/vector_store.py:288-299",
            "entity_guard": "校名/年份不一致则降级RAG @ mysql_qa/retrieval/bm25_search.py",
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {
        "tag": tag,
        "meta": meta,
        "total_questions": len(rows),
        "faq_hits": len(faq_records),
        "rag_with_context": len(rag_records),
        "skipped_no_context": len(skipped),
        "layer_distribution": {
            "faq": len(faq_records),
            "rag_with_context": len(rag_records),
            "rag_direct_answer": len(skipped),
        },
        "avg_latency_ms": round(sum(r["latency_ms"] for r in both) / len(both), 1) if both else 0.0,
        "avg_latency_by_layer": {
            "faq": avg(faq_records, "latency_ms"),
            "rag": avg(rag_records, "latency_ms"),
        },
        "records": rag_records,
        "faq_records": faq_records,
        "skipped": skipped,
    }


def run_ragas(rows, tag):
    """对一轮样本跑 RAGAS 四指标，返回 ({metric: mean}, csv_path)。"""
    from datasets import Dataset
    from langchain.chat_models import init_chat_model
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        # contexts 必须是「列表的列表」：每条样本一个字符串列表（eval_ragas.py:205-210）
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    # 裁判 LLM：沿用与 eval_ragas.py 完全相同的 DeepSeek 配置，保证可比。
    # RAGAS 会用它拆句判断忠实度、生成反事实问题测相关性。
    llm = init_chat_model(
        model=config.LLM_MODEL,
        model_provider="openai",
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )
    embeddings = HuggingFaceEmbeddings(model_name=EVAL_EMBEDDING_MODEL)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    csv_path = os.path.join(C.OUT_DIR, f"{tag}_ragas.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    means = df.mean(numeric_only=True)   # pandas mean 默认 skipna，单条 NaN 不会毁整列
    return {n: (float(means[n]) if n in means else None) for n in METRIC_NAMES}, csv_path


def summarize(rows):
    s = C.summarize_retrieval(rows)
    answers = [r["answer_chars"] for r in rows]
    s["answer_chars_mean"] = statistics.mean(answers) if answers else None
    s["answer_chars_median"] = statistics.median(answers) if answers else None
    s["empty_answers"] = sum(1 for a in answers if a == 0)
    s["layer_dist"] = dict(collections.Counter(str(r["layer"]) for r in rows))
    s["category_dist"] = dict(collections.Counter(str(r["category"]) for r in rows))
    s["latency_mean_ms"] = statistics.mean([r["latency_ms"] for r in rows]) if rows else None
    return s


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（验证链路）")
    ap.add_argument("--preflight", action="store_true",
                    help="预检：等价于 --limit 2，确认链路与 RAGAS 都通")
    ap.add_argument("--tag", default="exp_c2", help="汇总文件名（不含扩展名）")
    ap.add_argument("--no-ragas", action="store_true", help="只跑链路不跑 RAGAS（调试用）")
    args = ap.parse_args()

    print("=" * 108)
    print("C 组实验（二）· Prompt 版本消融   v1 / v2 / v3   （含 RAGAS 四指标）")
    print("=" * 108)

    if args.preflight and args.limit is None:
        args.limit = 2
        print("[预检] --preflight：只跑 2 条 × 3 版本，确认链路与 RAGAS 都能通")

    samples = C.build_samples()
    if args.limit:
        samples = samples[:args.limit]
        print(f"[样本] --limit 生效，仅跑前 {len(samples)} 条")

    C.force_faq_sink()
    print("[口径] 生成层真实执行；FAQ 层被强制直通，故本实验 = "
          "FAQ 不命中时 RAG 层的 prompt 差异\n")

    qa = IntegratedQASystem()
    print(f"共 {len(VERSIONS)} 组 × {len(samples)} 题\n")

    results = {}
    for version in VERSIONS:
        print(f">>> prompt {version}（EXPERIMENT_PROMPT={version}）")
        # 每组前清 answer:*：缓存 key 虽含 prompt 版本，但清掉可避免跨版本残留；
        # 更关键的是 Redis 里可能残留上一次实验写的答案，those 会绕过本轮 prompt。
        C.clear_answer_cache(verbose=False)
        rows = await run_version(qa, samples, version)
        payload = build_payload(rows, f"{args.tag}_{version}", version)
        outp = os.path.join(C.OUT_DIR, f"{args.tag}_{version}.json")
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        sm = summarize(rows)
        results[version] = {"summary": sm, "payload": payload, "payload_path": outp,
                            "rows": rows}
        print(f"    → 词召回 {C.fmt(sm['recall_term_mean'])}  ctx {sm['avg_ctx_chars']:>6.0f} 字  "
              f"答 {sm['answer_chars_mean']:>6.0f} 字  中位延迟 {sm['latency_median_ms']:>6.0f} ms  "
              f"RAG有上下文 {payload['rag_with_context']}/{payload['total_questions']}")
        print(f"      落盘 {outp}\n")

    # -------- RAGAS --------
    if not args.no_ragas:
        for version in VERSIONS:
            rows = [r for r in results[version]["rows"] if r.get("contexts")]
            if not rows:
                print(f">>> RAGAS 跳过 prompt {version}：没有带上下文的样本\n")
                results[version]["ragas"] = None
                continue
            print(f">>> RAGAS 评分 · prompt {version}（{len(rows)} 条样本，需多次裁判 LLM 调用）")
            metrics, csv_path = run_ragas(rows, f"{args.tag}_{version}")
            results[version]["ragas"] = metrics
            results[version]["ragas_csv"] = csv_path
            print("    " + "  ".join(f"{k}={C.fmt(v)}" for k, v in metrics.items()))
            print(f"    明细 CSV → {csv_path}\n")

    # -------- 汇总表 --------
    print("=" * 108)
    print("汇总")
    print("=" * 108)
    print(f"{'版本':<6}{'faithful':<11}{'answer_rel':<12}{'ctx_prec':<11}"
          f"{'ctx_recall':<12}{'答案均字数':<12}{'中位延迟':<10}{'词召回':<9}")
    print("-" * 108)
    for version in VERSIONS:
        r = results[version]
        m = r.get("ragas") or {}
        s = r["summary"]
        print(f"{version:<6}{C.fmt(m.get('faithfulness')):<11}"
              f"{C.fmt(m.get('answer_relevancy')):<12}"
              f"{C.fmt(m.get('context_precision')):<11}"
              f"{C.fmt(m.get('context_recall')):<12}"
              f"{s['answer_chars_mean']:<12.0f}{s['latency_median_ms']:<10.0f}"
              f"{C.fmt(s['recall_term_mean']):<9}")
    print("-" * 108)

    # -------- 检索层一致性检查 --------
    # 三组共用同一条检索链路，context_* 与检索层指标理论上应几乎相同。
    # 若极差明显不为 0，那不是 prompt 的效果，而是检索不稳定或裁判波动，
    # 报告里必须如实归因，不能算成"v3 让检索变好了"。
    print("\n检索层一致性检查（三组共用检索链路，差异应≈0；不为 0 即检索/裁判波动，"
          "不可归因给 prompt）")
    for k in ["avg_n_ctx", "avg_ctx_chars", "recall_term_mean", "rank_hit_rate"]:
        vals = [results[v]["summary"].get(k) for v in VERSIONS]
        nums = [x for x in vals if isinstance(x, (int, float))]
        spread = (max(nums) - min(nums)) if len(nums) >= 2 else None
        print(f"  {k:<20} " + "  ".join(f"{v}={C.fmt(x)}" for v, x in zip(VERSIONS, vals))
              + (f"   极差={spread:.4g}" if spread is not None else ""))

    # -------- 逐题：答案长度差异最大的题，便于人工抽查 --------
    print("\n答案长度逐题对比（|v3-v1| 最大的前 5 题，建议人工读一遍答案）")
    by_q = {v: {r["question"]: r for r in results[v]["rows"]} for v in VERSIONS}
    pairs = []
    for q, r1 in by_q["v1"].items():
        r3 = by_q["v3"].get(q)
        if r3:
            pairs.append((abs(r3["answer_chars"] - r1["answer_chars"]), q, r1, r3))
    for d, q, r1, r3 in sorted(pairs, reverse=True)[:5]:
        print(f"  v1 {r1['answer_chars']:>4} 字 → v3 {r3['answer_chars']:>4} 字  {q[:40]}")

    path = os.path.join(C.OUT_DIR, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "C2 Prompt 版本消融",
            "sample_size": len(samples),
            "samples_by_subject": dict(collections.Counter(s["subject"] for s in samples)),
            "meta": {
                "chain": "IntegratedQASystem（FAQ 层 + RAG 检索层完整链路）",
                "faq_forced_off": True,
                "faq_threshold_forced": C.FAQ_SINK_THRESHOLD,
                "generation_executed": True,
                "ragas_metrics": METRIC_NAMES,
                "samples_per_subject": C.SAMPLES_PER_SUBJECT,
                "qa_path": os.path.basename(C.QA_PATH),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "usage_note": "本文件是 C 组汇总；喂 eval_ragas.py 请用 exp_c2_<v>.json",
            },
            "versions": {v: {"summary": results[v]["summary"],
                             "ragas": results[v].get("ragas"),
                             "ragas_csv": results[v].get("ragas_csv"),
                             "payload_path": results[v]["payload_path"],
                             "rows": results[v]["rows"]}
                         for v in VERSIONS},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已写入：{path}")
    print("逐版本 payload（可直接喂 eval_ragas.py）：")
    for v in VERSIONS:
        print(f"  {results[v]['payload_path']}")


if __name__ == "__main__":
    asyncio.run(main())
