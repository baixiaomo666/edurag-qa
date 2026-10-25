"""
评估闭环 第 ⑭ 步：真实跑一遍「FAQ + RAG 双层系统」，采集答案、命中层与检索上下文。

【为什么必须单独有这一步】
原 ragas_evaluate.py 直接从 rag_evaluate_data.json 读取 answer / context 去打分，
数据完全不经过系统。后果是：改动任何检索参数（reranker / 分块 / 融合权重 /
上下文数量 M），系统行为确实变了，但喂给 RAGAS 的数据没变 → 打分结果一模一样，
对照实验不可能成立。本脚本把「系统的真实行为」导出成文件，评估器才第一次能看到
变量的影响。在这一步存在之前，任何"调参带来提升"的结论都无法被验证。

【为什么跑 IntegratedQASystem 而不是只跑 RAGSystem】
用户真实提问先经过 FAQ 关键词层，只有降级才进 RAG。只测 RAGSystem 会得到两个
失真结论：① FAQ 能秒回的标准问答对，在指标里被算成"检索失败 / 答非所问"；
② 系统级命中率（FAQ 命中率 + RAG 有效率）根本无处可测。
所以这里跑完整链路，并逐条记录走了哪一层（layer）。
代价是必须按层分流：FAQ 命中样本没有 contexts，而 RAGAS 的
context_precision / context_recall 在空上下文上无定义，混在一起算会污染均值。

【三类样本的划分】
  layer == "faq"              → faq_records：标准问答对命中，用答案相似度类指标
  layer == "rag" 且上下文非空   → records    ：RAGAS 四指标可算
  layer == "rag" 但上下文为空   → skipped    ：BERT 判为通用知识走直答，context 指标无定义

【在链路中的位置】
    ⑬ 评测集 → ⑭ 本脚本 → ⑮ eval_ragas.py → ⑯ Bad Case 定位 → 回到检索层改参数

【用法】在项目根目录执行：
    python rag_qa/rag_assesment/eval_pipeline.py --tag system_baseline
    python rag_qa/rag_assesment/eval_pipeline.py --tag system_baseline --limit 5
"""
import argparse
import asyncio
import difflib
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------- 路径处理
# 为什么必须手动注入 sys.path：
# base / rag_qa 是项目根目录下的顶层包，而本文件位于 rag_qa/rag_assesment/。
# 执行 python rag_qa/rag_assesment/eval_pipeline.py 时，sys.path[0] 是
# rag_qa/rag_assesment 而不是项目根，此时 `import base` 会 ModuleNotFoundError。
# 用 __file__ 反推项目根目录，保证从任意工作目录启动都能正确导入。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.config import config                    # noqa: E402

# ---------------------------------------------------------------- 日志落点
# 必须在 `from base.logger import ...` 之前改 config.LOG_FILE：
# base/logger.py:8 的签名是 `def setup_logging(log_file=config.LOG_FILE)`，
# 默认参数在模块导入（函数定义执行）时就求值，紧接着 :42 立刻调用 setup_logging()，
# 于是 FileHandler 会马上去打开 config.ini 里写的 logs/app.log。
# 本机工作区的 logs/ 目录对子进程不可写，不改这一步脚本会在 import 阶段就
# PermissionError 退出（连 --help 都看不到）。项目根目录可写，所以把日志落到
# 根目录下的这个文件；日志级别与格式完全不变，只是换了落点。
config.LOG_FILE = "eval_run.log"

from base.logger import logger                    # noqa: E402
from faq_milvus import IntegratedQASystem         # noqa: E402

# 评测集与输出目录用绝对路径，避免「在哪个目录执行」影响结果。
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(HERE, "kaoyan_qa_100.json")
# 产物必须落在可写目录：rag_qa/rag_assesment/ 对本机子进程是只读的，
# 写 runs/ 会 permission denied。项目根可写，所以统一放到 <项目根>/eval_runs/。
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "eval_runs")


def _norm(text):
    """去掉所有空白，用于中文包含判断（"40 人" 与 "40人" 应视为相同）"""
    return re.sub(r"\s+", "", text or "")


def _numbers(text):
    """抽取文本里的数字事实（分数、年份、人数、比例、学制等）"""
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _number_coverage(ground_truth, answer):
    """参考答案里的数字有多少出现在 FAQ 返回的答案中；参考答案没有数字时返回 None。

    为什么需要这个指标（实测得出，不是理论推演）：
    FAQ 返回的是题库里的标准答案，评测集的 ground_truth 是同一知识点的另一种
    写法，二者措辞不同、几乎从不构成字符串包含关系。实测 4 条通用题：
    相似度 0.70 / 0.72 / 0.78 / 0.90，而 contains_gt 全部为 False ——
    单看"是否包含原文"会把正确答案全判成错。
    数字是硬事实（学制 3 年、总分 500 分、录取 40 人），改写时必须保持一致，
    所以"参考答案里的数字是否都出现"比字符串包含可靠得多。
    """
    gt_nums = _numbers(ground_truth)
    if not gt_nums:
        return None
    return len(gt_nums & _numbers(answer)) / len(gt_nums)


async def run_one(qa, question, source_filter=None):
    """
    跑一条问题，返回 (完整答案, trace, 耗时秒)。

    ① query() 是异步生成器（FAQ 分支每字 await asyncio.sleep(0.01) 模拟打字），
      必须 async for 消费；
    ② 必须 join 成完整字符串——拿到的是分片，不是答案；
    ③ trace 是调用方与系统之间的唯一约定：传空 dict 进去，FAQ 层与 RAG 层
      各自把中间产物回填进来。生产链路 app.py 不传，所以对线上零影响——
      可观测性不能改变被测对象，否则实验测的就不是真实系统。
    """
    trace = {}
    parts = []
    start = time.perf_counter()
    async for text, is_complete in qa.query(question, source_filter, None, trace):
        parts.append(text)
        if is_complete:
            break
    # perf_counter 是单调时钟，不受系统时间调整影响，适合测耗时
    cost_ms = (time.perf_counter() - start) * 1000
    return "".join(parts), trace, cost_ms


async def collect(questions_path, out_path, tag, limit=None, source_filter=None):
    """逐条跑完整链路，并按层分流落盘。"""

    # ---------------- 1. 读评测集 ----------------
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # limit 用于"先跑 5 条确认链路通"，避免一上来就付出全部 LLM 调用成本。
    if limit:
        questions = questions[:limit]

    logger.info(f"评测集={questions_path} 本次将评估 {len(questions)} 条 tag={tag}")

    # ---------------- 2. 构造被测系统（只构造一次） ----------------
    # 放在循环外是硬性要求，不是优化：IntegratedQASystem.__init__ 会加载 BM25 题库、
    # BERT 分类器、BGE-M3、bge-reranker-large 与 LLM client，
    # 放进循环意味着 N 条样本付出 N 次模型加载代价。
    qa = IntegratedQASystem()

    faq_records = []   # FAQ 层直接命中（无上下文）
    rag_records = []   # RAG 层且有真实上下文 → RAGAS 可算
    skipped = []       # RAG 层但被分类为通用知识，走直答，无上下文

    # ---------------- 3. 逐条采集 ----------------
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        answer, trace, cost_ms = await run_one(qa, question, source_filter)

        layer = trace.get("layer")
        contexts = trace.get("contexts") or []
        category = trace.get("category")

        base = {
            "index": index,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "layer": layer,                                   # faq / rag / fallback
            "category": category,                             # RAG 层才有：通用知识 / 专业咨询
            "latency_ms": round(cost_ms, 1),
            "faq_score": trace.get("faq_score"),               # FAQ 归一化置信度
            "matched_question": trace.get("matched_question"),  # FAQ 命中的标准题
            "reject_reason": trace.get("reject_reason"),       # FAQ 拒绝原因（归因用）
        }

        if layer == "faq":
            # FAQ 命中：答案是题库里的标准答案，没有检索上下文。
            # 用「与 ground_truth 的字符相似度」和「是否包含标准答案」做可计算指标，
            # 而不是硬塞进 RAGAS：context_* 在空上下文上无定义；
            # faithfulness 衡量的是"生成内容是否忠于证据"，对直接复制的标准答案无意义。
            base["similarity"] = round(
                difflib.SequenceMatcher(None, _norm(ground_truth), _norm(answer)).ratio(), 4
            )
            base["contains_gt"] = _norm(ground_truth) in _norm(answer)
            # 数字覆盖度：参考答案的数字是否都出现在 FAQ 答案里。
            # contains_gt 是严格口径（要求原文包含），改写过的标准答案会被判 False，
            # 所以必须同时给出这个更鲁棒的判据；参考答案本身无数字时为 None。
            base["number_coverage"] = _number_coverage(ground_truth, answer)
            faq_records.append(base)
            logger.info(f"[{index}/{len(questions)}] FAQ命中 score={base['faq_score']} "
                        f"{cost_ms:.0f}ms {question}")

        elif contexts:
            base["contexts"] = contexts                        # → RAGAS: contexts
            base["n_contexts"] = len(contexts)
            # 上下文规模用字符数：中文 1 字约合 1 token 量级；
            # 精确 token 数需额外加载 tokenizer，组间对比用字符数已经够用。
            base["context_chars"] = sum(len(c) for c in contexts)
            rag_records.append(base)
            logger.info(f"[{index}/{len(questions)}] RAG n_contexts={len(contexts)} "
                        f"{cost_ms:.0f}ms {question}")

        else:
            # layer == "rag" 但没有任何上下文：BERT 判定为通用知识，系统走直答分支。
            # 这类样本上 context_precision / context_recall 无定义，强行算会污染均值。
            # 它反映的是"评测集里混入了公共知识题"，所以单独计数、单独落盘、最后报占比，
            # 而不是静默丢弃。
            skipped.append(base)
            logger.warning(f"[{index}/{len(questions)}] 跳过（无上下文）layer={layer} "
                           f"category={category} {question}")

    # ---------------- 4. 组装并写出 ----------------
    # meta 记录本次运行的全部可变量。为什么必须写：一周后不可能记得"这组的阈值是几、
    # reranker 开没开"；缺少 meta 的结果文件无法与其他组对比、无法复现，等于废数据。
    meta = {
        "tag": tag,
        "questions_path": questions_path,
        "system": "IntegratedQASystem（FAQ 关键词层 + RAG 检索层完整链路）",
        "source_filter": source_filter,
        "faq_threshold": config.FAQ_THRESHOLD,        # FAQ 命中阈值（随题库重新标定）
        "llm_model": config.LLM_MODEL,
        # 以下来自 config.ini，是真正可调的分块/检索参数
        "parent_chunk_size": config.PARENT_CHUNK_SIZE,
        "child_chunk_size": config.CHILD_CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "retrieval_k": config.RETRIEVAL_K,
        "candidate_m": config.CANDIDATE_M,
        # 以下当前是代码常量，不在 config.ini 里，改它们要动源码。
        # 记录在这里是为了让结果文件自带"当时的代码是什么行为"这一信息。
        "code_constants": {
            "weighted_ranker": "WeightedRanker(0.7, 1.0) @ rag_qa/core/vector_store.py:247",
            "context_source": "parent_content（父块回溯）@ rag_qa/core/vector_store.py:288-299",
            "entity_guard": "校名/年份不一致则降级RAG @ mysql_qa/retrieval/bm25_search.py",
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def avg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 1) if rows else 0.0

    both = faq_records + rag_records
    payload = {
        "tag": tag,
        "meta": meta,
        "total_questions": len(questions),
        "faq_hits": len(faq_records),
        "rag_with_context": len(rag_records),
        "skipped_no_context": len(skipped),
        "layer_distribution": {
            "faq": len(faq_records),
            "rag_with_context": len(rag_records),
            "rag_direct_answer": len(skipped),
        },
        # 整体均值保持数字类型（历史结果文件与 eval_ragas.py 都读这个字段），
        # 分层均值另开一个字段，避免破坏下游解析。
        "avg_latency_ms": round(sum(r["latency_ms"] for r in both) / len(both), 1) if both else 0.0,
        "avg_latency_by_layer": {
            "faq": avg(faq_records, "latency_ms"),
            "rag": avg(rag_records, "latency_ms"),
        },
        "records": rag_records,        # ← eval_ragas.py 的打分输入（字段与历史版本一致）
        "faq_records": faq_records,    # ← FAQ 层单独统计，不进 RAGAS
        "skipped": skipped,
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------------- 5. 打印摘要 ----------------
    print(f"\n===== 采集完成 tag={tag}（FAQ+RAG 完整链路） =====")
    print(f"  问答题总数            {len(questions)}")
    print(f"  FAQ 层命中             {len(faq_records)}")
    print(f"  RAG 层（有上下文）      {len(rag_records)}")
    print(f"  RAG 层（直答无上下文）   {len(skipped)}")
    if faq_records:
        hit = sum(1 for r in faq_records if r["contains_gt"])
        sim = sum(r["similarity"] for r in faq_records) / len(faq_records)
        print(f"  FAQ 答案含标准答案      {hit}/{len(faq_records)} ({hit / len(faq_records):.0%})")
        print(f"  FAQ 答案相似度均值      {sim:.3f}")
        nc = [r["number_coverage"] for r in faq_records if r.get("number_coverage") is not None]
        if nc:
            print(f"  FAQ 数字覆盖率均值      {sum(nc) / len(nc):.0%}（{len(nc)} 条参考答案含数字可比对）")
    if rag_records:
        print(f"  RAG 平均上下文条数      {sum(r['n_contexts'] for r in rag_records) / len(rag_records):.2f}")
        print(f"  RAG 平均上下文字数      {sum(r['context_chars'] for r in rag_records) / len(rag_records):.0f}")
    print(f"  平均延迟              FAQ {payload['avg_latency_by_layer']['faq']:.0f} ms / "
          f"RAG {payload['avg_latency_by_layer']['rag']:.0f} ms")
    print(f"  结果文件              {out_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="评估采集：真实跑 FAQ+RAG 完整链路，导出 answer / contexts / 命中层")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="评测集 JSON 路径，默认同目录的 kaoyan_qa_100.json")
    parser.add_argument("--out", default=None,
                        help="结果输出路径；不传则写到 <项目根>/eval_runs/<tag>.json")
    parser.add_argument("--tag", default="system_baseline",
                        help="本次实验标签，如 system_baseline / faq_off（默认 system_baseline）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条，用于先验证链路是否通")
    parser.add_argument("--source", default=None,
                        help="学科过滤，默认不过滤")
    args = parser.parse_args()

    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f"{args.tag}.json")
    asyncio.run(collect(args.questions, out_path, args.tag, args.limit, args.source))


if __name__ == "__main__":
    main()
