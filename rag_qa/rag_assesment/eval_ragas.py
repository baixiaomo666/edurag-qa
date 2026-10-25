"""
评估闭环 第 ⑮ 步：对 eval_pipeline 采到的真实数据跑 RAGAS 四指标，并统计 FAQ 层表现。

【为什么和第 ⑭ 步拆成两个脚本】
两步的成本结构和失败模式完全不同：
  ⑭ 采集：要加载 BERT + BGE-M3 + reranker，要逐条调 LLM 生成答案 —— 贵、慢、依赖 Milvus。
  ⑮ 打分：只用 RAGAS 调 LLM + Embedding —— 相对便宜，依赖的是评测集与指标配置。
拆开之后：
  1) 同一份 eval_runs/*.json 可以反复重打分（换指标、换评测 LLM、调 prompt），不必重跑系统；
  2) 出问题时能快速判断是"采集坏了"还是"打分坏了"；
  3) 采集产物落盘即成为证据——"这些 context 确实是系统检索出来的"可以被复查。
这是评估流水线的标准做法：把「数据采集」与「评分」解耦。

【为什么 FAQ 层指标和 RAGAS 指标必须分开算】
完整链路有两条出口：
  FAQ 命中  → 直接返回题库里的标准答案，没有 contexts
  RAG 降级  → 检索父块 + LLM 生成，有 contexts
而 RAGAS 的四个指标全部建立在"存在检索上下文"这个前提上：
  context_precision / context_recall 在空上下文上无定义；
  faithfulness 衡量的是"生成内容是否忠于给定证据"，对直接复制的标准答案同样没有意义。
所以 FAQ 命中样本绝不能混进 RAGAS 求平均，只能用它自己的指标：
  答案是否包含标准答案（contains_gt）、与标准答案的字符相似度（similarity）。

【为什么必须同时看"层分布"】
改了 FAQ 阈值之后，样本会在两层之间迁移：
阈值调低 → 更多问题被 FAQ 拦下 → 进入 RAG 的样本变少、剩下的偏难。
此时若只看 RAGAS 分数，会得到"分数变好了"的假象——真实原因是那批答不好的样本
根本没进 RAG。层分布（FAQ / RAG有上下文 / RAG直答）是识别这种迁移的唯一依据，
而样本数不同时 RAGAS 分数本来就不可直接比较。

【为什么支持一次评多组】
对照实验要跑多组（reranker 消融、上下文粒度、融合权重、上下文数量、策略路由、FAQ 阈值），
逐组跑完再手工抄数字横向对比，既慢又容易抄错，而且抄完就丢失了过程记录。
所以本脚本接受多个采集结果，一次评完并自动汇总：

    python rag_qa/rag_assesment/eval_ragas.py                          # 只评默认 system_baseline
    python rag_qa/rag_assesment/eval_ragas.py eval_runs/a.json eval_runs/b.json

传多个时会额外打印一张横向对比表，并把每组指标追加进 eval_runs/summary.csv。
同一 tag 重跑会覆盖旧行，所以反复调试不会让汇总表越堆越乱。
"""
import argparse
import csv
import json
import os
import sys

# 第三方依赖：不依赖项目根目录，放在 sys.path 注入之前更规范
from datasets import Dataset
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：答案是否只依据给定上下文（测幻觉）
    answer_relevancy,    # 答案相关性：是否正面回答了问题
    context_precision,   # 上下文精准度：检索结果中有效证据的占比（测检索纯度）
    context_recall,      # 上下文召回率：该找到的证据是否被找到（测检索完备性）
)

# 与 eval_pipeline.py 相同的路径处理：base 包在项目根下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.config import config      # noqa: E402

# 与 eval_pipeline.py 同样的理由：必须在导入 base.logger 之前改掉日志落点，
# 否则 base/logger.py:26 会去打开只读的 logs/app.log 并抛 PermissionError。
config.LOG_FILE = "eval_run.log"

from base.logger import logger      # noqa: E402

# RAGAS 计算 answer_relevancy 需要 Embedding 模型。
# 这里与历史脚本 ragas_evaluate.py 保持完全一致——换模型会让新结果与历史结果不可比。
EVAL_EMBEDDING_MODEL = r"D:\Program\Develop\bge-m3"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# 汇总表落点：与 eval_pipeline 的采集结果同目录，多组实验结果天然放在一起
SUMMARY_CSV = os.path.join(PROJECT_ROOT, "eval_runs", "summary.csv")

# 默认采集结果：与 eval_pipeline.py 的 --tag 默认值保持同名
DEFAULT_RUN = os.path.join(PROJECT_ROOT, "eval_runs", "system_baseline.json")


def _faq_stats(faq_records):
    """FAQ 层统计。

    单独成函数的原因：这部分不依赖 RAGAS，即使一条 RAG 样本都没有也能算，
    而"全场都被 FAQ 拦下"恰恰是调低阈值后可能出现的真实状态，不能因为没有
    RAGAS 分数就丢掉 FAQ 层的结论。
    """
    if not faq_records:
        return {
            "faq_hits": 0,
            "faq_contains_gt": 0,
            "faq_contains_gt_rate": None,
            "faq_avg_similarity": None,
            "faq_number_coverage_avg": None,
            "faq_number_verifiable": 0,
        }
    hit = sum(1 for r in faq_records if r.get("contains_gt"))
    sims = [r["similarity"] for r in faq_records if r.get("similarity") is not None]
    # 数字覆盖度：只统计"参考答案里确实含数字"的样本。
    # 为什么不能只看 contains_gt：FAQ 答案与参考答案是同一知识点的两种写法，
    # 措辞不同、几乎从不字符串包含，contains_gt 会把正确答案判成 False
    # （实测相似度 0.70~0.90 而 contains_gt 全 False）。数字是硬事实，改写必须一致。
    ncs = [r["number_coverage"] for r in faq_records if r.get("number_coverage") is not None]
    return {
        "faq_hits": len(faq_records),
        "faq_contains_gt": hit,
        # 严格口径：参考答案原文是否出现在 FAQ 答案里（通常偏低，仅作参考）
        "faq_contains_gt_rate": hit / len(faq_records),
        # 相似度均值：反映 FAQ 命中的"接近程度"，误命中（同模板不同实体）会拉低它
        "faq_avg_similarity": (sum(sims) / len(sims)) if sims else None,
        "faq_number_coverage_avg": (sum(ncs) / len(ncs)) if ncs else None,
        "faq_number_verifiable": len(ncs),
    }


def score(run_path, out_csv=None):
    """读取 eval_pipeline 的产物，跑 RAGAS 四指标 + FAQ 层统计，打印并输出明细 CSV。

    :return: dict 本次的结构化结果，供多组对比与汇总落盘使用
    """

    # 先给一句人话提示，而不是让 FileNotFoundError 的 traceback 糊一屏：
    # 直接点运行本文件、但还没跑过 eval_pipeline.py 时，这是最可能撞上的错误。
    if not os.path.exists(run_path):
        raise SystemExit(f"找不到采集结果：{run_path}\n"
                         f"请先运行 eval_pipeline.py（默认输出到 <项目根>/eval_runs/system_baseline.json）")

    # ---------------- 1. 读采集结果 ----------------
    with open(run_path, "r", encoding="utf-8") as f:
        run = json.load(f)

    records = run.get("records") or []              # RAG 有上下文样本 → 进 RAGAS
    faq_records = run.get("faq_records") or []      # FAQ 命中样本 → 单独统计
    skipped = run.get("skipped") or []              # RAG 直答（无上下文）
    layer_dist = run.get("layer_distribution") or {}

    tag = run.get("tag", os.path.basename(run_path))
    logger.info(f"载入 {run_path}：tag={tag} RAG样本 {len(records)} FAQ样本 {len(faq_records)} 跳过 {len(skipped)}")

    faq_stats = _faq_stats(faq_records)

    # ---------------- 2. 层分布（先打印，再决定要不要跑 RAGAS） ----------------
    # 层分布永远可算，且它是解读后面所有分数的前提，所以放在最前面。
    print(f"\n===== 评估 tag={tag} =====")
    if layer_dist:
        print(f"  【层分布】FAQ命中 {layer_dist.get('faq', len(faq_records))} / "
              f"RAG有上下文 {layer_dist.get('rag_with_context', len(records))} / "
              f"RAG直答 {layer_dist.get('rag_direct_answer', len(skipped))}")
    if faq_stats["faq_hits"]:
        rate = faq_stats["faq_contains_gt_rate"]
        sim = faq_stats["faq_avg_similarity"]
        ncv = faq_stats["faq_number_coverage_avg"]
        nvr = faq_stats["faq_number_verifiable"]
        if sim is not None:
            print(f"  【FAQ层】命中 {faq_stats['faq_hits']} 条，相似度均值 {sim:.3f}")
        else:
            print(f"  【FAQ层】命中 {faq_stats['faq_hits']} 条")
        if ncv is not None:
            print(f"          数字覆盖率 {ncv:.0%}（{nvr} 条参考答案含数字可比对）")
        if rate is not None:
            print(f"          严格口径原文包含率 {rate:.0%}"
                  f"（FAQ 答案为同义改写时通常偏低，仅供参考）")

    # ---------------- 3. 没有 RAG 样本时的分支 ----------------
    if not records:
        # 两种可能，必须区分清楚，不能笼统报错：
        #   ① 全部被 FAQ 层命中  → 结构完全正常，只是 RAGAS 无对象可评；
        #   ② 全部被跳过         → 评测集混入了公共知识题，是数据问题。
        # 原来的实现无条件 raise SystemExit，会在第 ① 种情况下误报成故障。
        if faq_records:
            reason = "全部样本被 FAQ 层命中，RAG 没有任何样本 → RAGAS 无对象可评"
        else:
            reason = ("既没有 FAQ 命中也没有 RAG 上下文 → 评测集可能全是公共知识题，"
                      "请检查 kaoyan_qa_100.json")
        print(f"  【RAG层】{reason}")
        logger.warning(reason)

        empty = {
            "tag": tag,
            "run": os.path.abspath(run_path),
            "samples": 0,
            "skipped": len(skipped),
            "metrics": {n: None for n in METRIC_NAMES},
            "avg_latency_ms": run.get("avg_latency_ms", 0.0) or 0.0,
            "avg_contexts": 0.0,
            "avg_context_chars": 0.0,
            "csv": None,
            "layer_distribution": layer_dist,
            **{k: v for k, v in faq_stats.items()},
        }
        return empty

    # ---------------- 4. 转成 RAGAS 要求的四字段格式 ----------------
    # RAGAS 的固定字段名是 question / answer / contexts / ground_truth。
    # 其中 contexts 必须是「列表的列表」：每条样本对应一个字符串列表。
    # eval_pipeline 落盘的 records[i]["contexts"] 本身已是 List[str]，
    # 所以整列拼出来自然就是 List[List[str]]——
    # 这正是 ragas_evaluate.py 中 item["context"] 那一行在做的事，
    # 唯一区别是：那份 context 是人工预写的，这份是系统真实检索出来的。
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    # ---------------- 5. 初始化 RAGAS 自己需要的 LLM 与 Embedding ----------------
    # 注意：这里的 llm 是「裁判」，不是被测系统用的那个模型。
    # RAGAS 靠它做指标判定（例如 faithfulness 要把答案拆句、逐句判断是否被上下文支持）。
    # 沿用同一套 DeepSeek 配置，保证与历史结果可比。
    llm = init_chat_model(
        model=config.LLM_MODEL,
        model_provider="openai",
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )
    embeddings = HuggingFaceEmbeddings(model_name=EVAL_EMBEDDING_MODEL)

    # ---------------- 6. 执行 RAGAS 评估 ----------------
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()

    # ---------------- 7. 打印指标 + 落盘明细 ----------------
    # CSV 用 utf-8-sig：带 BOM 才能在 Excel 里正确显示中文；
    # pandas 再读时 BOM 会被自动识别，不影响后续处理。
    out_csv = out_csv or os.path.splitext(run_path)[0] + "_ragas.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # pandas 的 mean() 默认 skipna=True，会自动跳过个别样本产生的 NaN，
    # 不会因为一条样本判定失败就把整列分数变成 NaN。
    means = df.mean(numeric_only=True)

    print(f"  【RAG层 · RAGAS】有效样本 {len(records)}")
    for name in METRIC_NAMES:
        if name in means:
            print(f"    {name:20s} {means[name]:.4f}")
        else:
            print(f"    {name:20s} （结果中无此列，请检查 RAGAS 版本的指标命名）")

    # 延迟与上下文规模不来自 RAGAS，而来自 eval_pipeline 的采集结果。
    # 它们必须和四个质量指标一起看——只看质量不看成本，结论是片面的。
    latency = [r["latency_ms"] for r in records]
    avg_latency = sum(latency) / len(latency)
    avg_contexts = sum(r["n_contexts"] for r in records) / len(records)
    avg_chars = sum(r["context_chars"] for r in records) / len(records)
    print(f"    {'avg_latency_ms':20s} {avg_latency:.0f}")
    print(f"    {'avg_contexts':20s} {avg_contexts:.2f}")
    print(f"    {'avg_context_chars':20s} {avg_chars:.0f}")
    print(f"  明细 CSV → {out_csv}")

    summary = {
        "tag": tag,
        "run": os.path.abspath(run_path),
        "samples": len(records),
        "skipped": len(skipped),
        "metrics": {n: (float(means[n]) if n in means else None) for n in METRIC_NAMES},
        "avg_latency_ms": avg_latency,
        "avg_contexts": avg_contexts,
        "avg_context_chars": avg_chars,
        "csv": os.path.abspath(out_csv),
        "layer_distribution": layer_dist,
    }
    summary.update(faq_stats)
    return summary


def print_compare(summaries):
    """把多组结果并排打印。

    为什么值得单独做一个对比表：单看某一组的分数无法判断调参有没有用，
    只有把 baseline 与实验组放在同一行、同一列看，差值才是可解释的证据。

    层分布必须出现在这张表里：改了 FAQ 阈值后样本会在两层之间迁移，
    只看 RAGAS 分数可能把"难样本被 FAQ 拦走"误读成"RAG 变好了"。
    """
    names = [s["tag"] for s in summaries]
    base = names[0]

    print("\n" + "=" * 96)
    print(f"对照实验汇总（第一列 {base} 为基准组，括号内为相对基准的差值）")
    print("=" * 96)
    header = f"{'指标':<24}" + "".join(f"{n:>17}" for n in names)
    print(header)
    print("-" * 96)

    rows = [(m, "metrics", m) for m in METRIC_NAMES]
    rows += [
        ("样本数(RAG)", "samples", None),
        ("跳过数", "skipped", None),
        ("FAQ命中数", "faq_hits", None),
        ("FAQ含标准答案率", "faq_contains_gt_rate", None),
        ("FAQ相似度均值", "faq_avg_similarity", None),
        ("avg_latency_ms", "avg_latency_ms", None),
        ("avg_contexts", "avg_contexts", None),
        ("avg_context_chars", "avg_context_chars", None),
    ]

    for label, key, metric in rows:
        cells = []
        for i, s in enumerate(summaries):
            v = s["metrics"].get(metric) if metric else s.get(key)
            base_v = summaries[0]["metrics"].get(metric) if metric else None
            if v is None:
                cells.append(f"{'—':>17}")
            elif i == 0 or metric is None:
                cells.append(f"{v:>17.4f}" if metric else f"{v:>17}")
            elif base_v is None:
                # 基准组没有该项（例如基准组全被 FAQ 命中、没有 RAG 样本）
                # 此时算差值没有意义，只显示本组数值
                cells.append(f"{v:>17.4f}")
            else:
                cells.append(f"{v:>10.4f}({v - base_v:+.3f})")
        print(f"{label:<24}" + "".join(cells))
    print("-" * 96)
    print("提醒：样本数不同的两组不可直接比较——先确认层分布与跳过数一致，再看指标差值。")


def append_summary(summaries, path=SUMMARY_CSV):
    """把本次各组结果追加进汇总 CSV；同 tag 的行会被替换，避免反复调试堆叠。

    用标准库 csv 而不是 pandas：这个动作很简单，没必要为此引入额外依赖。
    """
    fields = (["tag", "samples", "skipped"] + METRIC_NAMES
              + ["faq_hits", "faq_contains_gt", "faq_contains_gt_rate", "faq_avg_similarity",
                 "faq_number_coverage_avg", "faq_number_verifiable",
                 "layer_faq", "layer_rag_context", "layer_rag_direct",
                 "avg_latency_ms", "avg_contexts", "avg_context_chars", "run", "csv"])

    old_rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            old_rows = [r for r in csv.DictReader(f) if r.get("tag")]

    new_tags = {s["tag"] for s in summaries}
    old_rows = [r for r in old_rows if r["tag"] not in new_tags]

    for s in summaries:
        ld = s.get("layer_distribution") or {}
        row = {
            "tag": s["tag"],
            "samples": s["samples"],
            "skipped": s["skipped"],
            "faq_hits": s.get("faq_hits", 0),
            "faq_contains_gt": s.get("faq_contains_gt", 0),
            "faq_contains_gt_rate": _round_or_blank(s.get("faq_contains_gt_rate"), 4),
            "faq_avg_similarity": _round_or_blank(s.get("faq_avg_similarity"), 4),
            "faq_number_coverage_avg": _round_or_blank(s.get("faq_number_coverage_avg"), 4),
            "faq_number_verifiable": s.get("faq_number_verifiable", 0),
            "layer_faq": ld.get("faq", s.get("faq_hits", 0)),
            "layer_rag_context": ld.get("rag_with_context", s["samples"]),
            "layer_rag_direct": ld.get("rag_direct_answer", s["skipped"]),
            "avg_latency_ms": round(s["avg_latency_ms"], 1),
            "avg_contexts": round(s["avg_contexts"], 2),
            "avg_context_chars": round(s["avg_context_chars"], 1),
            "run": s["run"],
            "csv": s["csv"] or "",
        }
        for n in METRIC_NAMES:
            v = s["metrics"][n]
            row[n] = "" if v is None else round(v, 4)
        old_rows.append(row)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # utf-8-sig 同上：Excel 打开中文表头不乱码
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old_rows)
    print(f"\n汇总表 → {path}（共 {len(old_rows)} 组）")


def _round_or_blank(value, digits):
    """None 写成空字符串——CSV 里不能出现 'None' 字面量，否则 Excel 里无法做数值统计。"""
    return "" if value is None else round(value, digits)


def main():
    parser = argparse.ArgumentParser(
        description="对 eval_pipeline 的采集结果跑 RAGAS 四指标 + FAQ 层统计")
    # nargs="*" 而不是 "?"：不传参数时仍评默认组（保持"在 PyCharm 里直接点运行"可用），
    # 传多个路径时则逐个评完并汇总——多组对照实验一次跑完。
    parser.add_argument("runs", nargs="*", default=None,
                        help="eval_pipeline 产出的 JSON，可传多个；"
                             "默认 <项目根>/eval_runs/system_baseline.json")
    parser.add_argument("--out", default=None,
                        help="明细 CSV 输出路径，仅在只评一组时生效；默认 <run>_ragas.csv")
    parser.add_argument("--no-summary", action="store_true",
                        help="不写汇总表（只临时看一组结果时用）")
    args = parser.parse_args()

    # 默认路径用绝对路径（PROJECT_ROOT 拼出来）：
    # PyCharm 默认把工作目录设成脚本所在目录，用相对路径 eval_runs/xxx.json 会找不到文件。
    runs = args.runs or [DEFAULT_RUN]

    summaries = []
    for run_path in runs:
        summaries.append(score(run_path, args.out if len(runs) == 1 else None))

    if len(summaries) > 1:
        print_compare(summaries)
    if not args.no_summary:
        append_summary(summaries)

    return summaries


if __name__ == "__main__":
    main()
