# -*- coding: utf-8 -*-
"""
C 组实验（一）· 检索策略消融 —— direct / auto / hyde / subquery
================================================================================
待验证的命题
--------------------------------------------------------------------------------
StrategySelector.strategy_select 针对每条问题在三种检索策略间路由：
    直接检索 / 假设问题检索（HyDE）/ 子查询检索
而 `auto`（默认）要先花一次 LLM 调用让模型自己选。核心问题是：

  Q1  auto 的策略选择是否优于"永远直接检索"？即多出的那次 LLM 延迟换来了什么？
  Q2  HyDE / 子查询各自把哪类问题的检索质量拉起来了？
  Q3  auto 实际选了哪些策略？它是否真的分布到了 HyDE / 子查询上？

测量口径（三点必须在报告里同时声明，否则结论会被误读）
--------------------------------------------------------------------------------
1. **走完整链路**：被测对象是 IntegratedQASystem（FAQ 关键词层 + RAG 检索层），
   与 C2 和 eval_pipeline 同源；不是只调 VectorStore。
2. **FAQ 层被强制失效**：这 20 条是 MySQL 题库原题，默认阈值 0.68 下 20/20
   被 FAQ 层秒回，检索层根本没有样本。故进程内 config.FAQ_THRESHOLD=1.01
   让 FAQ 层直通（详见 _c_common.force_faq_sink）。因此本实验只反映 **RAG 层**
   的策略差异，不代表线上以 FAQ 为主体的端到端表现。
3. **只测检索层，测量生成层被替换为空流**：四组共 80 次请求，若真跑生成
   （父块上下文约 1786 字）每组答案长度不同，端到端延迟会把"答案写多长"算进
   策略成本，无法归因。故把 rag.llm.stream 换成空流：
       · QueryClassifier（BERT 分类）      —— 真实执行
       · StrategySelector（策略选择 + 检索）—— 真实执行，含其内部 LLM 调用
       · 答案生成                          —— 跳过（C2 负责生成层）
   注意 HyDE / 子查询 / auto 各自的 LLM 调用**属于检索策略的成本，未被替换**，
   由 _call_llm 计时器单独统计。

指标
--------------------------------------------------------------------------------
检索层（与 exp_b2.json 同键名，可直接与 B 组对比）：
    recall_term  参考答案实词覆盖率（主要判据）
    recall_num   参考答案数字覆盖率
    rank_hit     上下文是否命中参考答案核心词
    ctx_chars    送进 prompt 的上下文字数
    latency_ms   完整链路单题耗时（FAQ 层已直通，生成层已跳过）
策略层（C 组特有）：
    strategy_dist    auto 组实际选中的策略分布
    llm_calls / llm_ms  策略选择与 HyDE/子查询生成各自花掉的 LLM 次数与耗时

用法
--------------------------------------------------------------------------------
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_c1.py
  # 先验证链路（5 条，约 1 分钟）
  & "...\\python.exe" -X utf8 _exp_c1.py --limit 5
"""
import argparse
import asyncio
import collections
import json
import os
import statistics
import time

import _c_common as C

from faq_milvus import IntegratedQASystem                            # noqa: E402
from rag_qa.core.strategy_selector import StrategySelector           # noqa: E402
from rag_qa.rag_assesment.eval_pipeline import run_one               # noqa: E402

GROUPS = [
    ("C1 direct（强制直接检索）", "direct"),
    ("C1 auto（现状·LLM 选策略）", "auto"),
    ("C1 hyde（强制假设问题检索）", "hyde"),
    ("C1 subquery（强制子查询检索）", "subquery"),
]
STRATEGY_KEY = {"direct": "直接检索", "hyde": "假设问题检索", "subquery": "子查询检索"}
STRATEGY_NAMES = {"直接检索", "假设问题检索", "子查询检索"}


class _StrategyLLMTimer:
    """替换 StrategySelector._call_llm：只计时计数，不改行为，顺带抓策略名。

    为什么要包在这里而不是看总延迟：auto / hyde / subquery 三组各含 1 次策略层
    LLM 调用（auto 是"选策略"，hyde/subquery 是"生成假设答案 / 子问题"），这次
    调用是策略成本的主体。把它单独量出来，才能回答"auto 多花的那一次调用值不值"，
    也才能把检索本身（向量库耗时）与策略开销分开。

    为什么策略名要在这里抓：RAGSystem.generate_answer 每次请求都新建
    StrategySelector（rag_system.py:89），调用方拿不到那个实例，也就读不到
    strategy_select 挂在实例上的 last_strategy（strategy_selector.py:74）。
    auto 模式下 _call_llm 的返回值就是策略名，在此捕获是唯一不侵入被测代码的做法。
    """

    def __init__(self):
        self.calls = 0
        self.total_ms = 0.0
        self._current = None

    def wrap(self, original):
        def timed(_self_inner, prompt, *a, **kw):
            t0 = time.perf_counter()
            try:
                out = original(_self_inner, prompt, *a, **kw)
            finally:
                self.calls += 1
                self.total_ms += (time.perf_counter() - t0) * 1000
            # auto：返回策略名；hyde/subquery：返回生成的文本，不匹配任何策略名
            if isinstance(out, str) and out.strip() in STRATEGY_NAMES:
                self._current = out.strip()
            return out
        return timed

    def note_forced(self, mode):
        """强制模式下 last_strategy 由开关直接决定，不经过 LLM。"""
        self._current = STRATEGY_KEY.get(mode)

    def winner(self):
        return self._current

    def reset(self):
        self.calls = 0
        self.total_ms = 0.0
        self._current = None


class _SilentLLM:
    """替换被测实例的 llm：不产生任何 token，让生成层成本为 0。

    只 patch 被测实例（qa.rag.llm），不动 RAGSystem 类：类级赋值会污染
    同进程内其他 RAGSystem 实例，而 C1/C2 后续若要同进程跑就无法区分。
    """

    def stream(self, prompt):
        return iter(())


async def run_group(qa, samples, mode, timer):
    """跑一组：设置策略开关 → 逐题走完整链路 → 采集检索层指标。"""
    os.environ["EXPERIMENT_STRATEGY"] = mode
    # 每组前清 answer:*：BM25Search.search 在读阈值之前就查缓存，
    # 不清则第 2 组起会直接返回上一组的缓存答案，且环境变量切换失效。
    C.clear_answer_cache(verbose=False)
    timer.reset()
    rows = []
    try:
        for i, s in enumerate(samples, 1):
            # 每题的策略都重新判定：强制模式由开关决定，auto 由 _call_llm 捕获
            timer.note_forced(mode)
            _, trace, cost_ms = await run_one(qa, s["question"], None)
            actual = timer.winner()
            m = C.context_metrics(s["ground_truth"], trace.get("contexts"))
            row = {
                "question": s["question"],
                "subject": s["subject"],
                "layer": trace.get("layer"),
                "category": trace.get("category"),
                "forced_strategy": mode,
                "actual_strategy": actual,
                "faq_reject_reason": trace.get("reject_reason"),
                "latency_ms": round(cost_ms, 1),
                "answer_chars": 0,          # 生成层被跳过，恒 0
                **m,
            }
            rows.append(row)
            print(f"    [{i:>2}/{len(samples)}] {row['actual_strategy'] or '-':<8} "
                  f"ctx={row['n_ctx']} 条/{row['ctx_chars']:>5} 字  "
                  f"词召回 {C.fmt(row['recall_term'])}  "
                  f"核心词 {C.fmt(row['rank_hit'], '.0f')}  "
                  f"{cost_ms:>6.0f} ms  {s['question'][:26]}")
    finally:
        os.environ.pop("EXPERIMENT_STRATEGY", None)
    return rows


def summarize(rows, timer):
    s = C.summarize_retrieval(rows)
    n = len(rows)
    s["strategy_dist"] = dict(collections.Counter(str(r["actual_strategy"]) for r in rows))
    s["llm_calls_total"] = timer.calls
    s["llm_ms_total"] = round(timer.total_ms, 1)
    # 单题均摊：auto/hyde/subquery 每题 1 次调用，direct 为 0 次
    s["llm_ms_per_question"] = round(timer.total_ms / n, 1) if n else None
    s["layer_dist"] = dict(collections.Counter(str(r["layer"]) for r in rows))
    s["category_dist"] = dict(collections.Counter(str(r["category"]) for r in rows))
    return s


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（验证链路）")
    ap.add_argument("--tag", default="exp_c1", help="输出文件名（不含 .json）")
    args = ap.parse_args()

    print("=" * 108)
    print("C 组实验（一）· 检索策略消融   direct / auto / hyde / subquery")
    print("=" * 108)

    samples = C.build_samples()
    if args.limit:
        samples = samples[:args.limit]
        print(f"[样本] --limit 生效，仅跑前 {len(samples)} 条")

    C.force_faq_sink()
    print("[口径] 生成层已跳过（rag.llm.stream → 空流）；策略层 LLM 调用真实发生并单独计时\n")

    qa = IntegratedQASystem()
    # 只替换被测实例的生成层：链路的分类、策略选择、检索全部真实执行
    qa.rag.llm = _SilentLLM()
    rags = _StrategyLLMTimer()
    StrategySelector._call_llm = rags.wrap(StrategySelector._call_llm)

    print(f"共 {len(GROUPS)} 组 × {len(samples)} 题\n")
    results = {}
    for label, mode in GROUPS:
        print(f">>> {label}（EXPERIMENT_STRATEGY={mode}）")
        rows = await run_group(qa, samples, mode, rags)
        sm = summarize(rows, rags)
        results[label] = {"mode": mode, "summary": sm, "rows": rows}
        print(f"    → 词召回 {C.fmt(sm['recall_term_mean'])}  满召回 "
              f"{C.fmt(sm['recall_term_full'], '.1%')}  核心词 {C.fmt(sm['rank_hit_rate'], '.2%')}  "
              f"ctx {sm['avg_ctx_chars']:>6.0f} 字  中位延迟 {sm['latency_median_ms']:>6.0f} ms  "
              f"策略层 LLM {sm['llm_calls_total']} 次/{sm['llm_ms_total']:.0f} ms")
        print(f"      策略分布 {sm['strategy_dist']}  分层 {sm['layer_dist']}\n")

    # -------- 汇总表 --------
    print("=" * 108)
    print("汇总")
    print("=" * 108)
    print(f"{'分组':<26}{'词召回':<10}{'满召回':<10}{'核心词命中':<12}"
          f"{'上下文均字数':<14}{'中位延迟':<10}{'策略LLM均摊':<12}")
    print("-" * 108)
    for label, _ in GROUPS:
        s = results[label]["summary"]
        print(f"{label:<26}{C.fmt(s['recall_term_mean']):<10}"
              f"{C.fmt(s['recall_term_full'], '.1%'):<10}"
              f"{C.fmt(s['rank_hit_rate'], '.2%'):<12}"
              f"{s['avg_ctx_chars']:<14.0f}{s['latency_median_ms']:<10.0f}"
              f"{s['llm_ms_per_question']:<12.0f}")
    print("-" * 108)

    # -------- auto 的策略选择是否真的分流 --------
    auto = results[GROUPS[1][0]]["summary"]
    print(f"\nauto 组策略分布：{auto['strategy_dist']}")
    dist = auto["strategy_dist"]
    non_direct = sum(v for k, v in dist.items() if k != "直接检索")
    print(f"  其中非「直接检索」{non_direct}/{auto['n_questions']} 条 —— "
          f"若为 0，说明 auto 退化成 direct，HyDE/子查询在本题集上从不触发")

    # -------- 逐题差异：direct vs auto，抓出 auto 真正改判的题 --------
    def diff(a_label, b_label, threshold=0.02):
        print(f"\n逐题差异：{b_label}  相对  {a_label}（词召回变化 >{threshold}）")
        ra = {r["question"]: r for r in results[a_label]["rows"]}
        rb = {r["question"]: r for r in results[b_label]["rows"]}
        hit = 0
        for q, a in ra.items():
            b = rb.get(q)
            if not b or a["recall_term"] is None or b["recall_term"] is None:
                continue
            if abs(a["recall_term"] - b["recall_term"]) > threshold:
                hit += 1
                print(f"  [{a['recall_term']:.3f} → {b['recall_term']:.3f}] "
                      f"({a['actual_strategy']} → {b['actual_strategy']}) {q}")
        if hit == 0:
            print("  （无显著变化）")

    diff(GROUPS[0][0], GROUPS[1][0])
    diff(GROUPS[0][0], GROUPS[2][0])
    diff(GROUPS[0][0], GROUPS[3][0])

    path = os.path.join(C.OUT_DIR, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "C1 检索策略消融",
            "sample_size": len(samples),
            "samples_by_subject": dict(collections.Counter(s["subject"] for s in samples)),
            "meta": {
                "chain": "IntegratedQASystem（FAQ 层 + RAG 检索层完整链路）",
                "faq_forced_off": True,
                "faq_threshold_forced": C.FAQ_SINK_THRESHOLD,
                "generation_skipped": "rag.llm.stream 被替换为空流（只测检索层）",
                "samples_per_subject": C.SAMPLES_PER_SUBJECT,
                "qa_path": os.path.basename(C.QA_PATH),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "groups": {k: {"mode": v["mode"], "summary": v["summary"], "rows": v["rows"]}
                       for k, v in results.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    asyncio.run(main())
