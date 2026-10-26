# -*- coding: utf-8 -*-
"""
C 组实验公共模块 · 采样 / 强制下沉 / 缓存清理 / 指标
================================================================================
C1（检索策略消融）与 C2（Prompt 版本消融）共用以下四项约定，任何一项不在同一
口径上，两组结果就无法与 A/B 组横向比较。本模块把它们集中到一个地方：

1. 采样口径：与 _exp_b2.py / _exp_b2_llm.py 完全一致
   SAMPLES_PER_SUBJECT = {"kaoyan": 8, "cs": 7, "employment": 5} → 20 条
   采样源、顺序、学科配比都照抄，保证 C 组与 B 组的 20 条是同一批题。

2. 「强制下沉 RAG」：这是 C 组最关键、也最容易被误读的一步。
   实测（_c_common.verify_faq_sink 可复现）：这 20 条题取自 MySQL 题库原题，
   在 config.ini 的 faq_threshold=0.68 下 **20/20 全部命中 FAQ 层**
   （置信度 0.979~1.000，因为就是原题查原题）。
   若不做处理，完整链路下 20 条会被 FAQ 层秒回、contexts 恒为空，
   C1 检索层没有样本、C2 的 RAGAS 样本为 0，两组实验双双失效。

   因此 C 组把 FAQ 层在进程内强制失效：config.FAQ_THRESHOLD = 1.01。
   BM25Search.search 在每次调用时读 config.FAQ_THRESHOLD（bm25_search.py:138），
   而 softmax 输出恒 ≤ 1.0，故 best_score < 1.01 恒成立 → 全部走
   `reject_reason="below_threshold"` → (None, True) → 进入 RAG 层。
   * 只改进程内对象，不改 config.ini，脚本退出即恢复；
   * 主题代码零改动，被测量的链路本身没有被修改；
   * 代价：FAQ 层从「把关」变成「直通」，因此 C 组结论只覆盖 RAG 层，
     不代表线上以 FAQ 为主体的端到端表现（详见报告「口径声明」）。

3. Redis 答案缓存必须清：FAQ 层命中后会写 `answer:<问题原文>`（TTL 3600s），
   而 BM25Search.search 第 140 行 **在读阈值之前** 就查缓存。
   后果：C2 三轮 prompt 用同一批题，v2/v3 会直接吃到 v1 写入的缓存答案，
   根本不进 RAG，prompt_version 的影响被缓存完全吃掉，三轮数字一模一样。
   故每组运行前必须清 `answer:*`；`qa_*` 快照键与 `sys_*` 不在清理范围内。

4. 指标：照抄 _exp_b2.py 的 numbers / core_terms / process_text 口径，
   保证 C 组检索层指标与 exp_b2.json 逐字段可比。
"""
import collections
import json
import os
import re
import statistics
import tempfile

from base.config import config

# 与 _exp_b2.py:30 / _exp_b2_llm.py:37 同一模式：必须在 import base.logger 之前
# 改日志落点。logs/app.log 对子进程只读（logs/ 目录为空），沿用既有脚本的做法。
config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_c.log")

from mysql_qa.retrieval.bm25_search import BM25Search          # noqa: E402
from mysql_qa.util.preprocess import process_text              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
QA_PATH = os.path.join(HERE, "rag_qa", "rag_assesment", "kaoyan_qa_100.json")

# 照抄 B 组：同一批 20 条专业题
SAMPLES_PER_SUBJECT = {"kaoyan": 8, "cs": 7, "employment": 5}
EXPECTED_SAMPLES = sum(SAMPLES_PER_SUBJECT.values())   # 20
# > softmax 最大值（1.0），使 FAQ 层对所有问题都判 below_threshold
FAQ_SINK_THRESHOLD = 1.01

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------- 1. 强制下沉
def force_faq_sink(log=True):
    """进程内使 FAQ 层对全部问题失效，强制所有请求下沉到 RAG 层。

    返回 (old, new)，便于实验结束后恢复。
    """
    old = config.FAQ_THRESHOLD
    config.FAQ_THRESHOLD = FAQ_SINK_THRESHOLD
    if log:
        print(f"[口径] FAQ 层强制失效：config.FAQ_THRESHOLD {old} → "
              f"{FAQ_SINK_THRESHOLD}（仅进程内，config.ini 不变）")
    return old, FAQ_SINK_THRESHOLD


def restore_faq_threshold(old):
    config.FAQ_THRESHOLD = old


# ---------------------------------------------------------------- 2. 样本
def build_samples(verbose=True):
    """与 _exp_b2.py:68-79 完全一致的采样；额外做两件事：

    * 断言样本数 = 20（bucket 键不匹配 / subjects 缺失时立刻报错，而不是静默跑空）；
    * 报告 ground_truth 是否为空的样本数。评测集扩到 120 条（新增 20 条通用
      常识题）后 ground_truth 已补齐，实测采样 20 条中为空 0 条；此项保留是
      防御性的——将来若加入无标准答案的题，指标层会把 None 静默跳过。
    """
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

    if verbose:
        print(f"[样本] 题库 {len(bm.original_questions)} 条，评测集键 {len(qa)} 条；"
              f"各学科候选池 "
              f"{ {k: len(bucket.get(k, [])) for k in SAMPLES_PER_SUBJECT} }")
        print(f"[样本] 采样 {len(samples)} 条"
              f"{dict(collections.Counter(s['subject'] for s in samples))}")

    if len(samples) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"采样数 {len(samples)} != 期望 {EXPECTED_SAMPLES}。"
            f"通常是题库问题与 {os.path.basename(QA_PATH)} 的 question 字段不再一致，"
            f"或 Redis 的 qa_subjects 快照缺失。候选池："
            f"{ {k: len(bucket.get(k, [])) for k in SAMPLES_PER_SUBJECT} }"
        )

    empty_gt = [s["question"] for s in samples if not (s["ground_truth"] or "").strip()]
    if verbose:
        print(f"[样本] ground_truth 为空 {len(empty_gt)} 条"
              f"（这类题的词/数字召回为 None，会从均值分母里剔除）")
        for q in empty_gt:
            print(f"        · {q}")
    return samples


def samples_as_questions_json(samples, path):
    """把采样写成 eval_pipeline 需要的 [{question, ground_truth}] JSON。"""
    data = [{"question": s["question"], "ground_truth": s["ground_truth"]}
            for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------- 3. 缓存清理
def clear_answer_cache(verbose=True):
    """清 `answer:*`（FAQ 层答案缓存）。qa_* 快照与 sys_* 生产数据不动。

    为什么不直接调用 RedisClient：需要按前缀 scan，且要能看到清了多少键，
    以便在报告里证明"缓存没有跨轮污染"这件事确实做了。
    """
    import redis
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT,
                    db=config.REDIS_DB, decode_responses=True)
    keys = list(r.scan_iter("answer:*", count=1000))
    if keys:
        r.delete(*keys)
    if verbose:
        print(f"[清理] Redis answer:* 已清 {len(keys)} 个键"
              f"（防止 prompt/策略轮次之间串味）")
    return len(keys)


# ---------------------------------------------------------------- 4. 指标
def numbers(text):
    """参考答案/上下文里的数字集合（照抄 _exp_b2.py:59-60）"""
    return set(NUM_RE.findall(text or ""))


def core_terms(text):
    """核心词：长度≥3 的非数字实词（照抄 _exp_b2.py:63-65）"""
    return {t for t in process_text(text or "") if len(t) >= 3 and not t.isdigit()}


def context_metrics(ground_truth, contexts):
    """检索层指标，口径与 exp_b2.json 逐字段一致，便于与 B 组直接对比。"""
    ctx = "\n".join(contexts or [])
    gt_nums, ctx_nums = numbers(ground_truth), numbers(ctx)
    gt_terms = set(process_text(ground_truth or ""))
    ctx_terms = set(process_text(ctx))
    gt_core = core_terms(ground_truth)
    return {
        "n_ctx": len(contexts or []),
        "ctx_chars": len(ctx),
        "recall_num": (len(gt_nums & ctx_nums) / len(gt_nums)) if gt_nums else None,
        "recall_term": (len(gt_terms & ctx_terms) / len(gt_terms)) if gt_terms else None,
        "rank_hit": (1.0 if (gt_core & ctx_terms) else 0.0) if gt_core else None,
    }


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def summarize_retrieval(rows):
    """检索层汇总；与 _exp_b2.py:119-130 的 summarize 同键名。"""
    vals = [r["recall_term"] for r in rows if r.get("recall_term") is not None]
    return {
        "n_questions": len(rows),
        "n_scored": len(vals),
        "recall_term_mean": mean_of(rows, "recall_term"),
        "recall_term_median": statistics.median(vals) if vals else None,
        "recall_term_full": (sum(1 for v in vals if v == 1.0) / len(vals)) if vals else None,
        "recall_num_mean": mean_of(rows, "recall_num"),
        # rank_hit 的口径是"上下文是否命中参考答案核心词"，参考答案无核心词时记为 None，
        # 因此 key 存在但值可能为 None；这与 _exp_b2.py 的 summarize 同键名。
        "rank_hit_rate": mean_of(rows, "rank_hit"),
        "avg_rank_hit": mean_of(rows, "rank_hit"),
        "avg_ctx_chars": statistics.mean([r["ctx_chars"] for r in rows]),
        "avg_n_ctx": statistics.mean([r["n_ctx"] for r in rows]),
        "latency_median_ms": statistics.median([r["latency_ms"] for r in rows]),
    }


def fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "-"
