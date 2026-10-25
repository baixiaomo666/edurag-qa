# -*- coding: utf-8 -*-
"""
A 组实验 A1 / A2 / A3 —— FAQ 层三问
================================================================
A1 阈值扫描   : 全库 softmax 的阈值在不同取值下，判别力（正样本命中率 / 负样本误命中率）如何
A2 学科隔离   : 打开学科掩码后，跨学科误命中是否减少，且命中是否命中"自己"
A3 实体校验   : 实体一致性校验能拦下多少"同模板不同实体"的误命中，代价是多少正常命中

设计原则（重要）：
  * 纯计算，不调用 LLM、不写任何项目文件、不改 config.ini
  * 复现 search() 的分数计算路径（分词 → BM25 → 可选学科掩码 → softmax）
  * 阈值由参数传入，不依赖 config.FAQ_THRESHOLD，因此可以一次跑完七档
  * 不经过 search() 的答案缓存（answer:{query}），避免上一次实验污染下一次

用法：
  & "D:\\Program\\Develop\\Anaconda3\\envs\\edu_rag\\python.exe" -X utf8 _exp_a123.py
"""
import collections
import json
import os
import tempfile

import numpy as np

# ---- 日志重定向必须放在 import base.logger 之前（logs/ 对子进程只读）----
from base.config import config

config.LOG_FILE = os.path.join(tempfile.gettempdir(), "exp_a123.log")

from mysql_qa.retrieval.bm25_search import BM25Search          # noqa: E402
from mysql_qa.util.preprocess import process_text              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_runs")
os.makedirs(OUT_DIR, exist_ok=True)

CALIBRATED = 0.68  # 当前标定阈值，A2/A3 用它做判定

# ============================================================
# 负样本集：构造三类"不应该命中"的问题
#   1) 完全无关      —— 检验最基础的判别力
#   2) 其它考试      —— 同为考试域，但不在本题库覆盖范围
#   3) 同领域题库外  —— 最难的一类：与题库问题用词相近但题库没有
#   4) 同模板不同实体 —— A3 的靶子：词项几乎全重叠，只换校名/年份
# ============================================================
NEG_UNRELATED = [
    "今天天气怎么样？",
    "红烧肉怎么做最好吃？",
    "帮我写一首关于春天的诗",
    "推荐几部好看的电影",
    "怎么减肥比较健康？",
    "明天股市会涨吗？",
    "感冒了吃什么药好？",
    "北京到上海的高铁要多久？",
]
NEG_OTHER_EXAM = [
    "公务员考试怎么报名？",
    "高考总分是多少分？",
    "教师资格证怎么考？",
    "雅思考试报名费多少钱？",
    "注册会计师考试科目有哪些？",
]
NEG_SAME_DOMAIN = [
    "考研报名费是多少钱？",
    "考研英语怎么复习？",
    "复试应该穿什么衣服？",
    "怎么联系导师比较合适？",
    "考研备考需要多长时间？",
    "数学二考哪些内容？",
]
ENTITY_CONFLICT = [
    "青云大学2025年统考录取多少人？",
    "青云大学计算机学硕2025年统考录取多少人？",
    "东海理工大学2026年统考录取多少人？",
    "华清大学2025年统考录取多少人？",
]
NEG_ALL = NEG_UNRELATED + NEG_OTHER_EXAM + NEG_SAME_DOMAIN
NEG_LABEL = {q: "完全无关" for q in NEG_UNRELATED}
NEG_LABEL.update({q: "其它考试" for q in NEG_OTHER_EXAM})
NEG_LABEL.update({q: "同领域题库外" for q in NEG_SAME_DOMAIN})
NEG_LABEL.update({q: "同模板不同实体" for q in ENTITY_CONFLICT})


class Engine:
    """只读的 BM25 判定引擎：复现 search() 的分数路径"""

    def __init__(self):
        self.bm = BM25Search()
        self.questions = list(self.bm.original_questions)
        self.subjects = list(self.bm.subjects or ["common"] * len(self.questions))
        self.n = len(self.questions)

    def scores(self, query, subject=None):
        """
        返回 (softmax 数组, 掩码是否生效)；零词项重叠时返回 (None, False)
        与 search() 第 2~5 步等价，但不含缓存、不含实体校验
        """
        tokens = process_text(query)
        raw = np.array(self.bm.bm25.get_scores(tokens), dtype=float)
        if raw.size == 0 or raw.max() <= 0:
            return None, False
        masked = False
        if subject:
            allowed = np.array([(s == subject) or (s == "common") for s in self.subjects])
            if allowed.size == raw.size and allowed.any():
                raw = np.where(allowed, raw, -np.inf)
                masked = True
        return self.bm._softmax(raw), masked

    def top(self, query, subject=None):
        """返回 (best_idx, best_score, masked)；无匹配时 best_idx=-1"""
        sm, masked = self.scores(query, subject)
        if sm is None:
            return -1, 0.0, masked
        return int(np.argmax(sm)), float(np.nanmax(sm)), masked


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# A1：阈值扫描
# ============================================================
def exp_a1(e):
    hr("实验 A1 · FAQ 阈值扫描（环节 ⑥）")
    thresholds = [0.55, 0.60, 0.65, 0.68, 0.72, 0.80, 0.85]

    # 正样本：题库自身 100 条原题，看它能否命中"自己"
    pos = []
    for i, q in enumerate(e.questions):
        idx, score, _ = e.top(q)
        pos.append({"q": q, "score": score, "idx": idx, "self": idx == i})
    pos_scores = [p["score"] for p in pos]

    # 负样本：四类共 23 条
    neg = []
    for q in NEG_ALL + ENTITY_CONFLICT:
        idx, score, _ = e.top(q)
        neg.append({"q": q, "score": score, "idx": idx, "kind": NEG_LABEL[q]})

    print(f"正样本 {len(pos)} 条（题库原题）   "
          f"负样本 {len(neg)} 条（无关 {len(NEG_UNRELATED)} / 其它考试 {len(NEG_OTHER_EXAM)} / "
          f"同领域 {len(NEG_SAME_DOMAIN)} / 同模板不同实体 {len(ENTITY_CONFLICT)}）")
    print(f"\n正样本置信度：min {min(pos_scores):.4f}  "
          f"中位 {np.median(pos_scores):.4f}  max {max(pos_scores):.4f}")
    print(f"负样本置信度：min {min(n['score'] for n in neg):.4f}  "
          f"max {max(n['score'] for n in neg):.4f}")

    # 按类别看负样本最高分
    print("\n负样本各类最高置信度：")
    by_kind = collections.defaultdict(list)
    for n in neg:
        by_kind[n["kind"]].append(n["score"])
    for kind, scores in by_kind.items():
        print(f"  {kind:<14} max {max(scores):.4f}   "
              f"均值 {np.mean(scores):.4f}   n={len(scores)}")

    print("\n阈值扫描（正样本命中 = 置信度≥阈值且命中自己；负样本误命中 = 置信度≥阈值）")
    print("-" * 78)
    print(f"{'阈值':<8}{'正命中':<10}{'正命中率':<12}{'负误命中':<12}{'负误命中率':<12}{'判别力':<10}")
    print("-" * 78)
    rows = []
    for t in thresholds:
        ph = sum(1 for p in pos if p["score"] >= t and p["self"])
        nh = sum(1 for n in neg if n["score"] >= t)
        pr = ph / len(pos)
        nr = nh / len(neg)
        rows.append({"threshold": t, "pos_hit": ph, "pos_rate": round(pr, 4),
                     "neg_hit": nh, "neg_rate": round(nr, 4),
                     "separation": round(pr - nr, 4)})
        print(f"{t:<8}{ph:<10}{pr:<12.2%}{nh:<12}{nr:<12.2%}{pr - nr:<10.4f}")
    print("-" * 78)

    # 正样本里没命中自己的有多少，为什么
    miss_self = [p for p in pos if p["score"] < CALIBRATED or not p["self"]]
    print(f"\n在阈值 {CALIBRATED} 下未命中自己的正样本：{len(miss_self)} 条")
    for p in miss_self[:8]:
        hit = e.questions[p["idx"]] if p["idx"] >= 0 else "-"
        why = "分数不足" if p["score"] < CALIBRATED else f"命中了别人：{hit}"
        print(f"  [{p['score']:.4f}] {p['q']}  →  {why}")

    return {"pos_scores": pos_scores,
            "neg": [{"q": n["q"], "score": round(n["score"], 4), "kind": n["kind"]} for n in neg],
            "sweep": rows}


# ============================================================
# A2：学科隔离
# ============================================================
def exp_a2(e):
    hr("实验 A2 · FAQ 学科隔离（环节 ⑥）")
    print("问题：打开学科掩码后，跨学科误命中是否减少？命中的到底是不是「自己」？\n")

    scopes = [None, "common", "kaoyan", "cs", "employment"]
    result = {}
    print(f"{'查询学科':<12}{'命中自己':<10}{'命中他题':<10}{'未命中':<10}{'跨学科误命中率':<16}")
    print("-" * 78)
    for sc in scopes:
        self_hit = other_hit = miss = 0
        others = []
        for i, q in enumerate(e.questions):
            idx, score, masked = e.top(q, subject=sc)
            if idx < 0 or score < CALIBRATED:
                miss += 1
                continue
            if idx == i:
                self_hit += 1
            else:
                other_hit += 1
                others.append({
                    "query": q, "query_subject": e.subjects[i],
                    "hit": e.questions[idx], "hit_subject": e.subjects[idx],
                    "score": round(score, 4),
                })
        total_hit = self_hit + other_hit
        rate = other_hit / total_hit if total_hit else 0
        label = sc if sc else "全库(None)"
        print(f"{label:<12}{self_hit:<10}{other_hit:<10}{miss:<10}{rate:<16.2%}")
        result[label] = {"self": self_hit, "other": other_hit, "miss": miss,
                         "cross_rate": round(rate, 4), "others": others}
    print("-" * 78)

    # 关键对照：关掉掩码 vs 打开 cs 掩码，同一个 query 的命中差异
    print("\n抽样对照：同一个查询在「全库」与「cs 学科」下的命中")
    print("-" * 78)
    samples = [q for q, s in zip(e.questions, e.subjects) if s == "cs"][:3]
    samples += [q for q, s in zip(e.questions, e.subjects) if s == "employment"][:2]
    for q in samples:
        i = e.questions.index(q)
        i0, s0, _ = e.top(q, subject=None)
        i1, s1, _ = e.top(q, subject="cs")
        print(f"Q [{e.subjects[i]}] {q}")
        print(f"   全库   → {e.questions[i0] if i0 >= 0 else '-'}  ({s0:.4f})")
        print(f"   限cs   → {e.questions[i1] if i1 >= 0 else '-'}  ({s1:.4f})"
              f"{'  ← 降级' if s1 < CALIBRATED else ''}")
    return result


# ============================================================
# A3：实体一致性校验
# ============================================================
def exp_a3(e):
    hr("实验 A3 · 实体一致性校验（环节 ⑥）")
    from mysql_qa.retrieval.bm25_search import BM25Search as B

    print("靶子：4 条「同模板不同实体」问句（词项几乎全重叠，只换校名或年份）\n")
    print(f"{'查询':<38}{'最高分':<10}{'最相似标准题':<30}{'关校验':<10}{'开校验':<10}")
    print("-" * 100)
    rows = []
    for q in ENTITY_CONFLICT:
        idx, score, _ = e.top(q)
        best = e.questions[idx] if idx >= 0 else "-"
        conflict, detail = B._entity_conflict(q, best)
        print(f"{q:<38}{score:<10.4f}{best:<30}{'命中':<10}{'降级' if conflict else '命中':<10}")
        rows.append({"query": q, "score": round(score, 4), "matched": best,
                     "off_hit": True, "on_hit": not conflict, "detail": detail})
    print("-" * 100)
    blocked = sum(1 for r in rows if not r["on_hit"])
    print(f"开启实体校验后，4 条冲突样本降级 {blocked}/4")

    print("\n代价：对 100 条正常正样本，实体校验会误杀多少？")
    killed = []
    for i, q in enumerate(e.questions):
        idx, score, _ = e.top(q)
        if idx < 0 or score < CALIBRATED:
            continue
        conflict, detail = B._entity_conflict(q, e.questions[idx])
        if conflict:
            killed.append({"query": q, "matched": e.questions[idx], "detail": detail})
    print(f"  正样本中命中且被实体校验拦下的：{len(killed)} 条")
    for k in killed[:10]:
        print(f"    {k['query']}  →  {k['detail']}")
    if not killed:
        print("    （无——说明校验只在双方都有同类实体且冲突时才触发，未伤及正常命中）")
    return {"conflict_rows": rows, "blocked": blocked, "pos_killed": killed}


def main():
    e = Engine()
    hr("环境")
    print(f"题库 {e.n} 条   学科分布 {dict(collections.Counter(e.subjects))}")
    print(f"标定阈值 {CALIBRATED}（来自 config.ini [retrieval] faq_threshold）")

    r1 = exp_a1(e)
    r2 = exp_a2(e)
    r3 = exp_a3(e)

    payload = {"calibrated_threshold": CALIBRATED,
               "questions": e.n,
               "subjects": dict(collections.Counter(e.subjects)),
               "a1": r1, "a2": r2, "a3": r3}
    path = os.path.join(OUT_DIR, "exp_a123.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入：{path}")


if __name__ == "__main__":
    main()
