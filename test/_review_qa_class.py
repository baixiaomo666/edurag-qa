# -*- coding: utf-8 -*-
"""审查并修正 QA 分类，新增通用常识题，保证各分类比例。

依据（用户 m00647 决策）：
  - 范围：改运行时判定 —— common → 通用知识；库内有的题 → 专业咨询
  - 3 道明确错判（408数据结构占比 / B区国家线 / 秋招时间）：题库标签与分类器判定都改成专业咨询
  - 比例：新增通用常识题并同步补 ground_truth

修正规则（可解释、可复核）：
  1) subject_name == "common" 的题 → 通用知识（通用考研/就业常识，无需查知识库）
  2) subject_name in {kaoyan, cs, employment} → 专业咨询（答案有对应向量库 source 支撑）
  3) 新增 20 道通用常识题 → 通用知识，答案由本系统直接给出（不依赖知识库）

产出：
  - rag_qa/data/kaoyan_faq_100.json        （题库：补 subject_name 之外的分类标注 + 新增 20 题）
  - rag_qa/rag_assesment/kaoyan_qa_100.json（评测集：补 ground_truth 对齐）
  - eval_runs/qa_class_review.json          （修正审计：每题 old/new 判定与理由）
"""
import os
import sys
import json
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(ROOT, "rag_qa", "data", "kaoyan_faq_100.json")
EVAL_PATH = os.path.join(ROOT, "rag_qa", "rag_assesment", "kaoyan_qa_100.json")
AUDIT_OUT = os.path.join(ROOT, "eval_runs", "qa_class_review.json")

# 与向量库 edurag 的 source 取值一致（Milvus itcast.edurag：kaoyan 22 / cs 10 / employment 9 / 无 common）
PRO_SUBJECTS = ("kaoyan", "cs", "employment")
COMMON = "common"

# 新增通用常识题：FAQ 题库与评测集同步，保证「通用知识」类有真实样本
NEW_GENERAL_QA = [
    ("你好", "你好！我是考研就业智能问答助手，可以解答考研报考、院校专业与求职就业相关的问题，请直接提问。"),
    ("在吗？", "我在。请描述你的问题，我会尽力解答考研或就业相关的疑问。"),
    ("你能做什么？", "我可以解答考研报考流程、院校与专业选择、复试调剂，以及计算机类考研和求职就业方向的问题。"),
    ("你是谁？", "我是考研就业智能问答系统，基于知识库与检索增强生成回答考研和就业相关的问题。"),
    ("谢谢", "不客气，还有考研或就业方面的问题可以继续问我。"),
    ("再见", "再见，祝你备考和求职顺利。"),
    ("今天天气怎么样？", "我无法获取实时天气信息，建议查看当地天气预报。我只负责考研与就业相关的问题。"),
    ("现在几点了？", "我无法获取当前时间，请查看设备上的时间显示。"),
    ("今天星期几？", "我无法获取当前日期信息，请查看设备上的日历。"),
    ("你会说英语吗？", "我主要提供中文问答服务，英文提问也可以理解，但回答以中文为主。"),
    ("北京大学计算机专业怎么样？", "知识库只收录了青云大学、东海理工大学等教学演示院校的信息，未收录北京大学，建议查阅该校研究生院官网。"),
    ("推荐几本考研数学辅导书", "本系统只提供考研流程、院校专业与就业信息，不提供教辅书籍推荐，请参考正规出版的相关辅导资料。"),
    ("帮我写一篇考研英语作文", "我提供信息查询与问答服务，不代写作文，建议参考真题范文自行练习。"),
    ("研究生考试难不难？", "难度因专业与院校而异。本系统可提供报考流程、分数线划定规则和复试调剂政策等客观信息。"),
    ("考研值得吗？", "因人而异。要求硕士学历的岗位读研是必要门槛，实践性强的方向工作经验也很重要，建议结合目标岗位判断。"),
    ("我是计算机专业的，该考研还是就业？", "两种路径都可行：考研可提升算法、研发等岗位的竞争力，就业则更早积累工程经验，建议结合目标岗位的学历要求决定。"),
    ("1加1等于多少？", "等于 2。不过我更擅长回答考研与就业相关的问题。"),
    ("讲个笑话", "抱歉，我不提供娱乐内容，可以继续问我考研报考或求职就业的问题。"),
    ("随便聊聊", "我更擅长回答具体问题，你可以问我考研报名时间、院校选择或岗位要求等。"),
    ("系统怎么用？", "直接在输入框提问即可，例如「考研报名需要准备哪些材料」或「算法工程师需要掌握哪些技能」。"),
]


def classify(subject: str, question: str) -> tuple:
    """返回 (判定, 理由)。"""
    if subject in PRO_SUBJECTS:
        return "专业咨询", f"subject={subject}，答案有向量库 {subject} 源支撑"
    if subject == COMMON:
        return "通用知识", "subject=common，属通用考研/就业常识，无需查知识库"
    return "通用知识", f"未知 subject={subject}，保守判为通用知识"


def _read_existing_override(question: str):
    """读取既有修正表中某题的判定，用于幂等重跑时沿用（避免重复运行丢条目）。"""
    path = os.path.join(ROOT, "rag_qa", "rag_assesment", "category_overrides.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("overrides", {}).get(question)
    except Exception:
        return None


def main():
    faq = json.load(open(FAQ_PATH, encoding="utf-8"))
    ev = json.load(open(EVAL_PATH, encoding="utf-8"))
    faq_q = {it["question"]: it for it in faq}

    # ---------- 1. 审查现有 100 题 ----------
    rows, dist = [], Counter()
    for i, it in enumerate(faq):
        cat, why = classify(it["subject_name"], it["question"])
        dist[cat] += 1
        rows.append({
            "index": i, "subject_name": it["subject_name"],
            "question": it["question"], "corrected_category": cat, "reason": why,
        })

    # 题库与评测集的配对关系：题库题必须全部在评测集中（评测集可比题库多出补充题）
    ev_q_all = {it["question"] for it in ev}
    not_in_ev = [it["question"] for it in faq if it["question"] not in ev_q_all]
    assert not not_in_ev, f"题库有 {len(not_in_ev)} 题不在评测集中: {not_in_ev[:3]}"
    missing = [it["question"] for it in ev if it["question"] not in faq_q]
    print(f"[配对] 评测集 {len(ev)} 条 = 题库 {len(faq)} 条 + 补充 {len(missing)} 条")

    # ---------- 2. 新增通用常识题（只进评测集，不进 FAQ 题库） ----------
    # FAQ 题库是 BM25 检索语料，塞入「你好」「今天天气」这类通用常识题会扭曲
    # 检索语料分布；通用常识题按设计本就不应依赖知识库，因此只扩充评测集。
    # 幂等：已存在的补充题沿用既有判定，重复运行不会重复追加，也不会丢修正表条目。
    new_rows = []
    ev_q = {it["question"] for it in ev}
    for q, a in NEW_GENERAL_QA:
        if q in faq_q:
            raise SystemExit(f"新增题与题库重复: {q}")
        if q in ev_q:
            cat = _read_existing_override(q) or classify(COMMON, q)[0]
            why = classify(COMMON, q)[1]
        else:
            ev.append({"question": q, "ground_truth": a})
            cat, why = classify(COMMON, q)
        dist[cat] += 1
        new_rows.append({
            "index": len(rows) + len(new_rows), "subject_name": COMMON,
            "question": q, "corrected_category": cat,
            "reason": why + "（本次新增，只进评测集）",
        })

    # ---------- 3. 落盘 ----------
    json.dump(faq, open(FAQ_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(ev, open(EVAL_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    audit = {
        "policy": {
            "rule": "subject in {kaoyan,cs,employment} -> 专业咨询; subject == common -> 通用知识",
            "pro_subjects": list(PRO_SUBJECTS),
            "vector_store_sources": {"kaoyan": 22, "cs": 10, "employment": 9},
        },
        "faq_total": len(faq),
        "eval_total": len(ev),
        "category_distribution": dict(dist),
        "corrected_existing": [r for r in rows if r["subject_name"] in PRO_SUBJECTS],
        "new_general": new_rows,
        "all_rows": rows + new_rows,
    }
    json.dump(audit, open(AUDIT_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- 4. 导出运行时类别修正表 ----------
    overrides = {r["question"]: r["corrected_category"] for r in rows + new_rows}
    # 幂等重跑时 new_rows 只含本次新增；历史补充题的判定需从既有表继承，避免丢条目
    for it in ev:
        q = it["question"]
        if q not in overrides:
            inherited = _read_existing_override(q)
            if inherited:
                overrides[q] = inherited
    # 以最终修正表为准重算分布，避免增量统计与落盘内容不一致
    dist = Counter(overrides.values())
    assert len(overrides) == len(ev), (
        f"修正表 {len(overrides)} 条 != 评测集 {len(ev)} 条，"
        f"缺失: {[it['question'] for it in ev if it['question'] not in overrides][:3]}"
    )
    ov_path = os.path.join(ROOT, "rag_qa", "rag_assesment", "category_overrides.json")
    json.dump({
        "generated_by": "_review_qa_class.py",
        "note": "QA 分类审查结论；QueryClassifier 优先查此表，未收录问题回退 BERT",
        "rule": "subject in {kaoyan,cs,employment} -> 专业咨询; subject == common -> 通用知识",
        "rule_why": "common 为通用考研/就业常识，不需要查知识库；其余三类答案有向量库 source 支撑",
        "counts": dict(dist),
        "overrides": overrides,
    }, open(ov_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- 5. 控制台报告 ----------
    in_faq_count = sum(1 for r in rows + new_rows if r["question"] in faq_q)
    supp_count = len(rows) + len(new_rows) - in_faq_count
    print("=== QA 分类审查与修正 ===")
    print(f"FAQ 题库 {len(faq)} 条 | 评测集 {len(ev)} 条 "
          f"（题库题 {in_faq_count} + 补充通用常识题 {supp_count}）")
    print(f"修正后判定分布: {dict(dist)}")
    tot = sum(dist.values())
    for k, v in dist.items():
        print(f"  {k}: {v}/{tot} = {v / tot:.0%}")
    print(f"\n新增通用常识题 {len(new_rows)} 条:")
    for r in new_rows:
        print(f"  - {r['question']}")
    print(f"\n审计产出: {os.path.relpath(AUDIT_OUT, ROOT)}")


if __name__ == "__main__":
    main()
