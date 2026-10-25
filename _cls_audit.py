# -*- coding: utf-8 -*-
"""分类器分流审计：量化「专业问题被误判为通用知识 → 直接空作答、不走检索」的规模。

背景：C1/C2 的 20 条采样题里有 3 条 n_ctx=0，报告初稿归因为「分块/库覆盖」。
实测反证：这 3 条直连 VectorStore 各召回 2 条，且答案就在库里；
它们在完整链路里根本没进检索——BERT 分类器把它们判成「通用知识」，
rag_system.py:74 的通用知识分支直接 contexts=[] 回答。

本脚本不改任何系统代码，只调 predict_category 做只读统计。
产出：eval_runs/cls_audit.json
"""
import os
import sys
import json
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.config import config as conf                       # noqa: E402

# logs/app.log 对子进程只读，必须在 import logger 前改路径
conf.LOG_FILE = os.path.join(os.environ.get("TEMP", "."), "cls_audit.log")

from base.logger import logger                               # noqa: E402
from rag_qa.core.rag_system import RAGSystem                 # noqa: E402

QUESTIONS = os.path.join("rag_qa", "rag_assesment", "kaoyan_qa_100.json")
OUT = os.path.join("eval_runs", "cls_audit.json")


def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        items = json.load(f)
    logger.info(f"分类器分流审计开始：{len(items)} 条")

    rag = RAGSystem()
    rows = []
    for i, it in enumerate(items):
        q = it["question"].strip()
        cat = rag.classifier.predict_category(q)
        rows.append({"index": i, "question": q, "category": cat})
        if (i + 1) % 20 == 0:
            logger.info(f"  {i + 1}/{len(items)}")

    cnt = Counter(r["category"] for r in rows)
    general = [r for r in rows if r["category"] == "通用知识"]
    n = len(rows)
    summary = {
        "total": n,
        "专业咨询": cnt.get("专业咨询", 0),
        "通用知识": cnt.get("通用知识", 0),
        # 「通用知识」在完整链路里 contexts=[] 且不检索，等价于"未走知识库"
        "general_share": round(cnt.get("通用知识", 0) / n, 4) if n else 0.0,
        "general_questions": [r["question"] for r in general],
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== 分类器分流审计 ===")
    print(f"总题数            : {n}")
    print(f"专业咨询（走检索）: {summary['专业咨询']} ({1 - summary['general_share']:.1%})")
    print(f"通用知识（不检索）: {summary['通用知识']} ({summary['general_share']:.1%})")
    print(f"产出              : {OUT}")
    print("\n被判为「通用知识」的题目：")
    for q in summary["general_questions"]:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
