# -*- coding: utf-8 -*-
"""QueryClassifier 二分类器行为探针（只读，不改任何系统状态）。

用途：剥离 category_overrides.json 后，观察纯 BERT 的判定分布、置信度与句式敏感性。
用法：python -X utf8 _probe_classifier.py
"""
import json
import os
import statistics
import sys
import tempfile
import time

# 必须在 import base.logger 之前改 LOG_FILE：base/logger.py 的默认参数在函数定义时求值
import base.config as conf
conf.config.LOG_FILE = os.path.join(tempfile.gettempdir(), "probe_classifier.log")

import torch  # noqa: E402
from rag_qa.core.query_classifier import QueryClassifier  # noqa: E402

QA_PATH = os.path.join("../rag_qa", "rag_assesment", "kaoyan_qa_100.json")
OVERRIDES = os.path.join("../rag_qa", "rag_assesment", "category_overrides.json")


def main():
    clf = QueryClassifier()
    cfg = clf.bert_model.config
    print("=" * 78)
    print("[0] 模型元信息")
    print(f"    num_labels      = {cfg.num_labels}")
    print(f"    id2label        = {getattr(cfg, 'id2label', None)}")
    print(f"    label2id        = {getattr(cfg, 'label2id', None)}")
    print(f"    problem_type    = {cfg.problem_type}")
    print(f"    参数量          = {sum(p.numel() for p in clf.bert_model.parameters()):,}")
    print(f"    vocab_size      = {cfg.vocab_size}  (tokenizer 实际 {clf.tokenizer.vocab_size})")
    print(f"    修正表条数      = {len(clf._overrides)}")

    # 剥离修正表 -> 观察纯 BERT
    clf._overrides = {}

    def scores(q):
        enc = clf.tokenizer(q, truncation=True, padding=True, max_length=128,
                            return_tensors="pt")
        with torch.no_grad():
            logits = clf.bert_model(**enc).logits[0]
        p = torch.softmax(logits, dim=-1).tolist()
        n_tok = int(enc["input_ids"].shape[1])
        return p[0], p[1], n_tok           # p_general, p_pro, token 数

    def predict(q):
        g, p, _ = scores(q)
        return "专业咨询" if p > g else "通用知识"

    # ---------- 1) 全量分布 + 速度 ----------
    data = json.load(open(QA_PATH, encoding="utf-8"))
    qs = [d["question"] for d in data]
    t0 = time.perf_counter()
    confs, labs, toks = [], [], []
    for q in qs:
        g, p, n = scores(q)
        confs.append(max(g, p))
        labs.append("专业咨询" if p > g else "通用知识")
        toks.append(n)
    dt = (time.perf_counter() - t0) * 1000
    n_pro = labs.count("专业咨询")
    print("=" * 78)
    print(f"[1] 纯 BERT 对评测集 {len(qs)} 条的判定分布")
    print(f"    专业咨询 {n_pro}  /  通用知识 {len(qs) - n_pro}")
    ov = json.load(open(OVERRIDES, encoding="utf-8")).get("overrides", {})
    agree = sum(1 for q, l in zip(qs, labs) if ov.get(q) == l)
    diff = sum(1 for q, l in zip(qs, labs) if ov.get(q) and ov[q] != l)
    print(f"    与人工修正表一致 {agree}/{len(qs)}   分歧 {diff}")
    print(f"    平均最大概率 {statistics.mean(confs):.4f}   "
          f"最低 {min(confs):.4f}   中位 {statistics.median(confs):.4f}")
    buckets = {"<0.60": 0, "0.60-0.80": 0, "0.80-0.95": 0, ">0.95": 0}
    for c in confs:
        buckets["<0.60"] += c < 0.60
        buckets["0.60-0.80"] += 0.60 <= c < 0.80
        buckets["0.80-0.95"] += 0.80 <= c < 0.95
        buckets[">0.95"] += c >= 0.95
    print(f"    最大概率分布: {buckets}")
    print(f"    平均 token 数 {statistics.mean(toks):.1f}（max_length=128）")
    print(f"    单条推理 {dt / len(qs):.2f} ms（CPU，共 {dt:.0f} ms）")

    # ---------- 2) 典型样本 ----------
    print("=" * 78)
    print("[2] 典型样本的两类概率（p_通用 / p_专业）")
    for q in ["你好", "在吗？", "什么是国家线？", "考研A区和B区是什么意思？",
              "408统考中数据结构占多少分？", "秋招一般是什么时间？",
              "青云大学2025年统考录取多少人？", "springboot的原理是什么",
              "算法的复杂度怎么算？", "AI学科的学费是多少"]:
        g, p, n = scores(q)
        tag = "专业咨询" if p > g else "通用知识"
        print(f"    {q:<26} {g:.4f} / {p:.4f}  -> {tag}   ({n} tok)")

    # ---------- 3) 句式敏感性 ----------
    print("=" * 78)
    print("[3] 句式敏感性：同一语义换问法（看判定是否翻转）")
    groups = [
        ["什么是国家线？", "国家线是什么？", "国家线", "请问国家线是什么意思？",
         "国家线一般在几月公布？", "国家线通常什么时候公布"],
        ["什么是秋招？", "秋招是什么？", "秋招一般是什么时间？", "秋招什么时候开始"],
        ["408统考中数据结构占多少分？", "数据结构在408里占多少分？",
         "什么是数据结构？", "408统考的科目分值怎么分布？"],
    ]
    for g in groups:
        print("    " + "-" * 70)
        for q in g:
            p_g, p_p, _ = scores(q)
            tag = "专业咨询" if p_p > p_g else "通用知识"
            print(f"      {q:<28} p_专业={p_p:.4f}  -> {tag}")

    # ---------- 4) 边界输入 ----------
    print("=" * 78)
    print("[4] 边界输入")
    for q in ["", " ", "a", "?", "你好呀朋友", "今天天气怎么样？", "帮帮我",
              "我想问一个关于考研的问题可以吗", "考研"]:
        if not q or not q.strip():
            print(f"    {repr(q):<22} -> 代码短路返回「通用知识」（不经过模型）")
            continue
        g, p, n = scores(q)
        tag = "专业咨询" if p > g else "通用知识"
        print(f"    {repr(q):<22} {g:.4f} / {p:.4f}  -> {tag}   ({n} tok)")

    # ---------- 5) 混淆矩阵与分歧明细 ----------
    print("=" * 78)
    print("[5] 人工目标 vs 纯 BERT 混淆矩阵 + 分歧明细")
    from collections import Counter
    subj = {}
    faq_path = os.path.join("../rag_qa", "data", "kaoyan_faq_100.json")
    if os.path.exists(faq_path):
        subj = {d["question"]: d["subject_name"]
                for d in json.load(open(faq_path, encoding="utf-8"))}
    cm, details = Counter(), []
    for q, l in zip(qs, labs):
        tgt = ov.get(q)
        if not tgt:
            continue
        cm[(tgt, l)] += 1
        if tgt != l:
            details.append((subj.get(q, "新增常识题"), tgt, l, q))
    for k, v in sorted(cm.items()):
        print(f"    目标{k[0]} -> 实测{k[1]}: {v}")
    print(f"    分歧明细 {len(details)} 条（subject / 目标 / 实测 / 问题）：")
    for s, tgt, l, q in details:
        print(f"      [{s:<8}] 目标{tgt} 实测{l}  {q}")

    # ---------- 6) 表外问题泛化（线上新提问走这条路径） ----------
    print("=" * 78)
    print("[6] 表外问题泛化（修正表未收录）")
    outside = [
        # 期望「专业咨询」：答案在知识库/向量库覆盖范围内
        ("复试的时候英语口语考什么？", "专业咨询"),
        ("调剂系统可以填几个志愿？", "专业咨询"),
        ("推免生还需要参加初试吗？", "专业咨询"),
        ("计算机专硕和学硕的学费差多少？", "专业咨询"),
        ("408和自命题专业课哪个更容易考上？", "专业咨询"),
        ("秋招和春招的岗位有什么区别？", "专业咨询"),
        ("国企和互联网公司的薪资差距大吗？", "专业咨询"),
        ("二战考研需要注意什么？", "专业咨询"),
        # 期望「通用知识」：与知识库无关的闲聊/常识
        ("今天股市涨了吗？", "通用知识"),
        ("推荐一部好看的电影", "通用知识"),
        ("帮我讲讲相对论", "通用知识"),
        ("你会下国际象棋吗？", "通用知识"),
        ("北京有哪些好吃的？", "通用知识"),
        ("我心情不好怎么办？", "通用知识"),
    ]
    n_hit = 0
    for q, exp in outside:
        g, p, _ = scores(q)
        pred = "专业咨询" if p > g else "通用知识"
        n_hit += pred == exp
        mark = "OK  " if pred == exp else "MISS"
        in_tbl = "表内" if q in ov else "表外"
        print(f"    {mark} [{in_tbl}] {q:<26} 期望{exp} 实测{pred}  p_专业={p:.4f}")
    print(f"    表外命中 {n_hit}/{len(outside)}")

    # ---------- 7) 「什么是 X」式的库内专有概念（最危险方向：该查不查） ----------
    print("=" * 78)
    print("[7] 「什么是 X」式的库内专有概念（期望专业咨询=该检索）")
    what_is = [
        "什么是调剂系统？", "什么是预调剂？", "什么是复试线？", "什么是破格复试？",
        "什么是保研夏令营？", "什么是拟录取？", "什么是调剂意向采集系统？",
        "什么是考研大纲？", "什么是校线？", "什么是院线？",
    ]
    n_hit = 0
    for q in what_is:
        g, p, _ = scores(q)
        pred = "专业咨询" if p > g else "通用知识"
        n_hit += pred == "专业咨询"
        mark = "OK  " if pred == "专业咨询" else "MISS"
        print(f"    {mark} {q:<24} 实测{pred}  p_专业={p:.4f}  "
              f"({'表内' if q in ov else '表外'})")
    print(f"    判为专业咨询 {n_hit}/{len(what_is)}")

    print("=" * 78)
    print("PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
