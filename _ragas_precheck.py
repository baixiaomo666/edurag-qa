# -*- coding: utf-8 -*-
"""RAGAS 预检：用 _exp_c2.py 已落盘的最小 payload 验证评分链路能否跑通。

只做一件事：确认 ragas.evaluate + init_chat_model(DeepSeek) + 本地 bge-m3
这个组合在当前环境真的能出分数。此前只验证到"payload 结构正确"，
评分环节从未执行过——若不预检，用户跑完 20 题才发现在评分处崩掉，
那 60 次生成的开销就白费了。
"""
import os
import sys

sys.path.insert(0, r"D:\Study\code6\intergrated_qa_system")
os.chdir(r"D:\Study\code6\intergrated_qa_system")

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
from base.config import config

PAYLOAD = r"eval_runs\_smoke_c2_v3.json"

import json
run = json.load(open(PAYLOAD, encoding="utf-8"))
records = run["records"]
print(f"读入 {PAYLOAD}: records={len(records)} faq={run['faq_hits']} skipped={run['skipped_no_context']}")
assert records, "没有 records，无法评分"

dataset = Dataset.from_dict({
    "question": [r["question"] for r in records],
    "answer": [r["answer"] for r in records],
    "contexts": [r["contexts"] for r in records],
    "ground_truth": [r["ground_truth"] for r in records],
})
print(f"dataset 列：{dataset.column_names}  行数 {len(dataset)}")

print(f"裁判 LLM: model={config.LLM_MODEL} base_url={config.LLM_BASE_URL} "
      f"key={'已配置' if config.LLM_API_KEY else '缺失'}")
llm = init_chat_model(
    model=config.LLM_MODEL,
    model_provider="openai",
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
)
emb = HuggingFaceEmbeddings(model_name=r"D:\Program\Develop\bge-m3")
print("embedding 加载完成")

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=emb,
)
df = result.to_pandas()
print("=" * 70)
print("RAGAS 明细：")
print(df.to_string())
means = df.mean(numeric_only=True)
print("-" * 70)
for n in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
    print(f"  {n:22s} {means.get(n)}")
print("RAGAS_PRECHECK_OK")
