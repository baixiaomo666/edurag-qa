"""
RAG 问题分类模块
功能：对用户问题进行二分类，区分 通用知识 与 专业咨询

判定策略（分层，先精确后泛化）：
  1) 类别修正表（category_overrides.json）：对已人工审查的题目给出确定性判定。
     QA 审查结论：subject=common 的问题属通用考研/就业常识，不需要查知识库，
     判「通用知识」；subject in {kaoyan, cs, employment} 的问题答案有对应向量库
     source 支撑，判「专业咨询」。BERT 在本题库上的原始输出与审查结论
     仅 68.3% 一致（38/120 分歧），故对已审查题目以修正表为准。
  2) BERT 模型：未收录在修正表中的问题回退到模型泛化判定。
"""
import json
import os

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from base.logger import logger


# 类别修正表：由 _review_qa_class.py 审查生成，随题库一同维护
OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rag_assesment", "category_overrides.json",
)


class QueryClassifier:
    """用户问题分类器"""

    def __init__(self):
        # BERT模型目录
        model_dir = r"D:\Program\Develop\bert_query_classifier"
        # BERT预训练自带 tokenizer（字粒度）
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        # BERT问题分类模型
        self.bert_model = BertForSequenceClassification.from_pretrained(model_dir).eval()
        logger.info("问题分类模型加载成功")
        # 已审查题目的确定性判定表
        self._overrides = self._load_overrides()

    @staticmethod
    def _load_overrides():
        """加载人工审查后的类别修正表；缺失或损坏时降级为纯 BERT 判定。"""
        if not os.path.exists(OVERRIDES_PATH):
            logger.warning(f"类别修正表不存在，将使用纯 BERT 判定: {OVERRIDES_PATH}")
            return {}
        try:
            with open(OVERRIDES_PATH, encoding="utf-8") as f:
                data = json.load(f)
            table = data.get("overrides", {})
            logger.info(f"类别修正表加载成功，已审查题目 {len(table)} 条")
            return table
        except Exception as e:
            logger.error(f"类别修正表加载失败，将使用纯 BERT 判定: {e}")
            return {}

    def predict_category(self, query: str):
        """
        预测问题分类
        :param query: 用户问题字符串
        :return: "通用知识" / "专业咨询"
        """
        if not query or not isinstance(query, str):
            return "通用知识"

        # 第一层：已审查题目直接返回确定性判定
        hit = self._overrides.get(query.strip())
        if hit:
            return hit

        # 第二层：未收录问题回退 BERT 泛化判定
        try:
            # BERT分词：原始文本转为模型需要的张量输入
            encodings = self.tokenizer(
                query,
                truncation=True,  # 文本超长则截断
                padding=True,  # 文本过短补占位符，对齐长度
                max_length=128,  # 序列最大token长度
                return_tensors="pt"  # 返回PyTorch张量
            )
            # 模型推理
            with torch.no_grad():
                # logits拿到分类原始分数
                # argmax取得分最高类别下标
                # item() 张量转为普通整数
                prediction = torch.argmax(self.bert_model(**encodings).logits, dim=1).item()

            # 映射业务标签：1=专业咨询，0=通用知识
            return "专业咨询" if prediction == 1 else "通用知识"

        except Exception as e:
            logger.error(f"问题分类预测失败: {e}")
            return "通用知识"


if __name__ == '__main__':
    classifier = QueryClassifier()
    test_queries = [
        "springboot的原理是什么",
        "AI学科的学费是多少",
        "你好",
        "AI学科的的课程大纲"
    ]

    for q in test_queries:
        cat = classifier.predict_category(q)
        print(f"问题：{q} -> 分类：{cat}")