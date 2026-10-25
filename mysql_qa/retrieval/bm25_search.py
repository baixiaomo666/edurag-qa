"""
BM25 关键词检索模块
功能：基于 BM25Okapi 实现FAQ关键词匹配检索
架构：Redis缓存 + MySQL持久数据源，冷热分层加速查询
能力：缓存快速命中、Softmax分数归一化、置信阈值过滤、按学科隔离检索
输出：匹配答案 + 是否需要走LLM/RAG标记
"""
import re

from rank_bm25 import BM25Okapi

from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.util.preprocess import process_text
from base.config import config
from base.logger import logger
import numpy as np


# 实体识别用正则而非 NER 模型：校名（"XX大学/学院"）与 4 位年份的形态极其固定，
# 正则零依赖、零额外推理开销；给 FAQ 这条"快通道"挂一个 NER 模型与它的定位相悖。
SCHOOL_PATTERN = re.compile(r"[\u4e00-\u9fa5]{2,6}?(?:大学|学院)")
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


class BM25Search:

    def __init__(self):
        """构造方法"""
        # 创建 MySQL、Redis客户端连接对象
        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient()
        # 声明BM25全局变量
        self.bm25 = None
        # 声明原始问题列表（用于查询问题答案）
        self.original_questions = None
        # 声明分词后问题列表（提供给BM25计算相似度）
        self.tokenized_questions = None
        # 声明每条问题所属学科（用于按学科隔离检索）
        self.subjects = None
        # 置信阈值从配置读取（题库规模变化后需重新标定，见 _calibrate_faq.py）
        self.threshold = config.FAQ_THRESHOLD
        # 初始化BM25检索模型
        self._load()

    def _load(self):
        """私有的初始化BM25检索模型"""
        # 1.读取redis中原始问题、分词问题、学科标签
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        subject_key = "qa_subjects"
        self.original_questions = self.redis_client.get_data(original_key)
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)
        self.subjects = self.redis_client.get_data(subject_key)
        # 2.如果缓存中没有，从mysql中查询，并分词处理
        if not self.original_questions or not self.tokenized_questions:
            # 查询mysql题库（含学科），ORDER BY id 保证顺序稳定
            rows = self.mysql_client.fetch_question_rows()
            # 提取一维问题列表
            self.original_questions = [row["question"] for row in rows]
            # 对原始问题进行分词处理
            self.tokenized_questions = [process_text(q) for q in self.original_questions]
            # 记录每条问题所属学科，供学科隔离检索使用
            self.subjects = [row["subject_name"] for row in rows]
            # 3.将原始问题、分词问题、学科标签存入redis中
            # 必须带 TTL：否则题库更新后旧快照会一直生效，而且没有任何报错
            self.redis_client.set_data(
                original_key, self.original_questions, ex=config.FAQ_SNAPSHOT_TTL
            )
            self.redis_client.set_data(
                tokenized_key, self.tokenized_questions, ex=config.FAQ_SNAPSHOT_TTL
            )
            self.redis_client.set_data(
                subject_key, self.subjects, ex=config.FAQ_SNAPSHOT_TTL
            )
        # 4.初始化 BM25 检索模型
        self.bm25 = BM25Okapi(self.tokenized_questions)
        logger.info(f"BM25 检索模型初始化成功，题库 {len(self.original_questions)} 条")

    def reload(self):
        """
        手动重建题库索引：先删掉 Redis 快照再重读 MySQL
        用途：题库（jpkb）变更后立即生效，不必等 TTL 到期
        """
        self.redis_client.delete("qa_original_questions", "qa_tokenized_questions", "qa_subjects")
        self._load()

    @staticmethod
    def _extract_entities(text):
        """抽取一段文本里的决定性实体：校名集合 + 年份集合"""
        text = text or ""
        return {
            "schools": set(SCHOOL_PATTERN.findall(text)),
            "years": set(YEAR_PATTERN.findall(text)),
        }

    @classmethod
    def _entity_conflict(cls, query, standard_question):
        """
        实体一致性校验：判断「用户问的实体」与「命中的标准题里的实体」是否矛盾。
        :return: (是否冲突:bool, 冲突说明:str)

        只在「双方都出现同一类实体、且集合无交集」时才判冲突。
        为什么不判"单方有实体"：用户没提学校、或题库那道题本身没提学校，
        都不构成张冠李戴，强行降级会把大量正常命中推给 RAG，得不偿失。
        而双方提了不同校名/年份时，返回标准题的答案必定是错的——这正是
        全库 softmax 饱和到 1.0 也无法识别的场景（实测 "青云大学2025年统考
        录取多少人" vs "东海理工大学2025年统考录取多少人" 置信度 1.0000）。
        """
        q = cls._extract_entities(query)
        s = cls._extract_entities(standard_question)
        if q["schools"] and s["schools"] and not (q["schools"] & s["schools"]):
            return True, f"校名不一致 {sorted(q['schools'])} vs {sorted(s['schools'])}"
        if q["years"] and s["years"] and not (q["years"] & s["years"]):
            return True, f"年份不一致 {sorted(q['years'])} vs {sorted(s['years'])}"
        return False, ""

    @staticmethod
    def _mark(trace, **fields):
        """把判定细节回填进可选的 trace；不传就直接跳过，生产链路零开销"""
        if trace is not None:
            trace.update(fields)

    def search(self, query, threshold=None, subject=None, trace=None):
        """
        BM25检索查询的方法
        :param query:  用户问题
        :param threshold:  归一化后的阈值；None 时使用 config.FAQ_THRESHOLD
        :param subject:  学科过滤（如 "cs"）。None 表示全库检索；
                         传入时仅在该学科题目 + common（通用题）中检索，
                         与 RAG 层 source_filter 的口径保持一致
        :param trace:  可选 dict；传入时回填本次判定细节（faq_score /
                       matched_question / reject_reason），供离线评估做归因。
                       不传时行为与原来完全一致。
        :return: (answer:str,need_llm:bool)
        """
        # 阈值优先取调用方传入值，否则用配置值（不再硬编码 0.85）
        threshold = self.threshold if threshold is None else threshold
        # 1.redis问题的精准匹配查询
        cache_answer = self.redis_client.get_answer(query)
        if cache_answer:
            self._mark(trace, faq_score=None, matched_question=query,
                       reject_reason=None, from_cache=True)
            return cache_answer, False
        # 2.用户问题进行分词，BM25进行检索
        query_tokens = process_text(query)
        raw_scores = np.array(self.bm25.get_scores(query_tokens), dtype=float)
        # 3.零词项重叠短路：BM25 分数全为 0（或负）说明查询与题库没有共同词项。
        #    此时 np.argmax 恒定返回下标 0（题库第一条），原实现靠"softmax 之后概率必然很小"
        #    侥幸降级——这是隐式行为，阈值一旦调低就会真的误命中题库第 1 条。
        #    显式短路让行为变得确定。
        if raw_scores.size == 0 or raw_scores.max() <= 0:
            logger.info("FAQ无匹配（查询与题库无共同词项），切换RAG检索")
            self._mark(trace, faq_score=0.0, reject_reason="no_term_overlap")
            return None, True
        # 4.按学科隔离检索：非目标学科且非 common 的题目分数置为 -inf，
        #    经 softmax 后权重为 0，不参与竞争
        if subject:
            allowed = np.array([
                (s == subject) or (s == "common") for s in (self.subjects or [])
            ])
            # 长度不一致或该学科无题时不做掩码，退化为全库检索
            if allowed.size == raw_scores.size and allowed.any():
                raw_scores = np.where(allowed, raw_scores, -np.inf)
        # 5.归一化处理的分数
        softmax_scores = self._softmax(raw_scores)
        # 6.查询最高分，索引
        best_score = np.nanmax(softmax_scores)
        best_idx = np.argmax(softmax_scores)
        # 7.最高分对比阈值
        if best_score < threshold:
            logger.info(f"FAQ匹配失败，置信阈值{best_score:.3f}")
            self._mark(trace, faq_score=float(best_score), reject_reason="below_threshold")
            # 分数<阈值，无匹配答案，启动LLM
            return None, True

        # 8.实体一致性校验：阈值只回答"像不像"，不回答"是不是同一个对象"。
        #    词项几乎全重叠、只换实体名时（青云大学 vs 东海理工大学）置信度会
        #    饱和到 1.0，任何阈值都拦不住，必须再做一次实体比对。
        original_question = self.original_questions[best_idx]
        conflict, detail = self._entity_conflict(query, original_question)
        if conflict:
            logger.info(f"FAQ实体校验不通过（{detail}），降级RAG检索：{query}")
            self._mark(trace, faq_score=float(best_score),
                       matched_question=original_question,
                       reject_reason=f"entity_conflict: {detail}")
            return None, True

        # 9.分数>阈值且实体一致，取答案：先查缓存，再回源 MySQL
        cache_answer = self.redis_client.get_answer(original_question)
        if cache_answer:
            logger.info(f"Redis缓存命中问题：{original_question}")
            answer = cache_answer
        else:
            # 缓存没有查询mysql
            answer = self.mysql_client.fetch_answer(original_question)
            logger.info(f"MySQL命中问题：{original_question}")
            # 同步到缓存（带 TTL，避免题库答案更新后长期不刷新）
            self.redis_client.set_data(
                f"answer:{original_question}", answer, ex=config.ANSWER_CACHE_TTL
            )
        logger.info(f"FAQ匹配成功，置信阈值{best_score:.3f}")
        self._mark(trace, faq_score=float(best_score),
                   matched_question=original_question, reject_reason=None)
        return answer, False

    def _softmax(self, scores):
        """
        私有归一化函数：将BM25无界原始分值转为0~1概率
        :param scores: BM25输出原始相关性分数列表（可含 -inf，表示被学科过滤排除）
        :return: 归一化概率数组，数值越大匹配度越高
        """
        temperature = 1
        scores = np.array(scores) / temperature
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()


if __name__ == '__main__':
    bm25 = BM25Search()
    answer, need_llm = bm25.search("考研报考流程是什么")
    if answer:
        print(f"FAQ答案：{answer}")
    if need_llm:
        print("FAQ没有结果，需要执行LLM的RAG")
