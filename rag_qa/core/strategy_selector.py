"""
RAG 检索策略选择核心模块
功能：
1. 动态选择检索策略：直接检索 / 子查询拆分 / 假设问题检索（HyDE）
2. 根据问题复杂度自动路由最优检索方式
3. 去重、排序、输出最终上下文
"""
import os

from base.logger import logger
from langchain.chat_models import init_chat_model

from base.config import config as conf
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.vector_store import VectorStore

# ---------------------- C 组实验开关（不设环境变量时行为与原实现完全一致） ----------------------
# EXPERIMENT_STRATEGY:
#   auto（默认）             → 让 LLM 自己选策略（原行为）
#   direct / hyde / subquery → 强制指定策略，跳过"选策略"那次 LLM 调用
_FORCE_STRATEGY = {
    "direct": "直接检索",
    "hyde": "假设问题检索",
    "subquery": "子查询检索",
}



class StrategySelector:
    """智能检索策略选择器"""

    def __init__(self,vector_store=None):
        # 1. 初始化大模型
        # 同 RAGSystem.__init__：必须显式传 provider / api_key / base_url，
        # 否则 model="deepseek-v4-flash" 会被推断成内置 deepseek provider，
        # 并在构造时因缺少 DEEPSEEK_API_KEY 抛 ValidationError。
        self.llm = init_chat_model(
            model=conf.LLM_MODEL,
            model_provider="openai",
            api_key=conf.LLM_API_KEY,
            base_url=conf.LLM_BASE_URL,
            temperature=0.3,
        )

        # 2. 初始化向量存储
        self.vector_store = vector_store or VectorStore()
        logger.info("检索策略引擎初始化完成")

    def strategy_select(self, query, source_filter=None):
        """
        核心入口：根据用户问题自动选择最优检索策略
        :param query: 用户原始问题
        :param source_filter: 学科过滤
        :return: 最优排序后的文档列表
        """
        if not query or not query.strip():
            logger.warning("输入问题为空，返回空上下文")
            return []

        # ====================== 1. 选择检索策略 ======================
        # 实验开关：EXPERIMENT_STRATEGY=direct/hyde/subquery 时强制指定，
        # 不设或设为 auto 时走原来的 LLM 自动选择。
        forced = os.environ.get("EXPERIMENT_STRATEGY", "auto").strip().lower()
        if forced in _FORCE_STRATEGY:
            strategy = _FORCE_STRATEGY[forced]
            logger.info(f"策略强制指定（EXPERIMENT_STRATEGY={forced}）：【{strategy}】 → {query}")
        else:
            strategy_prompt = RAGPrompts.strategy_select_prompt().format(query=query)
            strategy = self._call_llm(strategy_prompt).strip()
            logger.info(f"策略选择结果：【{strategy}】 → {query}")

        # 挂到实例上：原实现里 strategy 是局部变量，调用方无法得知每条问题实际走了哪个
        # 策略（C 组实验需要的正是这个）。不设开关时该值 = LLM 的选择结果，语义不变。
        self.last_strategy = strategy

        # ====================== 2. 执行对应检索策略 ======================
        ranked_docs = []
        if strategy == "子查询检索":
            ranked_docs = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == "假设问题检索":
            ranked_docs = self._retrieve_with_hyde(query, source_filter)
        else:
            ranked_docs = self.vector_store.hybrid_search_with_reranker(
                query=query,
                source_filter=source_filter
            )

        # ====================== 3. 截取最终候选集 ======================
        final_docs = ranked_docs[:conf.CANDIDATE_M]
        logger.info(f"策略执行完成，最终返回 {len(final_docs)} 条上下文")
        return final_docs

    def _retrieve_with_hyde(self, query, source_filter):
        """
        HyDE 假设答案检索策略
        先生成假设答案 → 用答案向量检索 → 语义更匹配
        """
        logger.info(f"执行 HyDE 假设答案检索：{query}")
        try:
            # 生成假设答案
            hyde_prompt = RAGPrompts.hyde_prompt().format(query=query)
            hypo_answer = self._call_llm(hyde_prompt).strip()
            logger.info(f"HyDE 生成答案：{hypo_answer[:100]}...")

            # 使用假设答案检索
            return self.vector_store.hybrid_search_with_reranker(
                query=hypo_answer,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败：{str(e)}")
            return []

    def _retrieve_with_subqueries(self, query, source_filter=None):
        """
        子查询拆分策略
        复杂问题 → 拆成 N 个小问题 → 分别检索 → 去重合并
        """
        logger.info(f"执行 子查询拆分检索：{query}")
        try:
            # 生成子问题
            subquery_prompt = RAGPrompts.subquery_prompt().format(query=query)
            subqueries_text = self._call_llm(subquery_prompt).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]

            if not subqueries:
                logger.warning("子查询生成为空，返回空列表")
                return []

            logger.info(f"拆分后子问题：{subqueries}")

            # 批量检索
            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_reranker(
                    query=sub_q,
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子问题 [{sub_q}] 召回 {len(docs)} 条")

            # 文本去重
            unique_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_dict.values())

            logger.info(f"合并去重后：{len(unique_docs)} 条")
            return unique_docs
        except Exception as e:
            logger.error(f"子查询策略失败：{str(e)}")
            return []

    def _call_llm(self, prompt):
        """
        统一 LLM 调用接口
        简化格式、稳定返回、异常兜底
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM 调用异常：{str(e)}")
            return "直接检索"


# 测试实例
if __name__ == '__main__':
    ss = StrategySelector()
    test_query = "AI智能应用学科的课程体系和就业方向是什么？"
    docs = ss.strategy_select(test_query)
    logger.info(f"最终获取上下文数量：{len(docs)}")