"""
RAG 智能问答主系统
功能：
1. 问题分类（通用知识 / 专业咨询）
2. 专业问题 → 向量库检索 + 重排序
3. 通用问题 → 直接回答
"""
from langchain.chat_models import init_chat_model
from base.config import config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.strategy_selector import StrategySelector


class RAGSystem:
    """RAG 问答系统核心类"""

    def __init__(self):
        logger.info("初始化 RAG 智能问答系统...")

        # 1. 初始化问题分类器（BERT 微调模型）
        self.classifier = QueryClassifier()
        logger.info("问题分类器加载完成")

        # 2. 初始化向量存储（Milvus + BGE-M3 + Reranker）
        self.vector_store = VectorStore()
        logger.info("向量检索引擎加载完成")

        # 3. 初始化大模型
        # 必须显式声明 provider / api_key / base_url：
        # config.LLM_MODEL="deepseek-v4-flash" 会被 langchain 推断成内置的 deepseek
        # provider，而它要求环境变量 DEEPSEEK_API_KEY，否则在构造函数里直接抛
        # ValidationError（"If using default api base, DEEPSEEK_API_KEY must be set"）。
        # 走 OpenAI 兼容接口（model_provider="openai" + DeepSeek 的 base_url）
        # 才是本项目实际可用的调用方式，与 ragas_evaluate.py:44-49 的写法一致。
        self.llm = init_chat_model(
            model=config.LLM_MODEL,
            model_provider="openai",
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0.1  # 低温度，保证答案专业、稳定
        )
        logger.info("大模型初始化完成")


    def generate_answer(self, query, source_filter=None, history=None, trace=None):
        """
        根据用户问题生成最终答案（核心入口）
        :param query: 用户问题
        :param source_filter: 学科过滤（可选）
        :param history: 最近若干轮对话历史
        :param trace: 可观测性出口（可选），传入一个 dict 会被回填：
                      category - 问题分类结果（'通用知识' / '专业咨询'）
                      contexts - 实际拼进 prompt 的上下文列表；空列表需结合
                                 category 区分「压根未检索」与「检索为空」
                      生产链路不传此参数，行为零改变。
        :return: 最终回答
        """
        if not query or len(query.strip()) == 0:
            return "请输入有效的问题~"

        # 获取历史记录, 拼接为字符串
        if history is None:
            history = []
        history_context = "\n".join([f"Q:{i['question']}\nA:{i['answer']}" for i in history])

        # ====================== 1. 问题分类 ======================
        query_category = self.classifier.predict_category(query.strip())
        logger.info(f"问题分类结果: [{query_category}] → {query}")

        # ====================== 2. 构建提示词 ======================
        if query_category == "通用知识":
            # 通用知识：无知识库，直接回答
            if trace is not None:
                trace["category"] = query_category
                trace["contexts"] = []   # 显式空：表示压根没走检索，区别于「检索为空」
            prompt = RAGPrompts.rag_prompt().format(
                context="通用知识",
                history=history_context,
                question=query,
                phone=config.CUSTOMER_SERVICE_PHONE
            )
        else:
            # 专业咨询：从 Milvus 混合检索 + 重排序
            logger.info("开始执行向量检索...")
            # documents = self.vector_store.hybrid_search_with_reranker(query=query,source_filter=source_filter)
            ss = StrategySelector(vector_store=self.vector_store)
            documents = ss.strategy_select(query=query, source_filter=source_filter)

            if trace is not None:
                # 此处记录的 documents 与下方拼进 {context} 的是同一批，
                # 保证评估指标对齐「真实送进 LLM 的上下文」
                trace["category"] = query_category
                trace["contexts"] = [doc.page_content for doc in documents] if documents else []

            if documents:
                # 拼接父块上下文（高质量长文本）
                context = "\n\n".join([doc.page_content for doc in documents])
                logger.info(f"检索到 {len(documents)} 条有效参考内容")
            else:
                logger.warning("向量库未检索到相关内容")
                context = "未找到相关专业知识"

            # 构建 RAG 提示词
            prompt = RAGPrompts.rag_prompt().format(
                context=context,
                history=history_context,
                question=query,
                phone=config.CUSTOMER_SERVICE_PHONE
            )

        logger.debug(f"最终发送大模型提示词: {prompt[:300]}...")

        # ====================== 3. 调用大模型生成答案 ======================
        try:
            # 流式返回
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"LLM 流式错误: {e}")
            yield f"抱歉，处理出错，请联系客服：{config.CUSTOMER_SERVICE_PHONE}"


if __name__ == '__main__':
    # 测试 RAG 主系统
    rag_system = RAGSystem()

    # 测试三种策略都会自动走
    #answer = rag_system.generate_answer(query="AI智能应用学科的课程目录是怎样的")# 直接检索
    #answer = rag_system.generate_answer(query="人工智能学科和AI智能应用学科区别是什么") # 子查询检索
    answer = rag_system.generate_answer(query="人工智能学科在教育领域的应用有哪些？")# 假设问题检索
    print(answer)