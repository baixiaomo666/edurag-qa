# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
import os

from langchain_core.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:

    # ====================== 基础 RAG 问答（有历史） ======================
    # C2 组实验开关 EXPERIMENT_PROMPT：
    #   v1（默认）= 原模板，明确允许"基于上下文和你的知识回答"
    #   v2        = 严格约束：只依据上下文，资料里没有就说不知道，禁止用先验知识补全
    #   v3        = 在 v2 基础上要求标注依据来源（考验 faithfulness 与 answer_relevancy 的权衡）
    @staticmethod
    def rag_prompt(version=None):
        v = (version or os.environ.get("EXPERIMENT_PROMPT", "v1")).strip().lower()
        if v == "v2":
            return RAGPrompts._rag_prompt_v2()
        if v == "v3":
            return RAGPrompts._rag_prompt_v3()
        return RAGPrompts._rag_prompt_v1()

    @staticmethod
    def _rag_prompt_v1():
        return PromptTemplate(
            template="""
        你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 基于提供的上下文（如果有）和你的知识回答问题。
           - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确的回答，避免无关信息。
           - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
        4. **最后检查**：
           - 若需要具体分数线、招生人数，调用工具查询。
        **对话历史**:
        {history}

        **上下文**: 
        {context}

        **问题**: 
        {question}
        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def _rag_prompt_v2():
        return PromptTemplate(
            template="""
        你是一个严谨的知识库问答助手，**只能依据下面提供的资料回答**。

        **硬性规则**：
        1. 回答内容必须完全来自【资料】。【资料】中没有写到的信息，一律不要输出，即使你自己知道答案。
        2. 禁止推测、禁止补充资料之外的常识、禁止编造任何数字或名称。
        3. 如果【资料】里找不到答案，直接原样回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
        4. 【对话历史】只用来理解指代（如“它”“这个”），不作为事实来源。

        **对话历史**:
        {history}

        **资料**:
        {context}

        **问题**:
        {question}
        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    @staticmethod
    def _rag_prompt_v3():
        return PromptTemplate(
            template="""
        你是一个严谨的知识库问答助手，**只能依据下面提供的资料回答，并标注结论出处**。

        **硬性规则**：
        1. 回答内容必须完全来自【资料】。【资料】中没有写到的信息，一律不要输出，即使你自己知道答案。
        2. 每条关键结论后面用「（依据：<资料原文片段>）」标注来源，原文片段不超过 20 个字。
        3. 禁止推测、禁止补充资料之外的常识、禁止编造任何数字或名称。
        4. 如果【资料】里找不到答案，直接原样回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
        5. 【对话历史】只用来理解指代，不作为事实来源。

        **对话历史**:
        {history}

        **资料**:
        {context}

        **问题**:
        {question}
        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )

    # ====================== 假设答案生成（HYDE） ======================
    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            input_variables=["query"],
        )

    # ====================== 复杂查询拆分子查询 ======================
    @staticmethod
    def subquery_prompt():
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            input_variables=["query"],
        )


    # 使用策略选择提示词模版
    @staticmethod
    def strategy_select_prompt():
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下三种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                * 示例：
                    * 查询：人工智能学科学费是多少？
                    * 策略：直接检索
                    * 查询：AI智能应用的课程大纲是什么？
                    * 策略：直接检索
            2.  **假设问题检索：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                * 示例：
                    * 查询：人工智能在教育领域的应用有哪些？
                    * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                * 示例：
                    * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                    * 策略：子查询检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )