"""考研就业智能问答系统 - FAQ 关键词匹配 + RAG 检索增强 双层集成"""
import asyncio

from base.logger import logger
from mysql_qa.retrieval.bm25_search import BM25Search
from rag_qa.core.rag_system import RAGSystem

from mysql_qa.db.mysql_client import MySQLClient


class IntegratedQASystem:
    def __init__(self):
        self.bm25 = BM25Search()
        self.rag = RAGSystem()
        self.mysql_client = MySQLClient()
        logger.info("集成问答系统初始化完成")

    async def query(self, q, source_filter=None, session_id=None, trace=None):
        """
        流式问答主入口（异步生成器）
        :param q: 用户问题
        :param source_filter: 学科过滤，同时作用于 FAQ 层与 RAG 层
        :param session_id: 会话 ID，用于持久化对话历史
        :param trace: 可选 dict；传入时回填本次请求走了哪一层（layer = faq / rag /
                      fallback）、FAQ 判定细节（faq_score / matched_question /
                      reject_reason）以及 RAG 层真实检索到的 contexts。
                      供 rag_qa/rag_assesment/eval_pipeline.py 采集使用；
                      生产链路（app.py）不传，行为与之前完全一致。
        :yield: (文本分片, 是否输出完成)
        """
        # FAQ 层：带上学科过滤，与 RAG 层 source_filter 口径一致
        # trace 透传下去，让 FAQ 层把置信度、命中题、拒绝原因写进来
        faq_ans, need_rag = self.bm25.search(q, subject=source_filter, trace=trace)
        if faq_ans:
            # 记录本次由 FAQ 层直接作答：没走检索，contexts 必然为空
            if trace is not None:
                trace["layer"] = "faq"
                trace["contexts"] = []
                trace["category"] = None
            # ========== 【会话记忆：将本轮对话存入 MySQL】 ==========
            if session_id:
                self.mysql_client.update_session_history(session_id, q, faq_ans)
            # 逐字符流式推送答案，模拟打字效果
            # 必须用 asyncio.sleep：app.py 的 WebSocket 处理函数是 async 的，
            # 同步 time.sleep 会阻塞事件循环——一条 30 字答案的全部打字时间里，
            # 其他所有 WebSocket 连接都会停止响应。
            for char in faq_ans:
                yield char, False
                await asyncio.sleep(0.01)
            yield "", True  # 空字符串+True 代表流式传输结束
            return
        if need_rag:
            logger.info("FAQ无匹配，切换RAG检索")
            # 交给 RAG 层：RAGSystem.generate_answer 会继续往同一个 trace 里回填
            # category（通用知识/专业咨询）与 contexts（真正拼进 prompt 的父块）
            if trace is not None:
                trace["layer"] = "rag"
            # 查询历史
            history = self.mysql_client.fetch_recent_history(session_id) if session_id else []
            rag_answer = ""
            # 调用RAG流式生成：RAG 内部是同步 LLM 流式调用（真实网络等待），
            # 这里保持同步消费，不属于人为阻塞
            for token in self.rag.generate_answer(q, source_filter, history, trace):
                rag_answer += token
                yield token, False
            # ========== 【会话记忆：将本轮对话存入 MySQL】 ==========
            if session_id:
                self.mysql_client.update_session_history(session_id, q, rag_answer)
            yield "", True
            return
        # FAQ与RAG均无可用结果，返回兜底文案
        if trace is not None:
            trace["layer"] = "fallback"
            trace["contexts"] = []
        no_ans = "未找到答案，请联系人工客服。"
        for char in no_ans:
            yield char, False
            await asyncio.sleep(0.02)
        yield "", True


if __name__ == "__main__":
    print("===== 考研就业集成问答 | exit退出 =====")
    qa = IntegratedQASystem()

    async def main():
        """命令行调试入口（同步生成器已改为异步生成器，必须用 async for 消费）"""
        while True:
            q = input("请输入问题：").strip()
            if q.lower() == "exit":
                logger.info("用户退出")
                break
            subj = input("学科过滤(回车跳过)：").strip() or None
            print("\n🚀 正在处理...\n")
            print("-" * 60)
            print(f"✅ 回答：")
            async for chunk, finished in qa.query(q, subj):
                print(chunk, end="", flush=True)
                if finished:
                    break
            print("-" * 60, "\n")

    asyncio.run(main())
