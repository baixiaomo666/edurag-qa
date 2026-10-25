"""MySQL知识库问答主入口"""
from mysql_qa.retrieval.bm25_search import BM25Search


class MySQLQASystem:
    def __init__(self):
        self.bm25 = BM25Search()

    def query(self, q, subject=None):
        # 阈值不再硬编码：默认走 config.FAQ_THRESHOLD（config.ini 的 [retrieval] faq_threshold）
        ans, need_llm = self.bm25.search(q, subject=subject)
        if need_llm:
            ans = "FAQ未找到答案"
        return ans


if __name__ == "__main__":
    print("===== 问答系统启动 | exit退出 =====")
    qa = MySQLQASystem()
    while True:
        q = input("请输入问题：").strip()
        if q.lower() == "exit":
            break
        print(f"✅ 答案：{qa.query(q)}\n")
