"""
MySQL 数据库操作客户端
"""
import atexit
import threading

import pymysql

from base.config import config
from base.logger import logger


class MySQLClient:
    """
    MySQL 客户端（进程内单例）

    为什么做成单例：FAQ 链路上 app.py、faq_milvus.py、bm25_search.py 各自 new 一次，
    会建立 3 条互不复用的长连接且从不 close；单例让全进程共用一条连接，
    并在进程退出时自动关闭。

    并发说明：pymysql 的单连接 + 单游标不是线程安全的，
    因此所有 SQL 通过 self._db_lock 串行执行。
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # 双检锁：保证多线程下也只创建一个实例
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化数据库连接（单例：重复实例化时直接复用已有连接）"""
        # 单例守卫：不是第一次实例化就直接返回，不重建连接
        if getattr(self, "_initialized", False):
            return
        # SQL 串行锁：pymysql 单连接非线程安全
        self._db_lock = threading.RLock()
        # 题库表名走配置，避免硬编码
        self.qa_table = config.MYSQL_QA_TABLE
        try:
            # 创建MySQL连接，utf8mb4支持emoji、全部中文汉字
            self.connection = pymysql.connect(
                host=config.MYSQL_HOST,
                user=config.MYSQL_USER,
                password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DATABASE,
                charset='utf8mb4'
            )
            # 创建游标对象，用于执行SQL语句
            self.cursor = self.connection.cursor()
            logger.info("MySQL 连接成功")
        except pymysql.MySQLError as e:
            logger.error(f"MySQL 连接失败: {e}")
            # 连接失败主动抛出异常，上层必须捕获，不能继续运行
            raise
        self._initialized = True
        # 进程退出时自动释放连接，不再依赖调用方手动 close()
        atexit.register(self.close)

    def fetch_question_rows(self):
        """
        查询题库全部问题及其所属学科（FAQ 检索与学科隔离都需要）
        :return: list[dict]，元素形如 {"question": str, "subject_name": str}
        """
        try:
            with self._db_lock:
                # ORDER BY id 保证题库顺序稳定：BM25 命中后用下标回查问题文本，
                # 顺序不稳会导致同分时命中结果不可复现
                self.cursor.execute(
                    f"SELECT question, subject_name FROM {self.qa_table} ORDER BY id"
                )
                # fetchall() 返回格式：[(问题1, 学科1), (问题2, 学科2), ...]
                res = self.cursor.fetchall()
            logger.info(f"获取到 {len(res)} 条问题")
            return [
                {"question": row[0], "subject_name": row[1] or "common"}
                for row in res
            ]
        except pymysql.MySQLError as e:
            logger.error(f"问题查询失败: {e}")
            # 异常返回空列表，防止上层for遍历直接崩溃
            return []

    def fetch_questions(self):
        """
        查询并返回数据库中所有问题列表
        :return: list[str] 一维问题字符串列表
        """
        return [row["question"] for row in self.fetch_question_rows()]

    def fetch_answer(self, question):
        """
        根据问题精确匹配，查询对应答案
        :param question: str 待检索的问题文本
        :return: str | None 查询到的答案；无数据返回None
        """
        try:
            with self._db_lock:
                # 使用 %s 占位符，防止SQL注入（不要手动拼接字符串！）
                self.cursor.execute(
                    f"SELECT answer FROM {self.qa_table} WHERE question = %s", (question,)
                )
                # fetchone() 只获取第一条匹配记录
                res = self.cursor.fetchone()
            # 三元表达式：查到数据取answer，没有则返回None
            return res[0] if res else None
        except pymysql.MySQLError as e:
            logger.error(f"答案查询失败: {e}")
            return None

    def close(self):
        """安全关闭游标和数据库连接，释放服务器资源"""
        try:
            with self._db_lock:
                if getattr(self, "cursor", None):
                    self.cursor.close()
                    self.cursor = None
                if getattr(self, "connection", None):
                    self.connection.close()
                    self.connection = None
            # 关闭后允许重新建立连接
            self._initialized = False
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            logger.error(f"连接关闭失败: {e}")

    # ==================================会话记忆相关=================================
    # 获取最近 5 轮对话历史
    def fetch_recent_history(self, session_id: str) -> list:
        """获取最近5轮对话历史"""
        try:
            with self._db_lock:
                self.cursor.execute("""
                    SELECT question, answer
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (session_id, 5))
                rows = self.cursor.fetchall()
            history = [{"question": row[0], "answer": row[1]} for row in rows]
            return history[::-1]  # 反转结果，按时间正序返回
        except pymysql.MySQLError as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    # 新增对话并保留最近 5 轮
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            with self._db_lock:
                # 1. 插入新对话
                self.cursor.execute("""
                    INSERT INTO conversations (session_id, question, answer, timestamp)
                    VALUES (%s, %s, %s, NOW())
                """, (session_id, question, answer))

                # 2. 删除超出5轮的旧记录（子查询过滤）
                self.cursor.execute("""
                    DELETE FROM conversations
                    WHERE session_id = %s AND id NOT IN (
                        SELECT id FROM (
                            SELECT id
                            FROM conversations
                            WHERE session_id = %s
                            ORDER BY timestamp DESC
                            LIMIT %s
                        ) AS sub
                    )
                """, (session_id, session_id, 5))

                self.connection.commit()
            return self.fetch_recent_history(session_id)
        except pymysql.MySQLError as e:
            logger.error(f"更新会话历史失败: {e}")
            self.connection.rollback()
            raise

    # 清空指定会话的所有记录
    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        try:
            with self._db_lock:
                self.cursor.execute("""
                    DELETE FROM conversations
                    WHERE session_id = %s
                """, (session_id,))
                self.connection.commit()
            logger.info(f"会话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            logger.error(f"清除会话历史失败: {e}")
            self.connection.rollback()
            return False

    # 清空所有会话的所有记录
    def clear_all_sessions_history(self) -> bool:
        """清除所有会话历史"""
        try:
            with self._db_lock:
                self.cursor.execute("""
                    DELETE FROM conversations
                """)
                self.connection.commit()
            logger.info("所有会话历史已清除")
            return True
        except pymysql.MySQLError as e:
            logger.error(f"清除所有会话历史失败: {e}")
            self.connection.rollback()
            return False

    # 获取所有会话列表（按最新时间排序）
    def fetch_all_sessions(self) -> list:
        """获取所有会话列表，返回 session_id 和对话轮数"""
        try:
            with self._db_lock:
                self.cursor.execute("""
                    SELECT session_id, COUNT(*) as count, MAX(timestamp) as last_time
                    FROM conversations
                    GROUP BY session_id
                    ORDER BY last_time DESC
                """)
                rows = self.cursor.fetchall()
            sessions = [{
                "session_id": row[0],
                "count": row[1],
                "last_time": row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else ""
            } for row in rows]
            return sessions
        except pymysql.MySQLError as e:
            logger.error(f"获取会话列表失败: {e}")
            return []

    # 获取指定会话的对话总数
    def get_session_count(self, session_id: str) -> int:
        """获取指定会话的对话总轮数"""
        try:
            with self._db_lock:
                self.cursor.execute("""
                    SELECT COUNT(*) 
                    FROM conversations 
                    WHERE session_id = %s
                """, (session_id,))
                total_count = self.cursor.fetchone()[0]
            return total_count
        except pymysql.MySQLError as e:
            logger.error(f"获取会话总数失败: {e}")
            return 0

    def query_score_line(self, school, major, year):
        """查询指定学校、专业、年份的复试线和招生人数。
        major 可传专业代码（如 085400）或专业名称（如 电子信息），支持模糊匹配。"""
        sql = """SELECT school, major, year, score, enroll FROM score_line
                 WHERE school LIKE %s AND major LIKE %s AND year = %s
                 ORDER BY major LIMIT 5"""
        try:
            with self._db_lock:
                self.cursor.execute(sql, (f"%{school}%", f"%{major}%", year))
                rows = self.cursor.fetchall()
        except pymysql.MySQLError as e:
            logger.error(f"查询复试线失败: {e}")
            return None

        if not rows:
            return None

        return "\n".join(
            f"{s} {m} {y}年复试线{sc}分，统考招生{en}人"
            for s, m, y, sc, en in rows
        )

if __name__ == '__main__':
    # 实例化数据库客户端
    client = MySQLClient()

    # 循环读取每条问题，查询并打印对应答案
    for single_question in client.fetch_questions():
        print(f"问题：{single_question}")
        print(f"答案：{client.fetch_answer(single_question)}")
        print("-------")

    # 程序结束，释放数据库连接
    client.close()
