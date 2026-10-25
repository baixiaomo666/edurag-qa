# base/config.py
import configparser
import os

# 获取项目根目录
"""
integrated_qa_system      # 🎯项目根目录
└── base
    ├── __init__.py
    └── config.py        # 当前文件
├── config.ini
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file='config.ini'):
        # 拼接绝对路径
        self.config_path = os.path.join(BASE_DIR, config_file)

        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(self.config_path, encoding='utf-8')

        # MySQL 配置
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')  # MySQL 主机地址
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')  # MySQL 用户名
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='root')  # MySQL 密码
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')  # MySQL 数据库名
        self.MYSQL_QA_TABLE = self.config.get('mysql', 'qa_table', fallback='jpkb')  # FAQ 题库表名

        # Redis 配置
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')  # Redis 主机地址
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)  # Redis 端口
        # self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='123456')       # Redis 密码
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)  # Redis 数据库编号

        # 日志配置
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')  # 日志文件路径


        # ===================== Milvus 向量库配置 =====================
        self.MILVUS_URI = os.getenv("MILVUS_PORT", self.config.get("milvus", "uri", fallback="http://localhost:19530"))
        self.MILVUS_DATABASE_NAME = os.getenv("MILVUS_DATABASE_NAME", self.config.get("milvus", "database_name", fallback="itcast"))
        self.MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", self.config.get("milvus", "collection_name", fallback="edurag_0421"))

        # ===================== LLM 大模型配置 =====================
        self.LLM_MODEL = self.config.get("llm", "model", fallback="deepseek-v4-flash")
        self.LLM_API_KEY= os.getenv("DEEPSEEK_API_KEY", self.config.get("llm", "llm_api_key"))
        self.LLM_BASE_URL= os.getenv("DEEPSEEK_BASE_URL",self.config.get("llm", "llm_base_url", fallback="https://api.deepseek.com"))

        # ===================== 检索分块参数 =====================
        self.PARENT_CHUNK_SIZE = self.config.getint("retrieval", "parent_chunk_size", fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint("retrieval", "child_chunk_size", fallback=300)
        self.CHUNK_OVERLAP = self.config.getint("retrieval", "chunk_overlap", fallback=50)
        self.RETRIEVAL_K = self.config.getint("retrieval", "retrieval_k", fallback=3)
        self.CANDIDATE_M = self.config.getint("retrieval", "candidate_m", fallback=2)

        # ===================== FAQ 关键词层参数 =====================
        # 置信阈值：全库 softmax 的分母随题库规模变化，同一数值在不同题库下含义不同，
        # 必须可配置并随题库重新标定（见项目根 _calibrate_faq.py）
        self.FAQ_THRESHOLD = self.config.getfloat("retrieval", "faq_threshold", fallback=0.66)
        # 题库快照与单条答案的缓存 TTL（秒），避免题库更新后旧快照/旧答案长期生效
        self.FAQ_SNAPSHOT_TTL = self.config.getint("retrieval", "faq_snapshot_ttl", fallback=3600)
        self.ANSWER_CACHE_TTL = self.config.getint("retrieval", "answer_cache_ttl", fallback=3600)

        # ===================== 应用业务配置 =====================
        self.VALID_SOURCES = eval(self.config.get("app", "valid_sources", fallback='["ai","java","test","ops","bigdata"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get("app", "customer_service_phone", fallback="13800138001")

# 单例模式：全局只加载一次配置
config = Config()

if __name__ == '__main__':
    print("MySQL 主机:", config.MYSQL_HOST)
    print("Redis 端口:", config.REDIS_PORT)
    print("日志路径:", config.LOG_FILE)
