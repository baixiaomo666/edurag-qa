"""
Redis 缓存客户端
"""
import redis
import json
from base.config import config
from base.logger import logger


class RedisClient:

    def __init__(self):
        """
        初始化 Redis 连接
        从全局配置读取连接参数，自动解码响应为字符串，记录连接状态
        """
        self.logger = logger
        try:
            # 创建 Redis 连接客户端，自动解析返回字符串
            self.client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接失败: {e}")
            # 连接失败抛出异常，终止程序执行
            raise

    def set_data(self, key, value, ex=None):
        """
        通用数据存储：将数据序列化为 JSON 存入 Redis
        :param key: 缓存键名
        :param value: 待存储数据（支持字典/列表等可序列化对象）
        :param ex: 过期时间（秒），None 表示永不过期
        """
        try:
            # 中文不转义，保证存储可读性
            # ex 由调用方传入：题库快照与答案缓存的 TTL 统一在 config.ini 配置，
            # 不设过期会导致题库更新后旧快照/旧答案一直生效（历史上已因此踩坑）
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ex)
            self.logger.info(f"数据已存储: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        """
        通用数据获取：从 Redis 读取并反序列化 JSON 数据
        :param key: 缓存键名
        :return: 解析后的数据 / None（无数据）
        """
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 获取数据失败: {e}")
            return None

    def get_answer(self, query):
        """
        问答专用：根据问题查询缓存答案
        :param query: 用户问题
        :return: 缓存答案 / None（未命中）
        """
        try:
            answer = self.client.get(f"answer:{query}")
            return json.loads(answer) if answer else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 查询失败: {e}")
            return None

    def delete(self, *keys):
        """
        删除指定缓存键
        用途：题库更新后手动失效快照（answer:{问题} 类缓存会随 TTL 自然过期）
        :param keys: 一个或多个键名
        :return: 实际删除的键数量
        """
        try:
            return self.client.delete(*keys)
        except redis.RedisError as e:
            self.logger.error(f"Redis 删除失败: {e}")
            return 0


if __name__ == '__main__':
    redcli = RedisClient()
    query = "测试问题"
    answer = "测试答案"
    redcli.set_data(f"answer:{query}", answer, ex=3600)
    print(redcli.get_answer("测试问题"))
