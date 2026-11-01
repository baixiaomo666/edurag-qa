"""
RAG 向量存储核心模块
功能：
1. 初始化 BGEM3 混合嵌入模型（稠密+稀疏向量）
2. 自动创建 Milvus 混合向量集合（含索引）
3. 为后续文本向量化、数据入库、混合检索提供基础
支持：父子块结构、学科来源、时间戳元数据存储
"""
import hashlib
import os
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType

from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from base.config import Config
from base.logger import logger

# 加载全局配置
conf = Config()


# ---------------------------------------------------------------------------
# 消融实验开关（B 组实验用）
#
# 设计原则：所有默认值 = 改动前的现状。不设置任何环境变量时，
#          hybrid_search_with_reranker 的行为与加开关之前逐字节一致。
#
#   EXPERIMENT_RERANKER = 1 | 0         是否执行 bge-reranker 精排（默认 1，开）
#   EXPERIMENT_CTX      = parent|child  送去 LLM 的是父块全文还是命中子块（默认 parent）
#   DENSE_W             = 0.7           WeightedRanker 稠密权重
#   SPARSE_W            = 1.0           WeightedRanker 稀疏权重
#   EXPERIMENT_M        = 0             Top-M 候选数，0 = 用 config.CANDIDATE_M（默认 2）
#   EXPERIMENT_K        = 0             初次召回条数，0 = 用 config.RETRIEVAL_K（默认 3）
#
# 之所以放在函数里读取而不是模块级常量：同一个进程内可以通过修改环境变量
# 连续跑多组实验，不必为每一组重启解释器（重启一次要重新加载 BGE-M3 + reranker）。
# ---------------------------------------------------------------------------
def _env_bool(name, default):
    """读取布尔型开关。未设置时返回 default；'0/false/no/off/空串' 视为 False。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_num(name, default):
    """读取数值型开关。未设置或无法解析时返回 default。"""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"环境变量 {name}={raw!r} 无法解析为数值，改用默认值 {default}")
        return default


class VectorStore:
    """
    向量存储管理类
    封装 Milvus 连接、BGE-M3 模型初始化、集合自动创建、索引构建逻辑
    """

    def __init__(self):
        """
        构造方法：执行完整初始化流程
        1. 加载本地 BGE-M3 混合嵌入模型
        2. 建立 Milvus 连接，自动创建目标数据库并切换
        3. 检测集合，不存在则自动创建 Schema + 稠密/稀疏索引
        """
        # ---------------------- 1. 初始化 BGE-M3 混合嵌入模型 ----------------------
        # 本地BGE-M3模型存放路径
        bge_m3_model_path = r"D:\Program\Develop\bge-m3"

        # 实例化混合向量模型，同时支持稠密向量、稀疏向量输出
        self.embedding = BGEM3EmbeddingFunction(model_name_or_path=bge_m3_model_path)
        logger.info("BGE-M3 混合嵌入模型初始化完成")

        # ---------------------- 2. Milvus 连接 + 数据库管理 ----------------------
       # 创建Milvus客户端连接
        self.client = MilvusClient(
            uri=conf.MILVUS_URI
        )
        logger.info(f"Milvus 基础连接成功")

        target_db = conf.MILVUS_DATABASE_NAME
        db_list = self.client.list_databases()
        # 判断数据库是否存在，不存在自动新建
        if target_db not in db_list:
            self.client.create_database(target_db)
            logger.info(f"数据库 {target_db} 不存在，自动创建完成")

        # 切换到业务目标数据库
        self.client.using_database(target_db)
        logger.info(f"已切换至数据库: {target_db}")

        # ---------------------- 3. 自动创建向量集合与索引 ----------------------
        self.create_collection()
         # ---------------------- 4. 初始化重排序模型 ----------------------

        bge_reranker_path = r"D:\Program\Develop\bge-reranker-large"
        self.reranker = CrossEncoder(bge_reranker_path)
        logger.info("BGE-Reranker 重排序模型初始化完成")
    def create_collection(self):
        """
        创建Milvus向量集合
        逻辑：集合不存在才创建，避免重复初始化
        内容：定义数据表Schema、字段含义、稠密向量索引、稀疏向量索引
        """
        coll_name = conf.MILVUS_COLLECTION_NAME
        if not self.client.has_collection(coll_name):
            # 创建表结构：关闭自动主键，开启动态字段支持
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)

            # 主键：文本切块唯一标识（MD5生成）
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
                description="文本切块唯一ID"
            )
            # 子块文本：用于检索匹配的最小文本单元
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                description="文本子块内容"
            )
            # BGE-M3输出稠密向量：捕捉语义信息
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embedding.dim['dense'],
                description="BGE-M3稠密语义向量"
            )
            # BGE-M3输出稀疏向量：捕捉关键词词项特征
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                description="BGE-M3稀疏词向量"
            )
            # 父子分块关联：子块归属的父文档ID
            schema.add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=100,
                description="父文本块ID，实现父子块关联"
            )
            # 父块完整原文：RAG回答生成时读取完整上下文
            schema.add_field(
                field_name="parent_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                description="父文本完整原文"
            )
            # 学科标签：检索时用作条件过滤
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=50,
                description="学科/数据源名称，用于检索过滤"
            )
            # 入库时间戳：用于数据生命周期管理
            schema.add_field(
                field_name="timestamp",
                datatype=DataType.VARCHAR,
                max_length=50,
                description="文档入库时间戳"
            )

            # 索引配置容器，批量定义多个向量索引
            index_params = self.client.prepare_index_params()

            # ========== 稠密向量索引：IVF_FLAT 倒排索引 ==========
            # nlist：聚类分组数量，用于加速检索
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )

            # ========== 稀疏向量索引：稀疏倒排索引 ==========
            # drop_ratio_build：建索引时裁剪每条向量权重最低20%弱特征，减少内存占用
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 执行创建集合 + 自动构建索引
            self.client.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"向量集合创建成功: {coll_name}")
        else:
            logger.info(f"向量集合已存在，无需重复创建: {coll_name}")


    def add_documents(self, documents):
        """ 将文档向量化之后存储到Milvus """

        # 获取文档的文本内容
        texts = [doc.page_content for doc in documents]
        # 使用bge-m3嵌入生成文档的嵌入
        embeddings = self.embedding(texts)
        # 初始化空列表，用于存储插入的数据
        data = []
        # 遍历每个文档，带上索引i
        for i, doc in enumerate(documents):
            # 生成文档内容的MD5 哈希值，作为唯一ID
            text_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            # 初始化稀疏向量字典
            sparce_vector = {}
            sparse_emb = embeddings["sparse"][i]  # 稀疏向量
            for index, val in zip(sparse_emb.col, sparse_emb.data):
                sparce_vector[int(index)] = float(val)

            # 将文档保存到列表中
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings['dense'][i],  # 稠密向量
                "sparse_vector": sparce_vector,  # 稀疏向量
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown"),
            })
        if data:
            # 使用 upsert 操作插入数据，覆盖重复 ID
            self.client.upsert(collection_name=conf.MILVUS_COLLECTION_NAME, data=data)
            logger.info(f"已插入或更新 {len(data)} 个文档")

    def hybrid_search_with_reranker(self, query, k=None, source_filter=None):
        """
        混合检索（稠密+稀疏）+ 加权融合 + BGE重排序 + 父块去重
        :param query: 用户问题
        :param k: 初步召回数量；None 时取 config.RETRIEVAL_K（可被 EXPERIMENT_K 覆盖）
        :param source_filter: 学科过滤
        :return: 重排后的候选列表；EXPERIMENT_CTX=child 时返回命中子块，否则返回父块全文
        """
        # ---- 实验开关：不设环境变量时，下面每个值都等于改动前的硬编码值 ----
        use_reranker = _env_bool("EXPERIMENT_RERANKER", True)      # 原：无条件执行
        ctx_mode = os.environ.get("EXPERIMENT_CTX", "parent").strip().lower()  # 原：恒为父块
        dense_w = _env_num("DENSE_W", 0.7)                          # 原：WeightedRanker(0.7, 1.0)
        sparse_w = _env_num("SPARSE_W", 1.0)
        exp_m = int(_env_num("EXPERIMENT_M", 0)) or conf.CANDIDATE_M    # 原：conf.CANDIDATE_M
        exp_k = int(_env_num("EXPERIMENT_K", 0)) or conf.RETRIEVAL_K    # 原：conf.RETRIEVAL_K
        if k is None:
            k = exp_k

        # 1. 生成查询向量
        embedding_result = self.embedding([query])
        dense_q = embedding_result['dense'][0]

        # 稀疏向量格式化
        sparse_q = {}
        row = embedding_result['sparse'][[0]]
        for idx, val in zip(row.indices, row.data):
            sparse_q[idx] = val

        # 2. 过滤条件
        filter_expr = f"source == '{source_filter}'" if source_filter else ""

        # 3. 构造检索请求
        dense_req = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )

        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=k,
            expr=filter_expr
        )

        # WeightedRanker(稠密权重, 稀疏权重)
        ranker = WeightedRanker(dense_w, sparse_w)

        # 4. 执行混合检索
        results = self.client.hybrid_search(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # 5. 转为 Document
        chunks = [self.__doc_from_hit(hit["entity"]) for hit in results]

        # 6. 组装候选：
        #    父块模式 —— 从命中子块回溯去重出父块全文（送完整上下文）
        #    子块模式 —— 直接用命中子块本身（送检索单元，上下文更短更聚焦）
        if ctx_mode == "child":
            candidates = chunks
        else:
            candidates = self.__get_unique_parent_docs(chunks)

        # 7. 重排序
        if use_reranker and len(candidates) >= 2:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self.reranker.predict(pairs)
            # 分数和文档绑定  (0.9,"a文档"),(0.5,"b文档")
            sorted_docs = sorted(zip(scores, candidates), reverse=True)
            # 获取排序后的文档列表
            candidates = [doc for score, doc in sorted_docs]

        # 返回候选集
        return candidates[:exp_m]

    def __doc_from_hit(self, entity):
        """Milvus 结果转为 Document 对象"""
        return Document(
            page_content=entity.get("text"),
            metadata={
                "parent_id": entity.get("parent_id"),
                "parent_content": entity.get("parent_content"),
                "source": entity.get("source"),
                "timestamp": entity.get("timestamp")
            }
        )

    def __get_unique_parent_docs(self, chunks):
        """从子块中提取去重父块"""
        seen = set()
        unique_docs = []
        for chunk in chunks:
            content = chunk.metadata.get("parent_content")
            if content and content not in seen:
                seen.add(content)
                unique_docs.append(
                    Document(page_content=content, metadata=chunk.metadata)
                )
        return unique_docs


if __name__ == '__main__':
    # 单元测试：初始化向量存储，自动完成模型加载、集合创建、索引构建
    vs = VectorStore()
    #
    # # 加载拆分文档
    # from document_processor import process_documents
    # # 直接传入目录路径
    # dir_path = r'D:\Study\code6\intergrated_qa_system\rag_qa\data'
    # chunks = process_documents(dir_path)
    # # 向量化存储
    # vs.add_documents(chunks)
    # 查询
    docs = vs.hybrid_search_with_reranker(query="电子科技大学 085400 复试分数线")
    for doc in docs:
        print(doc)
        print("--------------")