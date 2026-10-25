import os
from datetime import datetime
from langchain_community.document_loaders import TextLoader

# 自定义文本分割器
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
# 自定义OCR加载器（支持PDF/Word/PPT/图片）
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader

# 项目基础配置与日志
from base.config import config as conf
from base.logger import logger


# ====================== 文件加载器映射表 ======================
# 支持的文件类型与对应加载器，统一管理便于扩展
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".jpg": OCRIMGLoader,
    ".png": OCRIMGLoader,
    ".md": TextLoader
}


def load_documents_from_directory(directory_path):
    """
    从指定目录递归加载所有支持的文档文件
    自动识别文件类型、添加元数据、异常捕获、日志记录
    :param directory_path: 文档根目录路径（如 data，包含 ai_data、java_data 等子目录）
    :return: LangChain Document 列表
    """
    # 存储所有加载完成的文档
    documents = []
    # 获取支持的文件后缀集合
    supported_extensions = document_loaders.keys()

    # 遍历子目录（如 ai_data、java_data）
    for subject_dir in os.listdir(directory_path):
        subject_path = os.path.join(directory_path, subject_dir)

        # 只处理目录
        if not os.path.isdir(subject_path):
            continue

        # 从目录名提取学科来源（如 ai_data → ai，java_data → java）
        source_subject = subject_dir.replace("_data", "")
        logger.info(f"开始处理学科: {source_subject}")

        # 递归遍历学科目录下的所有文件
        for dirpath, dirnames, filenames in os.walk(subject_path):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                # 文件类型支持 → 执行加载
                if file_ext in supported_extensions:
                    try:
                        # 获取对应文件加载器
                        loader_cls = document_loaders[file_ext]

                        # TXT / MD 需指定 UTF-8 编码
                        if file_ext in (".txt", ".md"):
                            loader = loader_cls(file_path, encoding="utf-8")
                        else:
                            loader = loader_cls(file_path)

                        # 加载文档内容
                        loaded_docs = loader.load()

                        # 为每个文档统一添加元数据（用于后续检索溯源）
                        for doc in loaded_docs:
                            doc.metadata["source"] = source_subject  # 学科来源
                            doc.metadata["file_path"] = file_path  # 文件路径
                            doc.metadata["timestamp"] = datetime.now().isoformat()  # 入库时间

                        # 合并到总文档列表
                        documents.extend(loaded_docs)
                        logger.info(f"成功加载文件: {file_path}")
                    except Exception as e:
                        logger.error(f"加载文件失败: {file_path} | 错误信息: {str(e)}")
                # 不支持的文件类型
                else:
                    logger.warning(f"跳过不支持的文件类型: {file_path}")

    logger.info(f"目录加载完成，总计加载文档片段: {len(documents)}")
    return documents


def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                      child_chunk_size=conf.CHILD_CHUNK_SIZE,
                      chunk_overlap=conf.CHUNK_OVERLAP):
    """
    文档分层分割核心函数（父子块结构）
    流程：加载文档 → 父块分割 → 子块分割 → 元数据绑定
    :param directory_path: 文档目录
    :param parent_chunk_size: 父块大小
    :param child_chunk_size: 子块大小
    :param chunk_overlap: 切块重叠长度
    :return: 带完整元数据的子块列表（用于向量入库）
    """
    # 1. 加载目录下所有文档
    documents = load_documents_from_directory(directory_path)

    # 2. 初始化通用文本 / Markdown 专用分割器
    # 普通文本分割器（中文分层分割）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # Markdown 专用分割器（按标题层级分割）
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 存储最终分割完成的子块
    child_chunks = []

    # 3. 遍历所有文档，执行分层分割
    for i, doc in enumerate(documents):
        file_path = doc.metadata.get("file_path", "")
        file_ext = os.path.splitext(file_path)[1].lower()

        # 根据文件类型选择分割器
        is_markdown = (file_ext == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

        logger.info(f"处理文档: {file_path}, 分割器类型: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        # 4. 分割父块
        parent_docs = parent_splitter_to_use.split_documents([doc])

        # 5. 遍历父块，分割为子块并绑定元数据
        for j, parent_doc in enumerate(parent_docs):
            # 生成父块唯一ID
            parent_id = f"doc_{i}_parent_{j}"
            # 6. 父块分割为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            # 7. 子块元数据注入（关联父块）
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)

    logger.info(f"分层分割完成，最终子块总数: {len(child_chunks)}")
    return child_chunks


# ====================== 测试入口 ======================
if __name__ == '__main__':
    # 执行文档处理（加载+分割）
    chunks = process_documents(r'D:\Study\code6\intergrated_qa_system\rag_qa\data')
    # 输出切块结果
    print("最终生成文档切块数量：", len(chunks))