from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from processor.import_processor.base import setup_logging
from processor.import_processor.config import get_config

setup_logging()
_bge_m3_ef = None

def get_bge_m3_ef():
    global _bge_m3_ef
    config = get_config()
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=config.bge_m3_path,
        device=config.bge_m3_device,#使用 GPU
        use_fp16=config.bge_fp16#精度
    )
    return _bge_m3_ef

def generate_embeddings(texts):
    """
      为文本生成向量嵌入
      :param texts: 要生成嵌入的文本列表
      :return: 包含dense和sparse向量的字典
      """
    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)
    processed_sparse = []
    for i in range(len(texts)): #indice是索引，indptr是索引的指针，data是数据
        sparse_indices = embeddings["sparse"].indices[
                         embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_data = embeddings["sparse"].data[
                      embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)

    return {
        'dense':[emb.tolist() for emb in embeddings['dense']],
        'sparse':processed_sparse
    }

