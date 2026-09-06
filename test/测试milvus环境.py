from processor.import_processor.config import get_config

config = get_config()
res = config.milvus_url
print(res)