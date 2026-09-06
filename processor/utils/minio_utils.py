import json
import logging

from minio import Minio

from processor.import_processor.config import get_config


try:
    config = get_config()
    client = Minio(
        endpoint=config.minio_endpoint,
        access_key=config.minio_access_key,
        secret_key=config.minio_secret_key,
        secure=config.minio_secure
    )
    # secure=False 是否启用HTTPS加密连接；False=用HTTP，True=用HTTPS；本地/内网部署一律写False
    if not client.bucket_exists(config.minio_bucket_name):
        logging.getLogger().info(f"存储桶{config.minio_bucket_name}不存在，开始创建...")
        client.make_bucket(config.minio_bucket_name)
        logging.getLogger().info("创建存储桶成功!")

    # 设置存储桶策略为 Public Read (只读权限开放给匿名用户)
    # 这样前端可以直接通过 URL 访问图片，而不需要预签名 URL
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{config.minio_bucket_name}/*",
            },
        ],
    }

    client.set_bucket_policy(config.minio_bucket_name, json.dumps(policy))
except Exception as e:
    print(f'Minio init failed:{e}')
    client = None

def get_minio_client():
    return client