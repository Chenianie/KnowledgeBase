# test/fastapi/异步2.py

import asyncio
import time


async def download_file(name):
    print(f"开始下载：{name}")
    await asyncio.sleep(2)  # 模拟下载耗时 2 秒
    print(f"下载完成：{name}")

async def main():
    # 三个任务同时开始！
    await asyncio.gather(
        download_file("文件 1"),
        download_file("文件 2"),
        download_file("文件 3")
    )

time_begin = time.time()
asyncio.run(main())
time_end = time.time()

# 总耗时：约 2 秒（同时进行）
print(f"总耗时：{time_end - time_begin} 秒")