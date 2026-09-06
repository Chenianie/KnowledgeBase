# test/fastapi/同步1.py
import time

def download_file(name):
    print(f"开始下载：{name}")
    time.sleep(2)  # 模拟下载耗时 2 秒
    print(f"下载完成：{name}")

# 逐个下载
time_begin = time.time()
download_file("文件 1")
download_file("文件 2")
download_file("文件 3")
time_end = time.time()

# 总耗时：6 秒（2+2+2）
print(f"总耗时：{time_end - time_begin} 秒")
