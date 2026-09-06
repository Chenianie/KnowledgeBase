# \test\fastapi\async.py

import asyncio

from fastapi import FastAPI

app = FastAPI()

# ✅ 正确：异步函数可以用 await
@app.get("/fetch-data")
async def fetch_data():
    # 模拟耗时操作（如查数据库）
    await asyncio.sleep(1) # 让出CPU
    return {"data": "完成"}

# ❌ 错误：同步函数不能用 await
# @app.get("/fetch-data")
# def fetch_data():  # 少了 async
#     await asyncio.sleep(1)  # SyntaxError!
#     return {"data": "完成"}

# ⚠️ 勉强能用但不好：同步函数做耗时操作会阻塞
@app.get("/fetch-data-bad")
def fetch_data_bad():
    import time
    time.sleep(1)  # 阻塞整个线程
    return {"data": "完成"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uv run uvicorn test.import.fastapi.async:app --reload
