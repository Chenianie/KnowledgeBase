import asyncio

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 定义字典:
# key:会话id session_id
# value:异步队列
task_queues = {}


# 4. 定义异步耗时任务：直接往队列丢数据
async def long_task(session_id: str, query: str):
    # 为当前会话创建专属异步队列
    queue = asyncio.Queue()
    task_queues[session_id] = queue

    # 根据问题生成结果 TODO

    # 按查询词生成5条结果，每秒1条丢进队列
    for i in range(5):
        msg = f"【{query}】的第{i + 1}段回答：xxx{i + 1}"
        await queue.put(msg)  # 数据入队
        await asyncio.sleep(1)

    # 关键：放入结束标记，告诉SSE停止推送
    await queue.put(None)

class QueryRequest(BaseModel):
    query: str
    session_id: str

# 6. 提交任务接口：post形式
@app.post("/submit_query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    # 把查询词和会话ID传给后台任务
    background_tasks.add_task(long_task, req.session_id, req.query)
    return {"message": "任务已启动", "session_id": req.session_id}


# 7. 从队列取数据
@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    async def event_generator():
        # 等待当前会话的队列创建（防止SSE比任务先启动）
        while session_id not in task_queues:
            await asyncio.sleep(0.1)
        queue = task_queues[session_id]

        # 核心：循环从队列取数据，有数据就推，收到结束标记就停
        while True:
            msg = await queue.get()  # 阻塞等待队列数据
            if msg is None:  # 收到结束标记，退出循环
                break
            yield f"data: {msg}\n\n"  # 推送数据

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)