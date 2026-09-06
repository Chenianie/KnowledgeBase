import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, \
    StreamingResponse, Response

from tool.logger import logger

app = FastAPI()


@app.get('/', summary='第一个测试')
async def read_root():
    return {"Hello": "World"}


@app.get('/items/{item_id}', summary='获取指定参数')
async def read_item(item_id: int, q: str | None = None):
    return {'item_name': item_id, 'q': q}


# 接收? skip=? & limit = ?
@app.get("/items", summary="分页")
async def read_item(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


# POST 请求接收 JSON 数据
@app.post('/items/', summary='类型检查')
async def create_item(item: Item):
    # item 已经是验证过的 Item 对象
    # 如果客户端传来的 price 是字符串 "abc"，FastAPI 会自动报错
    return {"name": item.name, "price": item.price, "is_offer": item.is_offer}


# 1、路由处理函数返回一个 Pydantic 模型实例，FastAPI 将自动将其转换为 JSON 格式，并作为响应发送给客户端：
@app.post("/items/return", summary="返回 Pydantic 模型实例")
async def create_item(item: Item):
    return item


@app.delete("/items/{item_id}", summary="抛出异常")
async def read_item(item_id: int):
    if item_id == 42:
        raise HTTPException(status_code=404, detail="Item 找不到")
    return {"item_id": item_id}


@app.get("/api/user")
async def get_user():
    # 等价于直接 return {"name": "张三", "age": 20}（FastAPI 自动转 JSONResponse）
    return JSONResponse(
        content={"name": "张三", "age": 20},
        status_code=200,  # 可选，默认 200
        headers={"X-Custom-Header": "custom-value"}  # 可选，自定义响应头
    )


@app.get("/download/excel")
async def download_excel():
    excel_path = "D:/test.xlsx"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=excel_path,
        filename="月度报表.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.get('/hello')
async def hello(name: str = '游客'):
    html_content = f"""
<html>
<body>
<h1>你好 {name}</h1>
</body>
</html>
"""
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/text")
async def get_text():
    return PlainTextResponse(content="这是纯文本响应", status_code=200)

@app.get("/old-path")
async def redirect_old_path():
    # 重定向到 /new-path，状态码 307 表示临时重定向
    return RedirectResponse(url="/new-path", status_code=307)

@app.get('/new-path')
async def new_path():
    return {"message": "这是新路径"}

async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流

@app.get("/stream")
async def stream_response():
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.get("/custom")
async def custom_response():
    # 返回二进制数据，指定自定义 MIME 类型
    return Response(
        content="<h1>纯文本</h1>",
        # media_type="text/text",
        media_type="text/html",
        status_code=200)

if __name__ == '__main__':
    logger.info("File Import Service 服务启动中...")
    uvicorn.run(
        app=app,
        host="127.0.0.1",
        port=8000
    )
