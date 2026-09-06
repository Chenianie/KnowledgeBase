# \test\fastapi\async_sync.py

from fastapi import FastAPI

app = FastAPI()

# ✅ 异步版本
@app.get("/simple-async")
async def simple_async():
    return {"message": "Hello"}

# ✅ 同步版本
@app.get("/simple-sync")
def simple_sync():
    return {"message": "Hello"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uv run uvicorn test.import.fastapi.async_sync:app --reload