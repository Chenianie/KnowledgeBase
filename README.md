# KnowledgeBase

面向门店/售后场景的产品知识库：把说明书、安全手册等 PDF/Markdown 解析入库，再按「商品型号」约束做多路检索，生成可引用原文的回答。

项目用 **LangGraph** 编排导入与查询两条流水线，用 **FastAPI** 对外提供上传、问答和 SSE 流式接口。

## 能做什么

- **文档入库**：支持 `.pdf` / `.md`。PDF 经 MinerU 转 Markdown；文档内图片经视觉模型摘要后上传 MinIO，正文中的本地路径替换为可访问 URL。
- **商品名抽取**：导入时识别产品/型号，写入独立向量集合，查询时先确认主体再检索。
- **混合检索**：BGE-M3 稠密 + 稀疏向量写入 Milvus，查询侧做混合检索。
- **多路召回与融合**：普通向量检索、HyDE 假设性答案检索、DashScope MCP 联网搜索，经 RRF 融合后再用 `qwen3-rerank` 重排。
- **对话问答**：MongoDB 存会话历史；支持同步返回与 SSE 流式输出；答案约束为「只根据参考内容作答」，必要时附图片 URL。
- **离线评估**：`shopkeeper_brain/eval` 用 ragas 对问答集打分。

## 技术栈

| 层级 | 选型 |
| --- | --- |
| 编排 | LangGraph、LangChain |
| 服务 | FastAPI、Uvicorn |
| 向量库 | Milvus（集合 `kb_chunks`、`kb_item_names`） |
| 会话存储 | MongoDB |
| 对象存储 | MinIO |
| 嵌入 | 本地 BGE-M3（dense + sparse） |
| LLM / VL / 重排 | 阿里云 DashScope（OpenAI 兼容接口） |
| PDF 解析 | MinerU API |
| 联网搜索 | DashScope MCP WebSearch |
| 依赖管理 | uv，Python ≥ 3.11 |

默认 PyTorch 从 CUDA 12.6 源安装（见 `pyproject.toml`）。无 GPU 时需自行改安装源，并把 `BGE_DEVICE` 等改为 CPU。

## 目录结构

```
KnowledgeBase/
├── main.py                          # 占位入口
├── pyproject.toml / uv.lock
├── .env.example                     # 环境变量模板（复制为 .env）
├── processor/
│   ├── import_processor/            # 导入图：解析 → 切分 → 识别型号 → 向量化 → 入库
│   ├── query_processor/             # 查询图：确认型号 → 多路检索 → RRF → 重排 → 生成
│   └── utils/                       # Milvus / MinIO / Mongo / LLM / Embedding / SSE
├── web/
│   ├── api/import_service.py        # 导入 API（默认 8000）
│   ├── api/query_service.py         # 查询 API（默认 8001）
│   └── page/import.html, chat.html
├── shopkeeper_brain/
│   └── eval/                        # ragas 评估与示例 QA
├── tool/                            # 日志、BGE-M3 下载脚本
└── test/                            # FastAPI / GPU / Milvus 等试验代码
```

## 环境准备

1. **Python 3.11+** 与 [uv](https://docs.astral.sh/uv/)
2. 本机或局域网服务：
   - Milvus（默认 `http://localhost:19530`）
   - MongoDB（默认 `mongodb://localhost:27017`，库名 `kb001`）
   - MinIO（默认 `localhost:9000`，桶 `knowledge-base`）
3. 外部账号：
   - DashScope：`OPENAI_API_KEY`、MCP WebSearch、重排模型
   - MinerU：`MINERU_API_TOKEN`、`MINERU_BASE_URL`
4. 本地模型：BGE-M3（路径由 `BGE_M3_PATH` 指定）。可用：

```bash
uv run python tool/dowmload_bgem3.py
```

脚本内缓存目录需与 `.env` 中的 `MODELSCOPE_CACHE` / `BGE_M3_PATH` 一致。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
copy .env.example .env   # Windows
# cp .env.example .env   # Linux / macOS
```

按本机路径和密钥编辑 `.env`，至少确认：

- `OPENAI_API_KEY`、`OPENAI_API_BASE`
- `BGE_M3_PATH`、`BGE_DEVICE`
- `MILVUS_URL`、`MONGO_URL`、`MINIO_*`
- `MINERU_API_TOKEN`、`MINERU_BASE_URL`
- `DATA_BASED_ROOT_DIR`：导入文件本地落盘根目录

启动服务（两个进程）：

```bash
# 导入（页面 http://127.0.0.1:8000/import.html ，Swagger /docs）
uv run python web/api/import_service.py

# 查询（页面 http://127.0.0.1:8001/chat.html ，Swagger /docs）
uv run python web/api/query_service.py
```

也可在仓库根目录直接跑图（需改脚本里的本地文件路径）：

```bash
uv run python processor/import_processor/main_graph.py
uv run python processor/query_processor/main_graph.py
```

## 导入流水线

`KBImportWorkflow`（`processor/import_processor/main_graph.py`）：

```
node_entry
    ├─ PDF → node_pdf_to_md → node_md_img
    └─ MD  → node_md_img
         → node_document_split
         → node_item_name_recognition
         → node_bge_embedding
         → node_import_milvus
```

| 节点 | 作用 |
| --- | --- |
| `node_entry` | 按后缀路由 PDF / MD |
| `node_pdf_to_md` | MinerU 解析 PDF，得到 Markdown |
| `node_md_img` | VL 模型给图片写摘要，图片上传 MinIO 并改写 MD 链接 |
| `node_document_split` | 按长度/重叠切块 |
| `node_item_name_recognition` | LLM 识别商品/型号 |
| `node_bge_embedding` | BGE-M3 生成 dense / sparse |
| `node_import_milvus` | 写入切片集合与商品名集合 |

上传后为每个文件生成 `task_id`，后台执行图；前端轮询 `/status/{task_id}` 查看 `pending` / `processing` / `completed` / `failed` 及已完成节点。

## 查询流水线

`KBQueryWorkflow`（`processor/query_processor/main_graph.py`）：

```
node_item_name_confirm
    ├─ 已有明确答案（反问/拒答）→ node_answer_output
    └─ 确认型号后 → 并行：
           node_search_embedding
           node_search_embedding_hyde
           node_web_search_mcp
         → node_rrf → node_rerank → node_answer_output
```

- 问题过宽、库内有多个相近型号：先反问用户，不检索。
- 库中无该型号：直接说明未找到。
- 生成提示词要求不编造参考内容以外的信息；结构类问题可在答案末尾输出 `【图片】` 区块。

## HTTP API

### 导入服务（8000）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/import.html` | 上传页 |
| POST | `/upload` | `multipart/form-data` 上传 PDF/MD，返回 `task_ids` |
| GET | `/status/{task_id}` | 任务状态与节点进度 |

### 查询服务（8001）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/chat.html` | 对话页 |
| POST | `/query` | body：`query`、`session_id`（可选）、`is_stream` |
| GET | `/stream/{session_id}` | SSE，配合 `is_stream: true` |
| GET | `/history/{session_id}` | 会话历史 |
| DELETE | `/history/{session_id}` | 清空历史 |
| GET | `/health` | 探活 |

流式用法：先 `POST /query` 且 `is_stream: true`，再用同一 `session_id` 连接 `/stream/{session_id}`。

## 评估

`shopkeeper_brain/eval/eval.py` 读取 `qa.csv`（列：`question`,`ground_truth`），逐条跑查询图，用 ragas 计算忠实度、答案相关性、上下文精确率/召回率、答案正确性，结果写入 `qa_result.csv`。

```bash
uv run python shopkeeper_brain/eval/eval.py
uv run python shopkeeper_brain/eval/eval.py --limit 3
```

评估依赖已启动的 Milvus 等服务，以及与线上一致的 `.env`。

## 主要环境变量

完整列表见 `.env.example`。常用项：

| 变量 | 含义 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_API_BASE` | DashScope 兼容接口 |
| `LLM_DEFAULT_MODEL` / `VL_MODEL` / `ITEM_MODEL` | 问答、图片摘要、商品名相关模型 |
| `BGE_M3_PATH` / `BGE_DEVICE` / `BGE_FP16` | 本地嵌入 |
| `MILVUS_URL` / `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` | 向量库 |
| `MONGO_URL` / `MONGO_DB_NAME` | 会话库 |
| `MINIO_*` / `MINIO_IMG_DIR` | 原文与图片存储 |
| `MINERU_API_TOKEN` / `MINERU_BASE_URL` | PDF 解析 |
| `MCP_DASHSCOPE_BASE_URL` | 联网搜索 |
| `TEXT_RERANK_MODEL` | 重排 |
| `DATA_BASED_ROOT_DIR` | 导入文件本地根目录 |

不要把真实密钥提交进仓库。`.env` 仅本机使用。

## 说明

- `test/` 为实验代码，不是生产入口。
- `main.py` 仅为 uv 项目占位。
- 导入/查询任务进度默认存在进程内存中，重启服务会丢失进行中的状态；会话文本在 MongoDB。
