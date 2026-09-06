# -*- coding: utf-8 -*-
"""
================================================================================
 eval.py —— 基于 ragas 的 RAG 流程离线评估程序
================================================================================

【功能概述】
    1. 读取本脚本所在 eval 目录下的 qa.csv（默认列：question, ground_truth），
       逐条调用项目自身的 RAG 查询流程（processor/query_processor 的 KBQueryWorkflow）。
    2. 对每条问题取流程最终 state 中的重排结果（rerank_result / reranked_docs）作为
       检索上下文 context，取流程最终输出的 answer 作为回答。
    3. 使用 ragas 框架对以下 5 个指标进行批量评估：
           faithfulness（忠实度）
           answer_relevancy（答案相关性）
           context_precision（上下文精确率）
           context_recall（上下文召回率）
           answer_correctness（答案正确性）
    4. 评估结果写入本目录的 qa_result.csv，包含列：
           question, context, Answer, ground_truth,
           faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
    5. 评估使用的 LLM 与 Embedding 均直接复用项目工具函数：
           - LLM：      processor/utils/llm_utils.py 中的 get_llm_client()
           - Embedding：processor/utils/embedding_utils.py 中的 generate_embeddings()
                        （本地 BGE-M3 模型的 dense 向量，封装为 langchain Embeddings 接口）

【运行方式】
    在仓库根目录执行（uv 环境）：
        uv run python shopkeeper_brain/eval/eval.py
    或 cd 到 eval 目录后：
        uv run python eval.py

【命令行参数】
    --qa-input      输入 QA 文件路径（默认：本脚本同目录 qa.csv）
    --output        结果输出路径（默认：本脚本同目录 qa_result.csv）
    --limit         仅评估前 N 条（用于快速调试，默认评估全部）

【设计约定】
    - 函数分为：核心步骤函数（step_1_xxx ~ step_6_xxx）与私有辅助函数（_ 开头）。
    - 核心步骤按顺序执行、每一步有清晰函数说明与日志，任一步骤出现致命错误会终止。
    - 单条问题跑流程出错（如外部服务不可用）不会中断整体评估，只会在结果中留空并记录原因。
    - 本脚本为新增文件，不修改仓库内任何原有代码。

【兼容性说明】
    ragas 0.4.3 在导入时会无条件执行
        from langchain_community.chat_models.vertexai import ChatVertexAI
    而环境中锁定的 langchain-community 0.4.2 已移除该模块，导致 ragas 无法导入。
    本脚本在导入 ragas 之前先注入一个同名“占位模块”满足该 import（该符号仅被 ragas
    用于 isinstance 判断，不影响真实指标计算）。若日后升级依赖修复后，占位逻辑自动失效。
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
import types
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# 日志输出初始化（简单打印风格）
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_eval")


# ============================================================================
# 私有辅助函数区（_ 开头，非核心步骤）
# ============================================================================

def _find_repo_root(start_dir: Path) -> Path:
    """
    向上逐级查找仓库根目录（以存在 pyproject.toml 的目录为准）。

    无论 eval.py 被放在仓库下的任何层级（例如 <root>/eval 或
    <root>/shopkeeper_brain/eval），都能正确定位到仓库根目录，
    从而保证 processor / tool 等包可以被正常导入。
    """
    for parent in start_dir.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # 找不到时退回当前目录，交由后续 import 报错提示
    return start_dir


def _ensure_ragas_compat() -> None:
    """
    ragas 0.4.3 导入兼容补丁。

    在 import ragas 之前，把缺失的 langchain_community.chat_models.vertexai
    以“占位模块”形式注入 sys.modules，使其顶部的 import 语句能顺利通过。
    占位中的 ChatVertexAI 仅被 ragas 用作 isinstance 判定，评估过程不会被真正调用。
    """
    try:
        # 若目标环境已修复依赖（模块真实存在），则什么都不做
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        """占位类：仅用于满足 ragas 顶部的 import，不被实际调用。"""

    stub.ChatVertexAI = ChatVertexAI
    sys.modules.setdefault("langchain_community.chat_models.vertexai", stub)


# 必须在导入任何 ragas 模块之前调用兼容补丁
_ensure_ragas_compat()

# ============================================================================
# 仓库根目录 / 环境变量加载
# ============================================================================
_EVAL_DIR = Path(__file__).resolve().parent          # 脚本所在目录（即 eval 目录）
_REPO_ROOT = _find_repo_root(_EVAL_DIR)              # 仓库根目录

# 保证 processor、tool 等包可被 import（脚本可能从任意目录被调用）
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 加载仓库根目录下的 .env（若存在），保证后续 get_llm_client / 各节点配置可读到密钥
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # dotenv 缺失也不阻塞主流程，各节点自带 load_dotenv
    logger.warning("未找到 python-dotenv，跳过 .env 自动加载")

# ============================================================================
# ragas 相关导入（位于兼容补丁之后）
# ============================================================================
from ragas import evaluate                                          # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.run_config import RunConfig  # noqa: E402  ← 新增

# 5 个评估指标对应的 ragas 指标实例
# 说明：faithfulness 只用 LLM；answer_relevancy / answer_correctness 需要 LLM + Embedding

# ragas 指标 → 输出 CSV 列名（与需求给定的列名保持一致）
RAGAS_METRICS: List[Any] = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]
OUTPUT_METRIC_COLUMNS: List[str] = [
    "faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
    "Answer Correctness",
]

# 输出 CSV 的完整列定义（顺序与需求一致）
CSV_HEADERS: List[str] = [
    "question",
    "context",
    "Answer",
    "ground_truth",
] + OUTPUT_METRIC_COLUMNS

# 单次运行内对"缺失指标"的最大补测轮数（ragas 单 job 偶发网络错误会被吞成 NaN）
_MAX_FILL_ROUNDS = 3

# 必须继承 langchain_core.embeddings.Embeddings，而不是写成"看起来兼容"的普通类：
# ragas 的 evaluate() 仅在 isinstance(embeddings, LangchainEmbeddings) 成立时才会把对象
# 包装成 LangchainEmbeddingsWrapper（见 ragas/evaluation.py 的 aevaluate），该包装器
# (BaseRagasEmbeddings) 提供 AnswerSimilarity._ascore legacy 分支所需的 async embed_text()。
# 若不继承，answer_correctness 的"语义相似度"部分会因对象缺少 embed_text 抛 AttributeError，
# 异常被 ragas Executor 捕获并替换为 NaN，导致该指标整列为空（其他 4 个指标不受影响）。
from langchain_core.embeddings import Embeddings as LangchainEmbeddings  # noqa: E402


class ProjectBGEMM3Embeddings(LangchainEmbeddings):
    """
    把项目自有的本地 BGE-M3 模型封装为 langchain Embeddings 子类，
    供 ragas 的 answer_relevancy / answer_correctness 等指标使用。

    - embed_documents：复用 processor/utils/embedding_utils.generate_embeddings()
    - embed_query    ：复用 processor/utils/embedding_utils.get_bge_m3_ef().encode_queries()
    仅取 dense 向量（1024 维）。

    继承 Embeddings 后，ragas evaluate() 会将其自动包装成
    LangchainEmbeddingsWrapper（BaseRagasEmbeddings），使 AnswerSimilarity
    走如下 legacy 异步链路：
        embed_text -> embed_texts(is_async=True) -> aembed_documents
    因此下方需同时提供 aembed_documents / aembed_query 两个异步方法。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量把文档/片段文本转成 dense 向量列表。"""
        from processor.utils.embedding_utils import generate_embeddings

        texts = [t if isinstance(t, str) else str(t) for t in texts]
        dense = generate_embeddings(texts)["dense"]
        # generate_embeddings 已返回 list[list[float]]，这里兜底做一次转换
        return [[float(x) for x in vec] for vec in dense]

    def embed_query(self, text: str) -> List[float]:
        """把单个查询文本编码为 dense 向量（使用 BGE-M3 的查询编码器）。"""
        from processor.utils.embedding_utils import get_bge_m3_ef

        ef = get_bge_m3_ef()
        dense = ef.encode_queries([str(text)])["dense"][0]
        return [float(x) for x in dense]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量编码文档：委托给同步实现，用线程池避免阻塞事件循环。

        说明：ragas 的 AnswerSimilarity 会走 legacy 调用链
            BaseRagasEmbeddings.embed_text → embed_texts(is_async=True)
            → aembed_documents
        若封装类缺少本方法，answer_correctness 的“语义相似度”部分会抛
        AttributeError，导致该指标整列为空（其余指标不受影响）。
        """
        import asyncio

        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步编码单个查询：与 aembed_documents 同理，补充完整异步接口。"""
        import asyncio

        return await asyncio.to_thread(self.embed_query, text)


def _to_doc_text(doc: Any) -> str:
    """
    从重排结果中的单个文档对象里提取正文文本。

    rag 流程的文档字典通常包含 content / text / snippet / page_content 等字段，
    这里按常见优先级逐个尝试，兼容不同实现。
    """
    if doc is None:
        return ""
    if isinstance(doc, str):
        return doc.strip()
    if isinstance(doc, dict):
        for key in ("content", "text", "snippet", "page_content", "content_text"):
            if doc.get(key):
                return str(doc[key]).strip()
        return ""
    return str(doc).strip()


def _extract_rerank_contexts(state: Dict[str, Any]) -> List[str]:
    """
    从流程返回的最终 state 中提取重排后的检索片段文本列表。

    需求中描述的字段为 rerank_result；当前代码库实现中该字段名为 reranked_docs，
    因此这里按顺序兼容两者，取到哪个用哪个。
    """
    # 1) 优先取需求中约定的 rerank_result
    rerank = state.get("rerank_result")
    # 2) 兼容当前仓库实际实现的 reranked_docs
    if not rerank:
        rerank = state.get("reranked_docs")

    if rerank is None:
        return []
    # 3) 某些实现下可能是 {"docs": [...]} 之类的字典包装
    if isinstance(rerank, dict):
        for key in ("docs", "documents", "chunks", "contexts"):
            if isinstance(rerank.get(key), list):
                rerank = rerank[key]
                break

    if not isinstance(rerank, (list, tuple)):
        return []

    contexts: List[str] = []
    for doc in rerank:
        text = _to_doc_text(doc)
        if text:  # 过滤空片段
            contexts.append(text)
    return contexts


def _fmt_score(value: Any) -> str:
    """
    把 ragas 返回的原始得分规范化为可写入 CSV 的字符串。

    - 兼容 MetricResult 包装（取 .score）
    - NaN / inf / None 一律输出空字符串，避免污染 CSV
    """
    # 兼容 MetricResult 之类的包装对象
    if hasattr(value, "score"):
        value = value.score
    try:
        f = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    return f"{f:.4f}"


def _collect_metric_floats(rows: List[Dict[str, Any]], column: str) -> List[float]:
    """
    收集指定指标列在若干行中的有效得分（float 列表），供统计平均值使用。

    说明：_fmt_score 会把 None / NaN / inf / 非数值统一转成空字符串，
    空字符串表示该样本在该指标上没有有效得分（评估未返回分值），
    统计平均时应直接跳过而不是报错（这正是 float("") 抛异常的根源）。
    """
    floats: List[float] = []
    for row in rows:
        text = _fmt_score(row.get("metric_" + column))
        if text:  # 空字符串 = 无有效得分，跳过
            floats.append(float(text))
    return floats


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于 ragas 的 RAG 流程离线评估程序（读取 qa.csv 并输出 qa_result.csv）",
    )
    parser.add_argument("--qa-input", type=str, default="",
                        help="输入 QA 文件路径（默认：本脚本同目录下的 qa.csv）")
    parser.add_argument("--output", type=str, default="",
                        help="结果保存路径（默认：本脚本同目录下的 qa_result.csv）")
    parser.add_argument("--limit", type=int, default=0,
                        help="仅评估前 N 条，便于快速调试（默认 0 = 全部）")
    return parser.parse_args()


def _resolve_input_path(explicit: str) -> Path:
    """
    解析 qa.csv 的输入路径，按优先级：
      1) 命令行显式指定的 --qa-input
      2) 脚本同目录 qa.csv（常见布局：eval/qa.csv 与 eval/eval.py 同级）
      3) 当前目录下的 eval/qa.csv
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (_REPO_ROOT / p)
    p1 = _EVAL_DIR / "qa.csv"
    if p1.exists():
        return p1
    p2 = Path.cwd() / "eval" / "qa.csv"
    if p2.exists():
        return p2
    return p1  # 不存在时返回默认路径，由读取步骤给出明确报错


def _resolve_output_path(explicit: str) -> Path:
    """解析结果输出路径，默认保存到脚本同目录的 qa_result.csv。"""
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (_REPO_ROOT / p)
    return _EVAL_DIR / "qa_result.csv"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """按 utf-8-sig / gbk 顺序读取 CSV 并返回字典列表，兼容中文 Excel 导出的文件。"""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("qa.csv 编码无法识别，请转换为 utf-8 或 gbk")


# ============================================================================
# 核心步骤函数（step_1_xxx ~ step_6_xxx）
# ============================================================================

def step_1_load_qa_dataset(ctx: Dict[str, Any]) -> None:
    """
    步骤 1：读取 qa.csv，取出每条样本的 question 与 ground_truth。

    读取结果写入 ctx["rows"]，每条记录结构：
        {"question": str, "ground_truth": str,
         "context": str, "contexts": list[str], "answer": str, "error": str|None}
    context / contexts / answer 先置空，由步骤 3 跑流程后填充。
    """
    logger.info("=" * 70)
    logger.info("步骤 1/6：读取 QA 数据集 ...")
    qa_path: Path = ctx["qa_path"]
    if not qa_path.exists():
        raise FileNotFoundError(f"未找到 QA 文件：{qa_path}")

    records = _read_csv_rows(qa_path)
    if not records:
        raise ValueError(f"QA 文件为空或无有效表头：{qa_path}")

    # 校验必须列
    headers = list(records[0].keys())
    missing = [c for c in ("question", "ground_truth") if c not in headers]
    if missing:
        raise ValueError(f"qa.csv 缺少必要列 {missing}，实际表头为 {headers}")

    rows: List[Dict[str, Any]] = []
    for rec in records:
        q = (rec.get("question") or "").strip()
        gt = (rec.get("ground_truth") or "").strip()
        if not q:  # 跳过空行
            continue
        rows.append({
            "question": q,
            "ground_truth": gt,
            "context": "",       # 检索上下文（合并后的文本）
            "contexts": [],      # 检索上下文（片段列表，供 ragas 使用）
            "answer": "",        # 流程输出的最终答案
            "error": None,       # 流程运行错误信息
        })

    # 支持 --limit 快速调试
    limit = ctx["args"].limit
    if limit and limit > 0:
        rows = rows[:limit]

    ctx["rows"] = rows
    logger.info("成功加载 %d 条测试问题（来自 %s）", len(rows), qa_path.name)


def step_2_prepare_ragas_models(ctx: Dict[str, Any]) -> None:
    """
    步骤 2：准备 ragas 评估所需的 LLM 与 Embedding。

    - LLM      ：复用项目工具 processor/utils/llm_utils.py 的 get_llm_client()
                  （返回项目默认模型的 langchain ChatOpenAI 客户端）
    - Embedding：复用项目本地 BGE-M3 模型（封装成 langchain Embeddings 接口）
    上述对象会在步骤 5 的 evaluate() 中被 ragas 自动包装成其内部 llm/embeddings。
    """
    logger.info("=" * 70)
    logger.info("步骤 2/6：初始化评估模型（LLM + Embedding）...")

    # 复用项目 llm_utils —— 与 RAG 流程使用同一个 LLM 客户端
    from processor.utils.llm_utils import get_llm_client

    llm = get_llm_client()                 # langchain ChatOpenAI（默认模型，非 json 模式）
    embeddings = ProjectBGEMM3Embeddings() # 项目 BGE-M3 dense 向量（langchain 兼容封装）

    ctx["llm"] = llm
    ctx["embeddings"] = embeddings
    ctx["metrics"] = RAGAS_METRICS

    logger.info("LLM 模型 : %s", getattr(llm, "model_name", getattr(llm, "model", "?")))
    logger.info("Embedding: 项目本地 BGE-M3（dense）")


def step_3_run_rag_flow(ctx: Dict[str, Any]) -> None:
    """
    步骤 3：逐条调用项目自身的 RAG 查询流程，收集 context 与 answer。

    对每条 question：
        1) 生成独立的 session_id，避免不同样本之间互相干扰；
        2) 以 {"session_id", "original_query", "is_stream": False} 作为初始 state
           调用 processor.query_processor.main_graph 的 KBQueryWorkflow；
        3) 流程结束后，从最终 state 中提取：
               context —— 重排结果（rerank_result / reranked_docs）各片段文本；
               answer —— 流程输出的最终答案。
    单条失败不会中断整体，错误信息写入该条记录并继续后续样本。
    """
    logger.info("=" * 70)
    logger.info("步骤 3/6：逐条运行 RAG 查询流程 ...")

    # 懒加载流程类（导入会连带初始化 Mongo 等外部连接，故放在此步执行）
    from processor.query_processor.main_graph import KBQueryWorkflow

    workflow = KBQueryWorkflow()  # 编译一次，所有样本复用
    rows: List[Dict[str, Any]] = ctx["rows"]
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        question = row["question"]
        # 每次提问使用新的 session_id，保证历史上下文互不影响
        session_id = f"eval_{uuid.uuid4().hex[:16]}"
        init_state: Dict[str, Any] = {
            "session_id": session_id,
            "original_query": question,
            "is_stream": False,
        }
        logger.info("  [%d/%d] 提问: %s", i, total, question[:60])
        try:
            # 运行完整 RAG 流程，返回最终 state
            final_state: Dict[str, Any] = workflow.run(init_state)

            # 从最终 state 中取重排结果作为检索上下文
            contexts = _extract_rerank_contexts(final_state)
            row["contexts"] = contexts
            row["context"] = "\n".join(contexts)

            # 取流程输出的最终答案
            answer = final_state.get("answer")
            row["answer"] = str(answer).strip() if answer else ""

            logger.info("      → 检索片段 %d 段, 答案长度 %d 字符",
                        len(contexts), len(row["answer"]))
        except Exception as exc:  # 单条失败不影响整体评估
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("      → 该条运行失败: %s", row["error"])

    ok = sum(1 for r in rows if r["error"] is None)
    logger.info("流程运行完成：成功 %d / 共 %d", ok, total)


def step_4_assemble_eval_dataset(ctx: Dict[str, Any]) -> None:
    """
    步骤 4：把可评估样本组装成 ragas 的 EvaluationDataset。

    只有同时满足以下条件的样本才会参与 ragas 评估：
        - 流程运行成功（无 error）；
        - 有最终答案 answer；
        - 有至少 1 个检索片段（contexts 非空）。
    失败 / 上下文缺失的样本会保留在 CSV 中，但指标留空。

    ragas SingleTurnSample 字段映射：
        user_input        ← question
        retrieved_contexts ← 重排后的检索片段列表（每个片段独立元素）
        response          ← 流程输出的最终答案
        reference         ← 数据集中的标准答案 ground_truth
    """
    logger.info("=" * 70)
    logger.info("步骤 4/6：组装 ragas 评估数据集 ...")

    samples: List[SingleTurnSample] = []
    scored_rows: List[Dict[str, Any]] = []  # 与 samples 一一对应，便于回填分数

    for row in ctx["rows"]:
        if row["error"] is not None:
            continue  # 流程运行失败，跳过评估
        if not row["answer"]:
            logger.warning("样本无答案，跳过评估: %s", row["question"][:40])
            continue
        if not row["contexts"]:
            logger.warning("样本检索上下文为空，跳过评估: %s", row["question"][:40])
            continue

        samples.append(SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=row["contexts"],
            response=row["answer"],
            reference=row["ground_truth"],
        ))
        scored_rows.append(row)

    ctx["dataset"] = EvaluationDataset(samples=samples) if samples else None
    ctx["scored_rows"] = scored_rows
    logger.info("可参与 ragas 评估的样本数：%d", len(samples))


def _row_to_sample(row: Dict[str, Any]) -> SingleTurnSample:
    """把一行流程结果转成 ragas SingleTurnSample（与 step_4 组装逻辑保持一致）。"""
    return SingleTurnSample(
        user_input=row["question"],
        retrieved_contexts=row["contexts"],
        response=row["answer"],
        reference=row["ground_truth"],
    )


def _backfill_scores(rows: List[Dict[str, Any]], result: Any) -> None:
    """
    把一次 evaluate() 的逐样本得分回填到对应行（metric_ 前缀暂存）。

    rows 必须与 result.scores 按相同顺序一一对应；
    每个样本的得分以 metric_<输出列名> 为键写入，避免与最终输出列混淆。
    """
    metric_index: Dict[str, int] = {}
    for i, metric in enumerate(RAGAS_METRICS):
        metric_index[getattr(metric, "name", str(i))] = i

    for row, row_scores in zip(rows, result.scores):
        for metric_name, col_index in metric_index.items():
            label = OUTPUT_METRIC_COLUMNS[col_index]
            row["metric_" + label] = row_scores.get(metric_name)


def step_5_run_ragas_evaluation(ctx: Dict[str, Any]) -> None:
    """
    步骤 5：调用 ragas 批量评估 5 个指标，并对偶发失败的样本自动补测。

    使用 ragas.evaluate()，指标依次为：
        faithfulness / answer_relevancy / context_precision /
        context_recall / answer_correctness
    llm 与 embeddings 直接传步骤 2 中准备的项目模型，
    ragas 内部会自动完成对指标对象的注入，无需手工绑定。

    说明：ragas 的单个 job（样本 × 指标）若遇到 LLM 网络抖动（如
    APIConnectionError），在 raise_exceptions=False 下会被 Executor 吞成
    NaN，导致该样本该指标在 CSV 中留空。本步骤完成首轮完整评估后检测缺失项，
    仅对"仍有缺失的样本"按缺失指标重新评估，最多补测 _MAX_FILL_ROUNDS 轮，
    轮间等待时间递增，尽量消化偶发的网络/服务端抖动，避免产出残缺结果。
    """
    scored_rows = ctx.get("scored_rows", [])
    if ctx.get("dataset") is None or not scored_rows:
        logger.info("步骤 5/6：无可评估样本，跳过 ragas 评估。")
        ctx["raw_scores"] = []
        return

    logger.info("=" * 70)
    logger.info("步骤 5/6：执行 ragas 评估（共 %d 条 × %d 个指标）...",
                len(scored_rows), len(RAGAS_METRICS))

    def _evaluate(rows: List[Dict[str, Any]],
                  metrics: List[Any]) -> Any:
        """对 rows 对应样本执行一次 evaluate，并把得分回填到 rows。"""
        samples = [_row_to_sample(r) for r in rows]
        sub_dataset = EvaluationDataset(samples=samples)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = evaluate(
                dataset=sub_dataset,
                metrics=metrics,
                llm=ctx["llm"],
                embeddings=ctx["embeddings"],
                run_config=RunConfig(max_workers=4),  # 降低并发，避免16路同时打百炼排队导致超时
                raise_exceptions=False,  # 失败样本不中断整体，交由下方补测逻辑兜底
            )
        _backfill_scores(rows, result)
        return result

    # ---- 第 1 轮：全部样本 × 全部指标 ----
    result = _evaluate(scored_rows, RAGAS_METRICS)
    ctx["raw_scores"] = result.scores  # 与 scored_rows 对齐的原始得分

    # ---- 补测轮：仅重跑仍有缺失指标的样本 ----
    for attempt in range(1, _MAX_FILL_ROUNDS + 1):
        pending = [
            r for r in scored_rows
            if any(not _fmt_score(r.get("metric_" + col))
                   for col in OUTPUT_METRIC_COLUMNS)
        ]
        if not pending:
            break
        missing_cols = sorted({
            col for r in pending
            for col in OUTPUT_METRIC_COLUMNS
            if not _fmt_score(r.get("metric_" + col))
        })
        fill_metrics = [
            m for m, col in zip(RAGAS_METRICS, OUTPUT_METRIC_COLUMNS)
            if col in missing_cols
        ]
        wait_sec = 30 * attempt  # 给网络 / 服务端留恢复时间
        logger.warning(
            "检测到 %d 条样本缺失指标 %s；等待 %d 秒后执行第 %d/%d 轮补测 ...",
            len(pending), missing_cols, wait_sec, attempt, _MAX_FILL_ROUNDS)
        time.sleep(wait_sec)
        _evaluate(pending, fill_metrics)

    # 补测结束后仍缺失的项告警（对应 CSV 单元格将留空）
    still_missing = [
        (r["question"][:40], col)
        for r in scored_rows
        for col in OUTPUT_METRIC_COLUMNS
        if not _fmt_score(r.get("metric_" + col))
    ]
    if still_missing:
        logger.warning("补测后仍有 %d 项缺失（单元格将留空）：%s",
                       len(still_missing), still_missing)

    logger.info("ragas 评估完成。")
    # 打印每个样本的平均分便于观察（仅对 5 个指标的有效得分取平均）
    for i, row in enumerate(scored_rows):
        vals = [v for col in OUTPUT_METRIC_COLUMNS
                for v in _collect_metric_floats([row], col)]
        mean = sum(vals) / len(vals) if vals else 0.0
        logger.info("  #%d 平均分=%.4f | %s", i + 1, mean, row["question"][:40])


def step_6_save_result_csv(ctx: Dict[str, Any]) -> None:
    """
    步骤 6：把评估结果写入 qa_result.csv。

    输出列与需求一致：
        question, context, Answer, ground_truth,
        faithfulness, Answer Relevancy, Context Precision, Context Recall, Answer Correctness
    其中：
        context = 流程 state 重排结果合并文本；
        Answer  = 流程输出的最终答案；
        各指标值 = ragas 评估得分（失败样本留空）。
    编码使用 utf-8-sig，方便直接用 Excel 打开查看。
    """
    logger.info("=" * 70)
    logger.info("步骤 6/6：保存评估结果 ...")

    output_path: Path = ctx["output_path"]
    rows = ctx["rows"]
    scored_rows = set(map(id, ctx.get("scored_rows", [])))  # 参与评分的行对象集合

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: Dict[str, str] = {
                "question": row["question"],
                "context": row["context"],
                "Answer": row["answer"],
                "ground_truth": row["ground_truth"],
            }
            if id(row) in scored_rows:
                # 对参与评分的行写入 5 个指标得分
                for col in OUTPUT_METRIC_COLUMNS:
                    out[col] = _fmt_score(row.get("metric_" + col))
            writer.writerow(out)

    logger.info("评估结果已保存: %s", output_path)

    # ---- 打印汇总：各指标的平均值（仅统计参与评分且有有效得分的样本） ----
    evaluated = [r for r in rows if id(r) in scored_rows]
    if evaluated:
        logger.info("---- 各指标平均分（基于 %d 条成功样本）----", len(evaluated))
        for col in OUTPUT_METRIC_COLUMNS:
            vals = _collect_metric_floats(evaluated, col)
            mean = sum(vals) / len(vals) if vals else float("nan")
            logger.info("    %-22s = %.4f", col, mean)


# ============================================================================
# 程序入口
# ============================================================================

def main() -> None:
    """
    程序主流程：依次执行 step_1 ~ step_6。

    各步骤通过共享的 ctx 字典传递数据：
        rows / llm / embeddings / dataset / scored_rows / raw_scores ...
    任一步骤抛出致命异常（如 QA 文件缺失、表头错误）即终止并打印原因。
    """
    args = _parse_args()

    # 初始化共享上下文
    ctx: Dict[str, Any] = {
        "args": args,
        "qa_path": _resolve_input_path(args.qa_input),
        "output_path": _resolve_output_path(args.output),
        "rows": [],
    }

    logger.info("仓库根目录 : %s", _REPO_ROOT)
    logger.info("QA 输入文件 : %s", ctx["qa_path"])
    logger.info("结果输出文件: %s", ctx["output_path"])

    # 依序执行核心步骤
    try:
        step_1_load_qa_dataset(ctx)
        step_2_prepare_ragas_models(ctx)
        step_3_run_rag_flow(ctx)
        step_4_assemble_eval_dataset(ctx)
        step_5_run_ragas_evaluation(ctx)
        step_6_save_result_csv(ctx)
        logger.info("全部完成 ✔ 结果文件：%s", ctx["output_path"])
    except KeyboardInterrupt:
        logger.warning("用户中断，评估已停止。")
        sys.exit(130)
    except Exception as exc:
        logger.error("评估流程终止：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
