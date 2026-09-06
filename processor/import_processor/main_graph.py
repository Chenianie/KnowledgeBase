import json

from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_md_img import NodeMdImg
from processor.import_processor.nodes.node_pdf_to_md import NodePdfToMd
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from langgraph.graph import StateGraph,END

from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    def __init__(self,config=None):
        """
            初始化工作流
        """
        self._complied_graph = None

    @property
    def graph(self):
        if self._complied_graph is None:
            self._complied_graph=self.build_graph()
        return self._complied_graph

    @staticmethod #静态方法不需要改变什么，也就不需要self
    def router_after_entry(state:ImportGraphState)->str:
        """
        入口节点后的条件路由函数
        :param state: 当前状态
        :return: 下一个节点名称
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END

    def build_graph(self):
        """
              创建图结构
              :return: 编译后的图
        """
        # 1. 初始化LangGraph状态图
        graph = StateGraph(ImportGraphState)

        # 2. 注册节点到工作流
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePdfToMd())
        graph.add_node("node_md_img", NodeMdImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 3. 设置入口节点
        graph.set_entry_point('node_entry')

        # 4. 注册条件边
        graph.add_conditional_edges(
            'node_entry',
            self.router_after_entry,
            {
                'node_md_img':'node_md_img',
                'node_pdf_to_md':'node_pdf_to_md',
                END:END
            }
        )

        graph.add_edge('node_pdf_to_md','node_md_img')
        graph.add_edge('node_md_img','node_document_split')
        graph.add_edge('node_document_split','node_item_name_recognition')
        graph.add_edge('node_item_name_recognition','node_bge_embedding')
        graph.add_edge('node_bge_embedding','node_import_milvus')
        graph.add_edge('node_import_milvus',END)

        return graph.compile()

    def run(self,state:ImportGraphState,stream:bool=False):
        """
                统一执行入口，支持切换invoke/stream
                :param state: 初始状态
                :param stream: 是否流式输出
                :return:执行结果
        """

        self.graph.get_graph().print_ascii()

        if stream:
            return self.graph.stream(state)
        else:
            return self.graph.invoke(state)

if __name__ == "__main__":
    # 定义初始状态
    init_state = {
        "import_file_path": r"D:\doc\hak180使用说明书.pdf",
        'file_dir': r"D:\output"}

    # 方式1：实例化后使用（推荐方式，可复用）
    workflow = KBImportWorkflow()
    for event in workflow.run(init_state, stream=True):
        print(f"state: {event}")

    # # 方式2：非流式执行
    # final_state = workflow.run(init_state, stream=False)
    # print(json.dumps(final_state, ensure_ascii=False, indent=4))