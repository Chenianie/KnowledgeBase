import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


class NodePdfToMd(BaseNode):
    """
       PDF 转 Markdown 节点：PDF结构化解析
    """

    name = 'node_pdf_to_md'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
                :param state: `pdf_path`、`file_dir`
                :return: `md_path`、`md_content`
        """
        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_path_obj = self._step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_path_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state['md_path'] = str(md_path)
        state['md_content'] = md_content

        return state

    def _step_1_validate_paths(self, state: ImportGraphState):
        """
                步骤1：校验PDF文件路径和输出目录
                核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
                返回：合法的PDF文件Path对象、输出目录Path对象
                异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
        """
        # 1、参数非空校验
        pdf_path = state.get('pdf_path')
        if not pdf_path:
            raise StateFieldError(node_name=self.name, field_name='pdf_path', expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str)

        # 2、转换为Path对象统一处理路径
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3、PDF文件有效性校验
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF文件不存在: {pdf_path_obj}", node_name=self.name)

        # 4、确保输出目录存在，不存在则递归创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录不存在，自动创建：{file_dir_obj.absolute()}")  # 目录绝对路径
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
                步骤2：上传PDF至MinerU并轮询解析任务状态
                核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
                参数：pdf_path_obj-已校验的PDF Path对象
                返回：解析结果ZIP包下载链接full_zip_url
                异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2、从MinerU服务器获取上传链接
        token = self.config.mineru_api_token
        url = f'{self.config.mineru_base_url}/file-urls/batch'
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }


        response = requests.post(url, headers=header, json=data)
        if response.status_code != 200:
            raise PdfConversionError(
                message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")

            # 校验业务码
        result = response.json()
        if result.get('code') != 0:
            raise PdfConversionError(f"获取上传链接失败：返回数据：{result}")

            # 获取响应结果
        batch_id = result['data']['batch_id']
        signed_url = result['data']['file_urls'][0]  # 目前做的单个文件上传

            # 3、单个文件上传
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(signed_url, data=f)  # 文件上传，将上传回来的数据赋给res_upload
            if res_upload.status_code != 200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")
            self.logger.info('文件上传成功')

            #  4、批量获取任务结果
        poll_url = f'{self.config.mineru_base_url}/extract-results/batch/{batch_id}'

        start_time = time.time()  # 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔时间
        self.logger.info(f"【任务轮询】最大超时：{timeout_seconds}s，batch_id：{batch_id}")
            # 根据batch_id轮询任务状态直到成功"done"
        while True:
                # 已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")

                # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(poll_url, headers=header,timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，batch_id：{batch_id}")
                time.sleep(poll_interval)
                continue

            # 处理HTTP响应错误
            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】轮询失败：状态码：{res_poll.status_code}，响应结果：{res_poll}")

            # 解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data.get('code') != 0:
                raise PdfConversionError(f"【任务轮询】轮询失败：返回数据：{poll_data}")

            extract_result = poll_data['data']['extract_result']

            # 获取单个结果
            result_item = extract_result[0]
            data_state = result_item['state']

            # 状态为 done
            if data_state == 'done':
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，batch_id：{batch_id}")

                full_zip_url = result_item['full_zip_url']
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，batch_id：{batch_id}")

                return full_zip_url
            elif data_state == 'failed':
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败：错误信息：{err_msg}，batch_id：{batch_id}")
            else:
                self.logger.info(f"【任务轮询】任务未完成，状态：{data_state}，batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url:str, output_path_obj:Path, pdf_stem:str)->str:
        """
               步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
               核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
               参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
               返回：最终MD文件的字符串格式绝对路径
               异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
        """
        # 1、下载ZIP包
        self.logger.info(f"【下载ZIP】开始下载ZIP包：{zip_url}")
        response = requests.get(zip_url)

        # 对响应结果进行校验
        if response.status_code !=200:
            raise RuntimeError(f"【ZIP下载】ZIP包下载失败：状态码：{response.status_code}，响应结果：{response}")

        # 拼接ZIP包保存路径并保存
        zip_save_path = output_path_obj /f'{pdf_stem}_result.zip'
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        self.logger.info(f"【下载ZIP】ZIP包保存成功：{zip_save_path}")

        # 2. 如果目标文件夹已存在，先删除（确保环境干净）
        extract_target_dir = output_path_obj/pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)#删除目标文件夹
        self.logger.info(f"【清理】已存在目标文件夹，删除：{extract_target_dir}")

        # 3、创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 4、解压ZIP包
        self.logger.info(f"【解压ZIP】开始解压ZIP包：{zip_save_path}")
        with zipfile.ZipFile(zip_save_path,'r') as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir) #解压，解压到目标文件夹
        self.logger.info(f"【解压ZIP】ZIP包解压成功：{extract_target_dir}")

        # 5、重命名
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / 'full.md'
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f'{pdf_stem}.md')
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")
        return str(new_md_path.absolute())


if __name__ == '__main__':
    setup_logging()

    node_pdf_to_md = NodePdfToMd()

    init_state= {
        'pdf_path':"D:\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf",
        'file_dir':'D:\output'
    }
    res = node_pdf_to_md(init_state)
    logging.getLogger().info(json.dumps(res,ensure_ascii=False,indent=4))