from typing import Dict, Optional, Union

import json5

from qwen_agent.settings import DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_RAG_SEARCHERS
from qwen_agent.tools.base import TOOL_REGISTRY, BaseTool, register_tool
from qwen_agent.tools.doc_parser import DocParser, Record
from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES


def _check_deps_for_rag():
    try:
        import charset_normalizer  # noqa
        import jieba  # noqa
        import pdfminer  # noqa
        import pdfplumber  # noqa
        import rank_bm25  # noqa
        import snowballstemmer  # noqa
        from bs4 import BeautifulSoup  # noqa
        from docx import Document  # noqa
        from pptx import Presentation  # noqa
    except ImportError as e:
        raise ImportError('The dependencies for RAG support are not installed. '
                          'Please install the required dependencies by running: pip install "qwen-agent[rag]"') from e


@register_tool('retrieval')
class Retrieval(BaseTool):
    description = f'从给定文件列表中检索出和问题相关的内容，支持文件类型包括：{"/".join(PARSER_SUPPORTED_FILE_TYPES)}'
    # PARSER_SUPPORTED_FILE_TYPES: ['pdf', 'docx', 'pptx', 'txt', 'html', 'csv', 'tsv', 'xlsx', 'xls']
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '在这里列出关键词，用逗号分隔，目的是方便在文档中匹配到相关的内容，由于文档可能多语言，关键词最好中英文都有。',
        'required': True
    }, {
        'name': 'files',
        'type': 'array',
        'items': {
            'type': 'string'
        },
        'description': '待解析的文件路径列表，支持本地文件路径或可下载的http(s)链接。',
        'required': True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        # cfg: {
        #     'name': 'retrieval',
        #     'max_ref_token': 4000,
        #     'parser_page_size': 500,
        #     'rag_searchers': ['keyword_search', 'front_page_search']
        # }
        super().__init__(cfg)
        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        self.parser_page_size: int = self.cfg.get('parser_page_size', DEFAULT_PARSER_PAGE_SIZE)
        self.doc_parse = DocParser({'max_ref_token': self.max_ref_token, 'parser_page_size': self.parser_page_size})

        self.rag_searchers = self.cfg.get('rag_searchers', DEFAULT_RAG_SEARCHERS)
        if len(self.rag_searchers) == 1:
            self.search = TOOL_REGISTRY[self.rag_searchers[0]]({'max_ref_token': self.max_ref_token})
        else:
            from qwen_agent.tools.search_tools.hybrid_search import HybridSearch
            self.search = HybridSearch({'max_ref_token': self.max_ref_token, 'rag_searchers': self.rag_searchers})

    def call(self, params: Union[str, dict], **kwargs) -> list:
        """RAG tool.

        Step1: Parse and save files
        Step2: Retrieval related content according to query

        Args:
            params: The files and query.
        Returns:
            The parsed file list or retrieved file list.
        """
        # params: {'files': ['https://arxiv.org/pdf/1706.03762.pdf'], 'query': '{"keywords_zh": ["图二", "图 2"], "keywords_en": ["Figure 2"], "text": "介绍图二"}'}
        # kwargs: {}

        # TODO: It this a good place to check the RAG deps?
        _check_deps_for_rag()

        params = self._verify_json_format_args(params)
        files = params.get('files', [])
        # files: ['https://arxiv.org/pdf/1706.03762.pdf']
        if isinstance(files, str):
            files = json5.loads(files)
        records = []
        for file in files:
            # file: 'https://arxiv.org/pdf/1706.03762.pdf'
            # kwargs: {}
            _record = self.doc_parse.call(params={'url': file}, **kwargs)
            # _record: {
            #     'url': 'https://arxiv.org/pdf/1706.03762.pdf',
            #     'title': '1706.03762.pdf',
            #     'raw': [
            #         {'content': str, 'metadata': {'chunk_id': 0, 'source': 'https://arxiv.org/pdf/1706.03762.pdf', 'title': '1706.03762.pdf'}, 'token': 487}_0,
            #         {'content': str, 'metadata': {'chunk_id': 1, 'source': 'https://arxiv.org/pdf/1706.03762.pdf', 'title': '1706.03762.pdf'}, 'token': 450}_1,
            #         ...
            #         {'content': str, 'metadata': {'chunk_id': 23, 'source': 'https://arxiv.org/pdf/1706.03762.pdf', 'title': '1706.03762.pdf'}, 'token': 335}_23,
            #     ]
            # }
            records.append(_record)

        query = params.get('query', '')
        if records:
            return self.search.call(params={'query': query}, docs=[Record(**rec) for rec in records], **kwargs)
        else:
            return []
