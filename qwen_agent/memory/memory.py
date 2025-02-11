import json
from importlib import import_module
from typing import Dict, Iterator, List, Optional, Union

import json5

from qwen_agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, USER, Message
from qwen_agent.log import logger
from qwen_agent.settings import (DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_RAG_KEYGEN_STRATEGY,
                                 DEFAULT_RAG_SEARCHERS)
from qwen_agent.tools import BaseTool
from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES
from qwen_agent.utils.utils import extract_files_from_messages, extract_text_from_message, get_file_type


class Memory(Agent):
    """Memory is special agent for file management.

    By default, this memory can use retrieval tool for RAG.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None):
        """Initialization the memory.

        Args:
            rag_cfg: The config for RAG. One example is:
              {
                'max_ref_token': 4000,
                'parser_page_size': 500,
                'rag_keygen_strategy': 'SplitQueryThenGenKeyword',
                'rag_searchers': ['keyword_search', 'front_page_search']
              }
              And the above is the default settings.
        """
        # function_list: None
        # llm: TextChatAtOAI Obj
        # system_message: 'You are a helpful assistant.'
        # files: None
        # rag_cfg: None

        self.cfg = rag_cfg or {}
        # self.cfg: {}
        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        # self.max_ref_token: 4000
        self.parser_page_size: int = self.cfg.get('parser_page_size', DEFAULT_PARSER_PAGE_SIZE)
        # self.parser_page_size: 500
        self.rag_searchers = self.cfg.get('rag_searchers', DEFAULT_RAG_SEARCHERS)
        # self.rag_searchers: ['keyword_search', 'front_page_search']
        self.rag_keygen_strategy = self.cfg.get('rag_keygen_strategy', DEFAULT_RAG_KEYGEN_STRATEGY)
        # self.rag_keygen_strategy: 'SplitQueryThenGenKeyword'

        function_list = function_list or []
        # function_list: []
        super().__init__(function_list=[{
            'name': 'retrieval',
            'max_ref_token': self.max_ref_token,
            'parser_page_size': self.parser_page_size,
            'rag_searchers': self.rag_searchers,
        }, {
            'name': 'doc_parser',
            'max_ref_token': self.max_ref_token,
            'parser_page_size': self.parser_page_size,
        }] + function_list,
                         llm=llm,
                         system_message=system_message)

        self.system_files = files or []
        # self.system_files: []

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """This agent is responsible for processing the input files in the message.

         This method stores the files in the knowledge base, and retrievals the relevant parts
         based on the query and returning them.
         The currently supported file types include: .pdf, .docx, .pptx, .txt, .csv, .tsv, .xlsx, .xls and html.

         Args:
             messages: A list of messages.
             lang: Language.

        Yields:
            The message of retrieved documents.
        """
        # process files in messages

        # messages: [Message({'role': 'user', 'content': [{'text': '介绍图二'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]})]
        # lang: 'zh'
        # kwargs: {}

        rag_files = self.get_rag_files(messages)
        # rag_files: ['https://arxiv.org/pdf/1706.03762.pdf']

        if not rag_files:
            yield [Message(role=ASSISTANT, content='', name='memory')]
        else:
            query = ''
            # Only retrieval content according to the last user query if exists
            if messages and messages[-1].role == USER:
                query = extract_text_from_message(messages[-1], add_upload_info=False)
                # query: '介绍图二'

            # Keyword generation
            if query and self.rag_keygen_strategy.lower() != 'none':
                module_name = 'qwen_agent.agents.keygen_strategies'
                module = import_module(module_name)
                # module: <module 'qwen_agent.agents.keygen_strategies' from '/data0/csw/Qwen-Agent/qwen_agent/agents/keygen_strategies/__init__.py'>
                cls = getattr(module, self.rag_keygen_strategy)
                # cls: qwen_agent.agents.keygen_strategies.split_query_then_gen_keyword.SplitQueryThenGenKeyword
                keygen = cls(llm=self.llm)
                # keygen: SplitQueryThenGenKeyword object
                response = keygen.run([Message(USER, query)], files=rag_files)
                last = None
                for last in response:
                    continue
                if last:
                    # last: [Message({'role': 'assistant', 'content': '{"keywords_zh": ["图二", "图 2"], "keywords_en": ["Figure 2", "Diagram 2"], "text": "介绍图二"}'})]
                    keyword = last[-1].content.strip()
                    # keyword: '{"keywords_zh": ["图二", "图 2"], "keywords_en": ["Figure 2", "Diagram 2"], "text": "介绍图二"}'
                else:
                    keyword = ''

                if keyword.startswith('```json'):
                    keyword = keyword[len('```json'):]
                if keyword.endswith('```'):
                    keyword = keyword[:-3]
                try:
                    keyword_dict = json5.loads(keyword)
                    if 'text' not in keyword_dict:
                        keyword_dict['text'] = query
                    query = json.dumps(keyword_dict, ensure_ascii=False)
                    # query: '{"keywords_zh": ["图二", "图 2"], "keywords_en": ["Figure 2", "Diagram 2"], "text": "介绍图二"}'
                    logger.info(query)
                except Exception:
                    query = query

            content = self.function_map['retrieval'].call(
                {
                    'query': query,
                    # query: '{"keywords_zh": ["图二", "图 2"], "keywords_en": ["Figure 2", "Diagram 2"], "text": "介绍图二"}'
                    'files': rag_files
                    # rag_files: ['https://arxiv.org/pdf/1706.03762.pdf']
                },
                **kwargs,
            )
            # content: [{"url": 'https://arxiv.org/pdf/1706.03762.pdf', "text": [str_0, str_1, ..., str_8]}]
            if not isinstance(content, str):
                # 将 content 里的内容按 json 格式拼接成一个超长的字符串
                content = json.dumps(content, ensure_ascii=False, indent=4)

            yield [Message(role=ASSISTANT, content=content, name='memory')]

    def get_rag_files(self, messages: List[Message]):
        session_files = extract_files_from_messages(messages, include_images=False)
        files = self.system_files + session_files
        rag_files = []
        for file in files:
            f_type = get_file_type(file)
            if f_type in PARSER_SUPPORTED_FILE_TYPES and file not in rag_files:
                rag_files.append(file)
        return rag_files
