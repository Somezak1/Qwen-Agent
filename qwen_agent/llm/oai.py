import copy
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, Message
from qwen_agent.log import logger


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        # cfg: {'model': 'qwen2.5-14b-instruct', 'model_server': 'http://172.30.11.1:8020/v1', 'api_key': 'EMPTY'}
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            # this way
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        # messages: [
        #     Message({'role': 'system', 'content':
        #         'You are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)'
        #     }),
        #     Message({'role': 'user', 'content': "What's the weather like in San Francisco?"})
        # ]
        # delta_stream: False
        # generate_cfg: {'stop': ['✿RESULT✿', '✿RETURN✿'], 'seed': 736941439}

        messages = self.convert_messages_to_dicts(messages)
        # messages: [
        #     {'role': 'system', 'content': 'You are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)'},
        #     {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # ]

        print("*" * 60 + " Model Input " + "*" * 60)
        for i in messages:
            print(i)
        print("*" * 60 + " Model Input " + "*" * 60)

        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        yield [Message(ASSISTANT, chunk.choices[0].delta.content)]
            else:
                full_response = ''
                for idx, chunk in enumerate(response, start=1):
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        print(f"\n模型第 {idx} 次生成内容: {repr(full_response)}")
                        yield [Message(ASSISTANT, full_response)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            return [Message(ASSISTANT, response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    @staticmethod
    def convert_messages_to_dicts(messages: List[Message]) -> List[dict]:
        messages = [msg.model_dump() for msg in messages]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input:\n{pformat(messages, indent=2)}')
        return messages
