# Reference: https://platform.openai.com/docs/guides/function-calling
import json
import os

from qwen_agent.llm import get_chat_model


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})


def test(fncall_prompt_type: str = 'qwen'):
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        # 'model': 'qwen2.5-72b-instruct',
        # 'model_server': 'dashscope',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),
        # 'generate_cfg': {
        #     'fncall_prompt_type': fncall_prompt_type
        # },

        # Use the OpenAI-compatible model service provided by DashScope:
        # 'model': 'qwen2.5-72b-instruct',
        # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

        # Use the model service provided by Together.AI:
        # 'model': 'Qwen/qwen2.5-7b-instruct',
        # 'model_server': 'https://api.together.xyz',  # api_base
        # 'api_key': os.getenv('TOGETHER_API_KEY'),

        # Use your own model service compatible with OpenAI API:
        'model': 'qwen2.5-14b-instruct',
        'model_server': 'http://172.30.11.1:8020/v1',  # api_base
        'api_key': 'EMPTY',
        'generate_cfg': {
            'fncall_prompt_type': 'nous'
        },
    })
    # llm: TextChatAtOAI Obj

    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'user', 'content': "What's the weather like in San Francisco?"}]
    functions = [{
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA',
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                },
            },
            'required': ['location'],
        },
    }]
    tools = [
        {
            "type": "function",
            "function": {
                'name': 'get_current_weather',
                'description': 'Get the current weather in a given location',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'location': {
                            'type': 'string',
                            'description': 'The city and state, e.g. San Francisco, CA',
                        },
                        'unit': {
                            'type': 'string',
                            'enum': ['celsius', 'fahrenheit']
                        },
                    },
                    'required': ['location'],
                },
            },
        },
    ]

    print()
    print("*" * 60 + " Messages " + "*" * 60)
    for m in messages:
        print(m)
    print("*" * 60 + " Messages " + "*" * 60)

    print('\n# Assistant Response 1:')
    responses = []

    # llm.chat 调用流程:
    #     TextChatAtOAI.chat(...)
    #  -> BaseChatModel.chat(...)
    #  -> BaseFnCallModel._chat_with_functions(...)
    #  -> BaseFnCallModel._continue_assistant_response(...)
    #  -> BaseChatModel._chat(...)
    #  -> TextChatAtOAI._chat_stream(...)

    for responses in llm.chat(
            messages=messages,
            functions=tools,
            # functions=functions,
            stream=True,
            # Note: extra_generate_cfg is optional
            # extra_generate_cfg=dict(
            #     # Note: if function_choice='auto', let the model decide whether to call a function or not
            #     # function_choice='auto',  # 'auto' is the default if function_choice is not set
            #     # Note: set function_choice='get_current_weather' to force the model to call this function
            #     function_choice='get_current_weather',
            # ),
    ):
        print("llm.chat 返回内容: ", responses)

    # If you do not need streaming output, you can either use the following trick:
    #   *_, responses = llm.chat(messages=messages, functions=functions, stream=True)
    # or use stream=False:
    #   responses = llm.chat(messages=messages, functions=functions, stream=False)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    if last_response.get('function_call', None):

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            'get_current_weather': get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = last_response['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response['function_call']['arguments'])
        function_response = function_to_call(
            location=function_args.get('location'),
            unit=function_args.get('unit'),
        )
        print('\n# Function Response:')
        print(function_response)

        # Step 4: send the info for each function call and function response to the model
        messages.append({
            'role': 'function',
            'name': function_name,
            'content': function_response,
        })  # extend conversation with function response

        print()
        print("*" * 60 + " Messages " + "*" * 60)
        for m in messages:
            print(m)
        print("*" * 60 + " Messages " + "*" * 60)

        print('\n# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                functions=functions,
                stream=True,
        ):  # get a new response from the model where it can see the function response
            print("llm.chat 返回内容: ", responses)

        # ==================================================================================================================================
        # 使用 Qwen2 的模板进行 function call ↓
        #
        # 第一次调用 llm.chat(messages, functions, True) 所传入的 messages:
        # ************************************************************ Messages ************************************************************
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # ************************************************************ Messages ************************************************************
        #
        # OpenAI API 接口收到的 messages:
        # ************************************************************ Model Input ************************************************************
        # {'role': 'system', 'content': 'You are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)'}
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # ************************************************************ Model Input ************************************************************
        #
        # 模型实际接收到的 prompt:
        # '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)<|im_end|>\n<|im_start|>user\nWhat\'s the weather like in San Francisco?<|im_end|>\n<|im_start|>assistant\n'
        #
        # 模型第 3 次生成内容: '✿F'
        #
        # 模型第 4 次生成内容: '✿FU'
        #
        # 模型第 5 次生成内容: '✿FUN'
        #
        # 模型第 6 次生成内容: '✿FUNCTIO'
        #
        # 模型第 7 次生成内容: '✿FUNCTION✿: get_'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_', 'arguments': ''}}]
        #
        # 模型第 8 次生成内容: '✿FUNCTION✿: get_current_'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_', 'arguments': ''}}]
        #
        # 模型第 9 次生成内容: '✿FUNCTION✿: get_current_w'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_w', 'arguments': ''}}]
        #
        # 模型第 10 次生成内容: '✿FUNCTION✿: get_current_we'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_we', 'arguments': ''}}]
        #
        # 模型第 11 次生成内容: '✿FUNCTION✿: get_current_weathe'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weathe', 'arguments': ''}}]
        #
        # 模型第 12 次生成内容: '✿FUNCTION✿: get_current_weather'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 13 次生成内容: '✿FUNCTION✿: get_current_weather\n'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 14 次生成内容: '✿FUNCTION✿: get_current_weather\n✿AR'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 15 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"l'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"l'}}]
        #
        # 模型第 16 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"loc'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"loc'}}]
        #
        # 模型第 17 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"locat'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"locat'}}]
        #
        # 模型第 18 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location'}}]
        #
        # 模型第 19 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Fr'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Fr'}}]
        #
        # 模型第 20 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Fra'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Fra'}}]
        #
        # 模型第 21 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Franci'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Franci'}}]
        #
        # 模型第 22 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisc'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisc'}}]
        #
        # 模型第 23 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco,'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco,'}}]
        #
        # 模型第 24 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA"'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA"'}}]
        #
        # 模型第 25 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", '
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA",'}}]
        #
        # 模型第 26 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "u'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "u'}}]
        #
        # 模型第 27 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "un'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "un'}}]
        #
        # 模型第 28 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "unit": "fah'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fah'}}]
        #
        # 模型第 29 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "unit": "fahre'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahre'}}]
        #
        # 模型第 30 次生成内容: '✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "unit": "fahrenheit"}'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit"}'}}]
        #
        # # Function Response:
        # {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        #
        # 第二次调用 llm.chat(messages, functions, True) 所传入的 messages:
        # ************************************************************ Messages ************************************************************
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit"}'}}
        # {'role': 'function', 'name': 'get_current_weather', 'content': '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}'}
        # ************************************************************ Messages ************************************************************
        #
        # OpenAI API 接口收到的 messages:
        # ************************************************************ Model Input ************************************************************
        # {'role': 'system', 'content': 'You are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)'}
        # {'role': 'user', 'content': 'What\'s the weather like in San Francisco?\n\n✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "unit": "fahrenheit"}\n✿RESULT✿: {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}\n✿RETURN✿'}
        # ************************************************************ Model Input ************************************************************
        #
        # 模型实际接收到的 prompt:
        # '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n## You have access to the following tools:\n\n### get_current_weather\n\nget_current_weather: Get the current weather in a given location Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} Format the arguments as a JSON object.\n\n## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:\n\n✿FUNCTION✿: The tool to use, should be one of [get_current_weather]\n✿ARGS✿: The input of the tool\n✿RESULT✿: Tool results\n✿RETURN✿: Reply based on tool results. Images need to be rendered as ![](url)<|im_end|>\n<|im_start|>user\nWhat\'s the weather like in San Francisco?\n\n✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "San Francisco, CA", "unit": "fahrenheit"}\n✿RESULT✿: {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}\n✿RETURN✿<|im_end|>\n<|im_start|>assistant\n'
        #
        # 模型第 3 次生成内容: 'The '
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The'}]
        #
        # 模型第 4 次生成内容: 'The current temp'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temp'}]
        #
        # 模型第 5 次生成内容: 'The current tempera'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current tempera'}]
        #
        # 模型第 6 次生成内容: 'The current temperature'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature'}]
        #
        # 模型第 7 次生成内容: 'The current temperature in San Fr'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Fr'}]
        #
        # 模型第 8 次生成内容: 'The current temperature in San Franc'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Franc'}]
        #
        # 模型第 9 次生成内容: 'The current temperature in San Franci'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Franci'}]
        #
        # 模型第 10 次生成内容: 'The current temperature in San Francis'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Francis'}]
        #
        # 模型第 11 次生成内容: 'The current temperature in San Francisc'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Francisc'}]
        #
        # 模型第 12 次生成内容: 'The current temperature in San Francisco '
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Francisco'}]
        #
        # 模型第 13 次生成内容: 'The current temperature in San Francisco i'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Francisco i'}]
        #
        # 模型第 14 次生成内容: 'The current temperature in San Francisco is 72°F.'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current temperature in San Francisco is 72°F.'}]
        #
        # 使用 Qwen2 的模板进行 function call ↑
        # ==================================================================================================================================



        # ==================================================================================================================================
        # 使用 Qwen2.5 的模板进行 function call ↓
        #
        # 第一次调用 llm.chat(messages, tools, True) 所传入的 messages:
        # ************************************************************ Messages ************************************************************
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # ************************************************************ Messages ************************************************************
        #
        # OpenAI API 接口收到的 messages:
        # ************************************************************ Model Input ************************************************************
        # {'role': 'system', 'content': 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'}
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # ************************************************************ Model Input ************************************************************
        #
        # 模型实际接收到的 prompt:
        # '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the weather like in San Francisco?<|im_end|>\n<|im_start|>assistant\n'
        #
        # 模型第 2 次生成内容: '<tool_call>'
        #
        # 模型第 3 次生成内容: '<tool_call>\n'
        #
        # 模型第 4 次生成内容: '<tool_call>\n{"'
        #
        # 模型第 5 次生成内容: '<tool_call>\n{"name'
        #
        # 模型第 6 次生成内容: '<tool_call>\n{"name":'
        #
        # 模型第 7 次生成内容: '<tool_call>\n{"name": "'
        #
        # 模型第 8 次生成内容: '<tool_call>\n{"name": "get'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get', 'arguments': ''}}]
        #
        # 模型第 9 次生成内容: '<tool_call>\n{"name": "get_current'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current', 'arguments': ''}}]
        #
        # 模型第 10 次生成内容: '<tool_call>\n{"name": "get_current_weather'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 11 次生成内容: '<tool_call>\n{"name": "get_current_weather",'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather",', 'arguments': ''}}]
        #
        # 模型第 12 次生成内容: '<tool_call>\n{"name": "get_current_weather", "'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 13 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 14 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments":'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': ''}}]
        #
        # 模型第 15 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"'}}]
        #
        # 模型第 16 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location'}}]
        #
        # 模型第 17 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location":'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location":'}}]
        #
        # 模型第 18 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "'}}]
        #
        # 模型第 19 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San'}}]
        #
        # 模型第 20 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco'}}]
        #
        # 模型第 21 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco,'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco,'}}]
        #
        # 模型第 22 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA'}}]
        #
        # 模型第 23 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA",'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA",'}}]
        #
        # 模型第 24 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "'}}]
        #
        # 模型第 25 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit'}}]
        #
        # 模型第 26 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit":'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit":'}}]
        #
        # 模型第 27 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "'}}]
        #
        # 模型第 28 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "f'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "f'}}]
        #
        # 模型第 29 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit'}}]
        #
        # 模型第 30 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit"}}\n'}}]
        #
        # 模型第 31 次生成内容: '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit"}'}}]
        #
        # # Function Response:
        # {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        #
        # 第二次调用 llm.chat(messages, tools, True) 所传入的 messages:
        # ************************************************************ Messages ************************************************************
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "San Francisco, CA", "unit": "fahrenheit"}'}}
        # {'role': 'function', 'name': 'get_current_weather', 'content': '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}'}
        # ************************************************************ Messages ************************************************************
        #
        # OpenAI API 接口收到的 messages:
        # ************************************************************ Model Input ************************************************************
        # {'role': 'system', 'content': 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'}
        # {'role': 'user', 'content': "What's the weather like in San Francisco?"}
        # {'role': 'assistant', 'content': '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>'}
        # {'role': 'user', 'content': '<tool_response>\n{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}\n</tool_response>'}
        # ************************************************************ Model Input ************************************************************
        #
        # 模型实际接收到的 prompt:
        # '<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather in a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the weather like in San Francisco?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
        #
        # 模型第 2 次生成内容: 'The'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The'}]
        #
        # 模型第 3 次生成内容: 'The current'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current'}]
        #
        # 模型第 4 次生成内容: 'The current weather'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather'}]
        #
        # 模型第 5 次生成内容: 'The current weather in'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in'}]
        #
        # 模型第 6 次生成内容: 'The current weather in San'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San'}]
        #
        # 模型第 7 次生成内容: 'The current weather in San Francisco'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco'}]
        #
        # 模型第 8 次生成内容: 'The current weather in San Francisco is'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is'}]
        #
        # 模型第 9 次生成内容: 'The current weather in San Francisco is '
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is '}]
        #
        # 模型第 10 次生成内容: 'The current weather in San Francisco is 7'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is 7'}]
        #
        # 模型第 11 次生成内容: 'The current weather in San Francisco is 72'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is 72'}]
        #
        # 模型第 12 次生成内容: 'The current weather in San Francisco is 72 degrees'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is 72 degrees'}]
        #
        # 模型第 13 次生成内容: 'The current weather in San Francisco is 72 degrees Fahrenheit'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is 72 degrees Fahrenheit'}]
        #
        # 模型第 14 次生成内容: 'The current weather in San Francisco is 72 degrees Fahrenheit.'
        # llm.chat 返回内容:  [{'role': 'assistant', 'content': 'The current weather in San Francisco is 72 degrees Fahrenheit.'}]
        #
        # 使用 Qwen2.5 的模板进行 function call ↑
        # ==================================================================================================================================



if __name__ == '__main__':
    # Run example of function calling with QwenFnCallPrompt
    test()

    # Run example of function calling with NousFnCallPrompt
    # test(fncall_prompt_type='nous')
