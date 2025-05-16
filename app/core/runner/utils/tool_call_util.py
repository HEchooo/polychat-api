"""
tool calls 常用转换方法
从 llm 接口获取的 tool calls 为 ChatCompletionMessageToolCall 形式，类型为 function
可通过 json() 获取 json 形式，格式如下：
{
    "id": "tool_call_0",
    "function": {
        "name": "file_search",
        "arguments": "{\"file_keys\": [\"file_0\", \"file_1\"], \"query\": \"query\"}"
    }
}
查询结果将放入 ["function"]["output"] 中
"""

import logging
from typing import List
import json
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall

from app.core.tools.base_tool import BaseTool
from app.core.tools.external_function_tool import ExternalFunctionTool
from config.llm import tool_settings


def tool_call_recognize(tool_call: ChatCompletionMessageToolCall, tools: List[BaseTool]) -> (BaseTool, dict):
    """
    对齐 tool call 和 tool，仅针对内部 tool call
    """
    tool_name = tool_call.function.name
    [tool] = [tool for tool in tools if tool.name == tool_name]
    if isinstance(tool, ExternalFunctionTool):
        tool = None
    return (tool, json.loads(tool_call.json()))


def safe_json_load(arguments_str: str) -> dict:
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        import re
        match = re.match(r'^\s*(\{.*?\})', arguments_str, re.DOTALL)
        if match:
            return json.loads(match.group(1)) 
        raise 

def internal_tool_call_invoke(tool: BaseTool, tool_call_dict: dict) -> dict:
    args = safe_json_load(tool_call_dict["function"]["arguments"])
    tool_name = tool_call_dict["function"].get("name")
    logging.info("tool_call_dict function %s", tool_call_dict["function"])
    if tool_name in tool_settings.SPECIAL_STREAM_TOOLS:
        output_stream = tool.run(**args)
        tool_call_dict["function"]["_stream"] = output_stream
        tool_call_dict["function"]["output"] = "[streamed]"
    else:
        output = tool.run(**args)
        tool_call_dict["function"]["output"] = json.dumps(output, ensure_ascii=False)

    return tool_call_dict

def tool_call_request(tool_call_dict: dict) -> dict:
    """
    tool call 结果需返回原始请求 & 结果
    库中未存储 tool_call 原始请求，需进行重新组装
    """
    return {
        "id": tool_call_dict["id"],
        "type": "function",
        "function": {"name": tool_call_dict["function"]["name"], "arguments": tool_call_dict["function"]["arguments"]},
    }


def tool_call_id(tool_call_dict: dict) -> str:
    """
    tool call id
    """
    return tool_call_dict["id"]


def tool_call_output(tool_call_dict: dict) -> str:
    """
    tool call output
    """
    return tool_call_dict["function"]["output"]
