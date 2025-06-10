from functools import partial
import logging
import traceback
import json
from typing import List, Iterator
from concurrent.futures import Executor

from sqlalchemy.orm import Session

from app.models.token_relation import RelationType
from config.config import settings
from config.llm import llm_settings, tool_settings

from app.core.runner.llm_backend import LLMBackend
from app.core.runner.llm_callback_handler import LLMCallbackHandler
from app.core.runner.memory import Memory, find_memory
from app.core.runner.pub_handler import StreamEventHandler
from app.core.runner.utils import message_util as msg_util
from app.core.runner.utils.tool_call_util import (
    tool_call_recognize,
    internal_tool_call_invoke,
    tool_call_request,
    tool_call_id,
    tool_call_output,
)
from app.core.runner.utils.system_prompt_util import SystemPromptUtils
from app.core.tools import find_tools, BaseTool
from app.libs.thread_executor import get_executor_for_config, run_with_executor
from app.models.message import Message, MessageUpdate
from app.models.run import Run
from app.models.run_step import RunStep
from app.models.token_relation import RelationType
from app.services.assistant.assistant import AssistantService
from app.services.file.file import FileService
from app.services.message.message import MessageService
from app.services.run.run import RunService
from app.services.run.run_step import RunStepService
from app.services.token.token import TokenService
from app.services.token.token_relation import TokenRelationService
from app.exceptions.exception import InterpreterNotSupported
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from app.core.runner.feishu_alert import feishu_notifier



class ThreadRunner:
    """
    ThreadRunner 封装 run 的执行逻辑
    """

    tool_executor: Executor = get_executor_for_config(tool_settings.TOOL_WORKER_NUM, "tool_worker_")

    def __init__(self, run_id: str, session: Session, stream: bool = False):
        self.run_id = run_id
        self.session = session
        self.stream = stream
        self.max_step = llm_settings.LLM_MAX_STEP
        self.event_handler: StreamEventHandler = None
        self.max_chat_history = 6

    def run(self):
        """
        完成一次 run 的执行，基本步骤
        1. 初始化，获取 run 以及相关 tools, 构造 system instructions;
        2. 开始循环，查询已有 run step, 进行 chat message 生成;
        3. 调用 llm 并解析返回结果;
        4. 根据返回结果，生成新的 run step(tool calls 处理) 或者 message
        """
        # TODO: 重构，将 run 的状态变更逻辑放到 RunService 中
        try:
            run = RunService.get_run_sync(session=self.session, run_id=self.run_id)
            self.event_handler = StreamEventHandler(run_id=self.run_id, is_stream=self.stream)

            run = RunService.to_in_progress(session=self.session, run_id=self.run_id)
            self.event_handler.pub_run_in_progress(run)
            logging.info("processing ThreadRunner task, run_id: %s", self.run_id)

            # get memory from assistant metadata
            # format likes {"memory": {"type": "window", "window_size": 20, "max_token_size": 4000}}
            ast = AssistantService.get_assistant_sync(session=self.session, assistant_id=run.assistant_id)
            metadata = ast.metadata_ or {}
            memory = find_memory(metadata.get("memory", {}))

            instructions = [run.instructions] if run.instructions else [ast.instructions]
            tools = find_tools(run, self.session)
            for tool in tools:
                tool.configure(session=self.session, run=run)
                instruction_supplement = tool.instruction_supplement()
                if instruction_supplement:
                    instructions += [instruction_supplement]
            instruction = "\n".join(instructions)

            llm = self.__init_llm_backend(run.assistant_id, run.model)
            loop = True
            while loop:
                run_steps = RunStepService.get_run_step_list(
                    session=self.session, run_id=self.run_id, thread_id=run.thread_id
                )
                loop = self.__run_step(llm, run, run_steps, instruction, tools, memory)
        except InterpreterNotSupported as e:
            logging.error(f'Do not support the OpenAI interpreter yet: {e}')
        except Exception as e:
            logging.error(f'Running failed with error: {e}')
            logging.error(traceback.format_exc())
        finally:
            # 任务结束
            self.event_handler.pub_run_completed(run)
            self.event_handler.pub_done()

    def __run_step(
        self,
        llm: LLMBackend,
        run: Run,
        run_steps: List[RunStep],
        instruction: str,
        tools: List[BaseTool],
        memory: Memory,
    ):
        logging.info("step %d is running", len(run_steps) + 1)

        chat_messages, system_message = self.__generate_chat_messages(
            MessageService.get_message_list(session=self.session, thread_id=run.thread_id)
        )

        if len(chat_messages) > 0 and len(run_steps) == 0:
            last_user_message = None
            for message in reversed(chat_messages):
                if message.get("role") == "user":
                    last_user_message = message
                    break
            
            if last_user_message:
                last_message_content = last_user_message.get("content")
                feishu_notifier.send_notify(self.run_id, last_message_content, run.assistant_id, run.thread_id)

        # get user info message
        system_user_message = None
        system_user_index = None
        extracted_system_text = None
        for idx, msg in enumerate(chat_messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.startswith("SYSTEM:"):
                system_user_message = msg
                system_user_index = idx
                extracted_system_text = content[len("SYSTEM:"):].strip()
                break
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text" and item.get("text", "").startswith("SYSTEM:"):
                        system_user_message = msg
                        system_user_index = idx
                        extracted_system_text = item["text"][len("SYSTEM:"):].strip()
                        break
                if extracted_system_text is not None:
                    break
        if system_user_index is not None:
            del chat_messages[system_user_index]
        extracted_system_text = f"{extracted_system_text};thread_id:{run.thread_id}"
        # logging.info("extracted_system_text: %s", extracted_system_text)

        # logging.info("chat_messages before processing: %s", chat_messages)

        cs_to_delete_indices = set()
        cs_deleted_user_messages = []  
        target_types = {"CS0001", "CS0002"}
        # target_types = {}
        last_index = len(chat_messages) - 1
        
        # Calculate the range of last max_chat_history messages
        max_history_start_idx = max(0, len(chat_messages) - self.max_chat_history) if len(chat_messages) > self.max_chat_history else 0

        for idx, msg in enumerate(chat_messages):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not any(content.startswith(prefix) for prefix in target_types):
                continue

            if idx != last_index:
                cs_to_delete_indices.add(idx)
                # Check if the user message before this CS message is in the last max_chat_history range
                if idx - 1 >= 0 and chat_messages[idx - 1].get("role") == "user":
                    user_msg_idx = idx - 1
                    if user_msg_idx >= max_history_start_idx:
                        # This user message is in the last max_chat_history range, preserve its content
                        user_content = chat_messages[user_msg_idx].get("content")
                        if isinstance(user_content, list):
                            extracted_texts = []
                            for item in user_content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_val = item.get("text", "")
                                    if isinstance(text_val, str):
                                        extracted_texts.append(text_val)
                            user_content = "；".join(filter(None, extracted_texts))
                        elif not isinstance(user_content, str):
                            user_content = str(user_content)
                        
                        if user_content and user_content.strip():
                            cs_deleted_user_messages.append(user_content)
                    
                    cs_to_delete_indices.add(user_msg_idx)

        chat_messages = [
            msg for idx, msg in enumerate(chat_messages) if idx not in cs_to_delete_indices
        ]
        chat_messages = [
            msg for msg in chat_messages
            if not (
                msg.get("role") == "assistant"
                and isinstance(msg.get("content"), str)
                and "CS0002" in msg.get("content")
            )
        ]

        tool_call_messages = [
            msg for step in run_steps if step.type == "tool_calls" and step.status == "completed"
            for msg in self.__convert_assistant_tool_calls_to_chat_messages(step)
        ]

        # logging.info("chat_messages before: %s", chat_messages)

        # Handle SR0001 processing with new logic
        sr_to_delete_indices = set()
        for idx, msg in enumerate(chat_messages):
            if (
                msg.get("role") == "assistant"
                and isinstance(msg.get("content"), str)
                and "SR0001" in msg["content"]
            ):
                is_last = (idx == len(chat_messages) - 1)
                
                # If not the last message, apply deletion logic
                if not is_last:
                    sr_to_delete_indices.add(idx)
                    
                    # Check if next message is SR0002 and delete it too
                    if idx + 1 < len(chat_messages):
                        next_msg = chat_messages[idx + 1]
                        next_content = next_msg.get("content", "")
                        if isinstance(next_content, str) and "SR0002" in next_content:
                            sr_to_delete_indices.add(idx + 1)
                    
                    # If previous message is user, add resolved tag
                    if idx > 0 and chat_messages[idx - 1].get("role") == "user":
                        prev_msg = chat_messages[idx - 1]
                        prev_content = prev_msg.get("content", "")
                        
                        if isinstance(prev_content, str):
                            # Add resolved tag to string content
                            prev_msg["content"] = prev_content + "（已解决，请勿重复回答这个问题）"
                        elif isinstance(prev_content, list):
                            # Add resolved tag to list content
                            for item in prev_content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    original_text = item.get("text", "")
                                    item["text"] = original_text + "（已解决，请勿重复回答这个问题）"
                                    break
                            else:
                                # If no text item found, add new text item
                                prev_content.append({
                                    "type": "text",
                                    "text": "（已解决，请勿重复回答这个问题）"
                                })

        # Handle RC0001 processing with original logic
        rc_to_delete_indices = set()
        for idx, msg in enumerate(chat_messages):
            if (
                msg.get("role") == "assistant"
                and isinstance(msg.get("content"), str)
                and "RC0001" in msg["content"]
            ):
                is_last = (idx == len(chat_messages) - 1)
                should_delete = False

                if not is_last:
                    next_msg = chat_messages[idx + 1]
                    next_content = next_msg.get("content", "")
                    # logging.info("Found RC0001, idx: %s, next_content: %s", idx, next_content)

                    if isinstance(next_content, str):
                        # logging.info("next_content: %s", next_content)
                        if "RC0002" not in next_content:
                            should_delete = True
                    elif isinstance(next_content, list):
                        next_str = json.dumps(next_content)
                        if "RC0002" not in next_str:
                            should_delete = True
                else:
                    if len(tool_call_messages) == 0:
                        should_delete = True

                if should_delete:
                    rc_to_delete_indices.add(idx)
                    if idx > 0 and chat_messages[idx - 1].get("role") == "user":
                        rc_to_delete_indices.add(idx - 1)

        # Combine all indices to delete
        all_to_delete_indices = sr_to_delete_indices | rc_to_delete_indices
        logging.info("sr_to_delete_indices: %s", sr_to_delete_indices)
        logging.info("rc_to_delete_indices: %s", rc_to_delete_indices)
        logging.info("all_to_delete_indices: %s", all_to_delete_indices)
        
        chat_messages = [
            msg for idx, msg in enumerate(chat_messages) if idx not in all_to_delete_indices
        ]
        # logging.info("chat_messages after SR/RC processing: %s", chat_messages)

        if tool_settings.FILTER_TAGS:
            filter_to_delete_indices = set()
            for idx, msg in enumerate(chat_messages):
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                    content = msg.get("content")
                    for tag in tool_settings.FILTER_TAGS:
                        if tag in content:
                            filter_to_delete_indices.add(idx)
                            if idx > 0 and chat_messages[idx - 1].get("role") == "user":
                                filter_to_delete_indices.add(idx - 1)
                            break
            
            # logging.info("filter_to_delete_indices: %s", filter_to_delete_indices)
            # logging.info("filtered tags: %s", tool_settings.FILTER_TAGS)
            chat_messages = [
                msg for idx, msg in enumerate(chat_messages) if idx not in filter_to_delete_indices
            ]
            # logging.info("chat_messages after tag filtering: %s", chat_messages)
        assistant_system_message = []
        if system_message:
            if system_message.get("content") and system_message.get("content").strip():
                assistant_system_message.append(system_message)
        instruction = extracted_system_text + "\n" + instruction
        assistant_system_message.append(msg_util.system_message(instruction))

        # clean the messages, remove consecutive user messages
        # cleaned_chat_messages = []
        # previous_role = None

        # for msg in chat_messages:
        #     current_role = msg.get("role")
        #     if current_role == "user" and previous_role == "user":
        #         cleaned_chat_messages[-1] = msg  
        #     else:
        #         cleaned_chat_messages.append(msg)
        #     previous_role = current_role

        # chat_messages = cleaned_chat_messages


        preserved_messages = []
        user_history_summary = ""

        consecutive_user_msgs = []
        if isinstance(chat_messages, list) and len(chat_messages) > 0:
            for i in range(len(chat_messages) - 2, -1, -1):
                if chat_messages[i].get("role") == "user":
                    content = chat_messages[i].get("content")
                    if isinstance(content, list):
                        extracted_texts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_val = item.get("text", "")
                                if isinstance(text_val, str):
                                    extracted_texts.append(text_val)
                        content = "；".join(filter(None, extracted_texts))
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    if content and content.strip():
                        consecutive_user_msgs.append(content)
                else:
                    break

        consecutive_user_msgs.reverse()

        if isinstance(chat_messages, list) and len(chat_messages) > self.max_chat_history:
            start_idx = len(chat_messages) - self.max_chat_history
            for i in range(start_idx - 1, -1, -1):
                if chat_messages[i].get("role") == "user":
                    preserved_messages = chat_messages[i:]  
                    truncated_messages = chat_messages[:i] 
                    break
            else:
                preserved_messages = chat_messages[-self.max_chat_history:]
                truncated_messages = chat_messages[:-self.max_chat_history]

            # logging.info("chat messages is too long, truncate to ensure start with user before last %d messages", self.max_chat_history)

            user_msgs = []
            for msg in truncated_messages:
                if msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, list):
                        extracted_texts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_val = item.get("text", "")
                                if isinstance(text_val, str):
                                    extracted_texts.append(text_val)
                        content = "；".join(filter(None, extracted_texts))
                    elif not isinstance(content, str):
                        content = str(content)
                    user_msgs.append(content)
            
            user_msgs.extend(cs_deleted_user_messages)
            user_msgs.extend(consecutive_user_msgs)
            
            if user_msgs:
                # logging.info("user_msgs: %s", user_msgs)
                last_5_user_msgs = user_msgs[-5:]
                user_history_summary = "；".join(last_5_user_msgs)
        else:
            preserved_messages = chat_messages
            all_user_history = cs_deleted_user_messages + consecutive_user_msgs
            if all_user_history:
                user_history_summary = "；".join(all_user_history)
                
        new_user_message = msg_util.user_message( f"""用户最近还曾问过：{user_history_summary}""")
        # logging.info("new_user_message: %s", new_user_message)
        # logging.info("preserved_messages: %s", preserved_messages)
        # logging.info("cs_deleted_user_messages: %s", cs_deleted_user_messages)
        preserved_messages.insert(0, new_user_message)
        messages = assistant_system_message + memory.integrate_context(preserved_messages) + tool_call_messages

        response_stream = llm.run(
            messages=messages,
            model=run.model,
            tools=[tool.openai_function for tool in tools],
            tool_choice="auto" if len(run_steps) < self.max_step else "none",
            stream=True,
            stream_options=run.stream_options,
            extra_body=run.extra_body,
            temperature=run.temperature,
            top_p=run.top_p,
            response_format=run.response_format,
        )

        # create message callback
        create_message_callback = partial(
            MessageService.new_message,
            session=self.session,
            assistant_id=run.assistant_id,
            thread_id=run.thread_id,
            run_id=run.id,
            role="assistant",
        )

        # create 'message creation' run step callback
        def _create_message_creation_run_step(message_id):
            return RunStepService.new_run_step(
                session=self.session,
                type="message_creation",
                assistant_id=run.assistant_id,
                thread_id=run.thread_id,
                run_id=run.id,
                step_details={"type": "message_creation", "message_creation": {"message_id": message_id}},
            )

        llm_callback_handler = LLMCallbackHandler(
            run_id=run.id,
            on_step_create_func=_create_message_creation_run_step,
            on_message_create_func=create_message_callback,
            event_handler=self.event_handler,
            assistant_id=run.assistant_id,
        )
        response_msg = llm_callback_handler.handle_llm_response(response_stream)
        message_creation_run_step = llm_callback_handler.step

        if msg_util.is_tool_call(response_msg):

            logging.info("response_msg: %s", response_msg)
            
            if response_msg.content and response_msg.content.strip():
                logging.info("Tool call has message content, saving to database: %s", response_msg.content)
                initial_message = MessageService.modify_message_sync(
                    session=self.session,
                    thread_id=run.thread_id,
                    message_id=llm_callback_handler.message.id,
                    body=MessageUpdate(content=response_msg.content),
                )
            
            # tool & tool_call definition dict
            tool_calls = [tool_call_recognize(tool_call, tools) for tool_call in response_msg.tool_calls]

            # new run step for tool calls
            new_run_step = RunStepService.new_run_step(
                session=self.session,
                type="tool_calls",
                assistant_id=run.assistant_id,
                thread_id=run.thread_id,
                run_id=run.id,
                step_details={"type": "tool_calls", "tool_calls": [tool_call_dict for _, tool_call_dict in tool_calls]},
            )
            self.event_handler.pub_run_step_created(new_run_step)
            self.event_handler.pub_run_step_in_progress(new_run_step)

            internal_tool_calls = [(tool, call) for tool, call in tool_calls if tool is not None]
            external_tool_call_dict = [call for tool, call in tool_calls if tool is None]

            # 为减少线程同步逻辑，依次处理内/外 tool_call 调用
            if internal_tool_calls:
                try:
                    tool_outputs = run_with_executor(
                        executor=self.tool_executor,
                        func=internal_tool_call_invoke,
                        tasks=internal_tool_calls,
                        timeout=tool_settings.TOOL_WORKER_EXECUTION_TIMEOUT,
                    )

                    tool_name = tool_outputs[0]["function"].get("name")
                    is_special_stream_tool = tool_name in tool_settings.SPECIAL_STREAM_TOOLS
                    is_special_normal_tool = tool_name in tool_settings.SPECIAL_NORMAL_TOOLS

                    if is_special_stream_tool:
                        logging.info("Special stream tool detected: %s", tool_outputs[0]["function"])
                        tool_stream = tool_outputs[0]["function"]["_stream"]

                        def wrap_stream(tool_chunk_iter):
                            buffer = "" 
                            prefix_checked = False
                            matched_prefix = None

                            for chunk in tool_chunk_iter:
                                if chunk.strip():
                                    if not prefix_checked:
                                        buffer += chunk
                                        if len(buffer) >= 6:
                                            prefix = buffer[:6]
                                            matched_prefix = prefix
                                            prefix_checked = True

                                            if matched_prefix == "RC0002":
                                                # Skip the RC0002 prefix and output the remaining content
                                                remaining = buffer[6:]
                                                logging.info("remaining %s", response_msg.content)
                                                if response_msg.content and "RC0001" in response_msg.content:
                                                    if remaining:
                                                        yield ChatCompletionChunk(
                                                            id="chatcmpl",
                                                            object="chat.completion.chunk",
                                                            created=0,
                                                            model="model",
                                                            choices=[
                                                                Choice(
                                                                    index=0,
                                                                    delta=ChoiceDelta(content=remaining, role="assistant"),
                                                                    finish_reason=None,
                                                                )
                                                            ],
                                                        )
                                                else: 
                                                    if remaining:
                                                        prefixed_remaining = '{ "msgs": [' + remaining
                                                        yield ChatCompletionChunk(
                                                            id="chatcmpl",
                                                            object="chat.completion.chunk",
                                                            created=0,
                                                            model="model",
                                                            choices=[
                                                                Choice(
                                                                    index=0,
                                                                    delta=ChoiceDelta(content=prefixed_remaining, role="assistant"),
                                                                    finish_reason=None,
                                                                )
                                                            ],
                                                        )
                                            else:
                                                # Output the entire buffer content
                                                yield ChatCompletionChunk(
                                                    id="chatcmpl",
                                                    object="chat.completion.chunk",
                                                    created=0,
                                                    model="model",
                                                    choices=[
                                                        Choice(
                                                            index=0,
                                                            delta=ChoiceDelta(content=buffer, role="assistant"),
                                                            finish_reason=None,
                                                        )
                                                    ],
                                                )
                                            buffer = ""
                                        continue

                                    # After prefix is checked, output chunks as received
                                    yield ChatCompletionChunk(
                                        id="chatcmpl",
                                        object="chat.completion.chunk",
                                        created=0,
                                        model="model",
                                        choices=[
                                            Choice(
                                                index=0,
                                                delta=ChoiceDelta(content=chunk, role="assistant"),
                                                finish_reason=None,
                                            )
                                        ],
                                    )

                            if matched_prefix == "RC0002":
                                # Add closing characters for RC0002 format
                                yield ChatCompletionChunk(
                                    id="chatcmpl",
                                    object="chat.completion.chunk",
                                    created=0,
                                    model="model",
                                    choices=[
                                        Choice(
                                            index=0,
                                            delta=ChoiceDelta(content="\t] \t }", role="assistant"),
                                            finish_reason=None,
                                        )
                                    ],
                                )

                            yield ChatCompletionChunk(
                                id="chatcmpl",
                                object="chat.completion.chunk",
                                created=0,
                                model="model",
                                choices=[
                                    Choice(index=0, delta=ChoiceDelta(content=None), finish_reason="stop")
                                ],
                            )

                        response_msg = llm_callback_handler.handle_llm_response(wrap_stream(tool_stream))

                        new_message = MessageService.modify_message_sync(
                            session=self.session,
                            thread_id=run.thread_id,
                            message_id=llm_callback_handler.message.id,
                            body=MessageUpdate(content=response_msg.content),
                        )
                        self.event_handler.pub_message_completed(new_message)

                        new_step = RunStepService.update_step_details(
                            session=self.session,
                            run_step_id=llm_callback_handler.step.id,  
                            step_details={"type": "message_creation", "message_creation": {"message_id": new_message.id}},
                            completed=True,
                        )
                        RunService.to_completed(session=self.session, run_id=run.id)
                        self.event_handler.pub_run_step_completed(new_step)
                        self.event_handler.pub_run_completed(run)

                        return False
                    elif is_special_normal_tool:
                        logging.info("Special tool output detected: %s", tool_outputs[0]["function"]["output"])
                        output_data = tool_outputs[0]["function"]["output"]
                        final_output = output_data

                        try:
                            parsed_output = json.loads(output_data)
                            if isinstance(parsed_output, dict) and "data" in parsed_output:
                                final_output = json.dumps(parsed_output["data"], ensure_ascii=False)
                        except Exception as e:
                            logging.warning("Failed to parse tool output json: %s", e)
                        response_msg = llm_callback_handler.handle_llm_response(fake_single_chunk_stream(final_output))

                        new_message = MessageService.modify_message_sync(
                            session=self.session,
                            thread_id=run.thread_id,
                            message_id=llm_callback_handler.message.id,
                            body=MessageUpdate(content=response_msg.content),
                        )
                        self.event_handler.pub_message_completed(new_message)

                        new_step = RunStepService.update_step_details(
                            session=self.session,
                            run_step_id=llm_callback_handler.step.id,
                            step_details={"type": "message_creation", "message_creation": {"message_id": new_message.id}},
                            completed=True,
                        )
                        RunService.to_completed(session=self.session, run_id=run.id)
                        self.event_handler.pub_run_step_completed(new_step)
                        self.event_handler.pub_run_completed(run)

                        return False
                    new_run_step = RunStepService.update_step_details(
                        session=self.session,
                        run_step_id=new_run_step.id,
                        step_details={"type": "tool_calls", "tool_calls": tool_outputs},
                        completed=not external_tool_call_dict,
                    )

                except Exception as e:
                    logging.error(f'Tool stream failed: {e}')
                    logging.error(traceback.format_exc())
                    RunStepService.to_failed(session=self.session, run_step_id=new_run_step.id, last_error=e)
                    raise e

            if external_tool_call_dict:
                run = RunService.to_requires_action(
                    session=self.session,
                    run_id=run.id,
                    required_action={
                        "type": "submit_tool_outputs",
                        "submit_tool_outputs": {"tool_calls": external_tool_call_dict},
                    },
                )
                self.event_handler.pub_run_step_delta(
                    step_id=new_run_step.id, step_details={"type": "tool_calls", "tool_calls": external_tool_call_dict}
                )
                self.event_handler.pub_run_requires_action(run)
            else:
                self.event_handler.pub_run_step_completed(new_run_step)
                return True
        logging.info("response_msg create new_message: %s", response_msg)
        new_message = MessageService.modify_message_sync(
            session=self.session,
            thread_id=run.thread_id,
            message_id=llm_callback_handler.message.id,
            body=MessageUpdate(content=response_msg.content),
        )
        self.event_handler.pub_message_completed(new_message)

        new_step = RunStepService.update_step_details(
            session=self.session,
            run_step_id=message_creation_run_step.id,
            step_details={"type": "message_creation", "message_creation": {"message_id": new_message.id}},
            completed=True,
        )
        RunService.to_completed(session=self.session, run_id=run.id)
        self.event_handler.pub_run_step_completed(new_step)

        return False

    def __init_llm_backend(self, assistant_id, model: str = None):
        if settings.AUTH_ENABLE:
            # init llm backend with token id
            token_id = TokenRelationService.get_token_id_by_relation(
                session=self.session, relation_type=RelationType.Assistant, relation_id=assistant_id
            )
            token = TokenService.get_token_by_id(self.session, token_id)
            return LLMBackend(base_url=token.llm_base_url, api_key=token.llm_api_key)
        else:
            # init llm backend with multi-provider settings based on model
            if model:
                provider_config = llm_settings.get_provider_config(model)
                return LLMBackend(base_url=provider_config["base_url"], api_key=provider_config["api_key"])
            else:
                # fallback to local provider
                return LLMBackend(base_url=llm_settings.LOCAL_BASE_URL, api_key=llm_settings.LOCAL_API_KEY)

    def __generate_chat_messages(self, messages: List[Message]):
        """
        根据历史信息生成 chat message
        """

        chat_messages = []
        system_message = None
        for message in messages:
            role = message.role
            if role == "user":
                message_content = []
                if message.file_ids:
                    files = FileService.get_file_list_by_ids(session=self.session, file_ids=message.file_ids)
                    for file in files:
                        chat_messages.append(msg_util.new_message(role, f'The file "{file.filename}" can be used as a reference'))
                else:
                    for content in message.content:
                        if content["type"] == "text":
                            message_content.append({"type": "text", "text": content["text"]["value"]})
                        elif content["type"] == "image_url":
                            message_content.append(content)
                    chat_messages.append(msg_util.new_message(role, message_content))
            elif role == "assistant":
                message_content = ""
                for content in message.content:
                    if content["type"] == "text":
                        message_content += content["text"]["value"]
                chat_messages.append(msg_util.new_message(role, message_content))
            elif role == "system":
                message_content = ""
                for content in message.content:
                    if content["type"] == "text":
                        message_content += content["text"]["value"]
                system_message = msg_util.system_message(message_content)

        return chat_messages, system_message

    def __convert_assistant_tool_calls_to_chat_messages(self, run_step: RunStep):
        """
        根据 run step 执行结果生成 message 信息
        每个 tool call run step 包含两部分，调用与结果(结果可能为多个信息)
        """
        tool_calls = run_step.step_details["tool_calls"]
        tool_call_requests = [msg_util.tool_calls([tool_call_request(tool_call) for tool_call in tool_calls])]
        tool_call_outputs = [
            msg_util.tool_call_result(tool_call_id(tool_call), tool_call_output(tool_call)) for tool_call in tool_calls
        ]
        return tool_call_requests + tool_call_outputs

def fake_single_chunk_stream(text: str) -> Iterator[ChatCompletionChunk]:
    yield ChatCompletionChunk(
        id="chatcmpl",
        object="chat.completion.chunk",
        created=0,
        model="model",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=text, role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield ChatCompletionChunk(
        id="chatcmpl",
        object="chat.completion.chunk",
        created=0,
        model="model",
        choices=[
            Choice(index=0, delta=ChoiceDelta(content=None), finish_reason="stop")
        ],
    )
