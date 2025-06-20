import logging
import re
from openai import Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from app.core.runner.pub_handler import StreamEventHandler
from app.core.runner.utils import message_util
from config.llm import tool_settings
from app.core.runner.feishu_alert import feishu_notifier

class LLMCallbackHandler:
    """
    LLM chat callback handler, handling message sending and message merging
    """

    def __init__(
        self, run_id: str, on_step_create_func, on_message_create_func, event_handler: StreamEventHandler, assistant_id: str, thread_id: str
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.final_message_started = False
        self.on_step_create_func = on_step_create_func
        self.step = None
        self.on_message_create_func = on_message_create_func
        self.message = None
        self.event_handler: StreamEventHandler = event_handler
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self._buffer_prefix = ""
        self._matched_prefix = None
        self.remain_text = ""
        self._phase_b_started = False
        self._two_phase_detected = False

        self._phase_a_prefixes = {"RC0001", "CS0001", "SR0001", "OD0001"}
        self._phase_a_buffered_prefixes = {"CS0001", "SR0001", "OD0001"} 
        self._phase_b_prefixes = {
            "RC0001": "RC0002",
            "CS0001": "CS0002",
            "SR0001": "SR0002",
            "OD0001": "OD0002",
        }
        base_special_tags = {"AD0001", "NC0001", "UP0001"}
        filter_tags = set(tool_settings.FILTER_TAGS) if tool_settings.FILTER_TAGS else set()
        self._special_tail_only_prefixes = base_special_tags | filter_tags

    def _process_prefix_buffer(self, incoming_text: str) -> str:
        self._buffer_prefix += incoming_text
        scan_window = self._buffer_prefix[:30]

        match = re.search(r'([A-Z]{2}\d{4})', scan_window)
        if match:
            self._matched_prefix = match.group(1)
            cut_index = match.end()
            suffix = self._buffer_prefix[cut_index:]

            if self._matched_prefix in self._phase_a_prefixes or self._matched_prefix in self._special_tail_only_prefixes:
                result = '{ "msgs": [' + suffix
            elif self._matched_prefix in self._phase_b_prefixes.values():
                result = suffix
            else:
                self._matched_prefix = None
                result = self._buffer_prefix

            self._buffer_prefix = ""
            return result

        if len(self._buffer_prefix) > 30:
            logging.warning("Prefix not found, forcing passthrough")
            result = self._buffer_prefix
            self._buffer_prefix = ""
            return result

        return ""

    def handle_llm_response(
        self,
        response_stream: Stream[ChatCompletionChunk],
    ) -> ChatCompletionMessage:
        message = ChatCompletionMessage(content="", role="assistant", tool_calls=[])
        index = 0
        raw_content = ""
        final_content = ""
        self.final_message_started = False

        try:
            last_chunk = None
            for chunk in response_stream:
                last_chunk = chunk
                # logging.debug("llm response chunk: %s", chunk)

                if chunk.usage:
                    self.event_handler.pub_message_usage(chunk)
                    # continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if not delta:
                    continue

                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        message_util.merge_tool_call_delta(message.tool_calls, tool_call_delta)

                elif delta.content:
                    if not self.final_message_started:
                        self.final_message_started = True
                        self.message = self.on_message_create_func(content="")
                        self.step = self.on_step_create_func(self.message.id)
                        logging.debug("create message and step (%s), (%s)", self.message, self.step)

                        self.event_handler.pub_run_step_created(self.step)
                        self.event_handler.pub_run_step_in_progress(self.step)
                        self.event_handler.pub_message_created(self.message)
                        self.event_handler.pub_message_in_progress(self.message)

                    raw_content += delta.content

                    if self._matched_prefix in self._phase_a_buffered_prefixes:            
                        self.remain_text += delta.content

                        if self._phase_b_started:
                            self._two_phase_detected = True
                            continue

                        expected_b = self._phase_b_prefixes.get(self._matched_prefix)
                        match = re.search(r'[A-Z]{2}\d{4}', self.remain_text)
                        if match and match.group() == expected_b:
                            self._phase_b_started = True
                            self._two_phase_detected = True
                            logging.info(f"Detected {expected_b} after {self._matched_prefix}, start buffering")
                            continue
                        elif match:
                            delta_to_send = self.remain_text
                            self.remain_text = ""
                        else:
                            continue

                    elif self._phase_b_started:
                        self.remain_text += delta.content
                        continue

                    else:
                        if self._matched_prefix is None:
                            processed = self._process_prefix_buffer(delta.content)
                            if not processed:
                                continue
                            delta_to_send = processed
                        else:
                            delta_to_send = delta.content

                    if delta_to_send:
                        # TODO: Fix json format issue, clean comma before '}' before sending
                        if '}' in delta_to_send:
                            delta_to_send = re.sub(r',(\s*\n\s*)?}', r'}', delta_to_send)
                        
                        final_content += delta_to_send
                        self.event_handler.pub_message_delta(self.message.id, index, delta_to_send, delta.role)
                        index += 1

        except Exception as e:
            logging.error("handle_llm_response error: %s", e)
            raise e

        if last_chunk and last_chunk.usage:
            usage = last_chunk.usage
            logging.info("prompt_tokens: %s, completion_tokens: %s, total_tokens: %s",
                         usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        else:
            logging.info("No usage information in the last chunk: %s", last_chunk)
        
        if self._buffer_prefix:
            # TODO: Fix json format issue, clean comma before '}' before sending
            buffer_to_send = self._buffer_prefix
            if '}' in buffer_to_send:
                buffer_to_send = re.sub(r',(\s*\n\s*)?}', r'}', buffer_to_send)
            
            final_content += buffer_to_send
            self.event_handler.pub_message_delta(self.message.id, index, buffer_to_send, "assistant")
            index += 1
            self._buffer_prefix = ""

        logging.info("llm response message _matched_prefix: %s", self._matched_prefix)

        if self.remain_text:
            if self._two_phase_detected:
                delta_to_emit = re.sub(r'[A-Z]{2}\d{4}', ",", self.remain_text, count=1)
                logging.info("Two-phase detected, emit with patch")
            else:
                delta_to_emit = self.remain_text
                logging.info("Single phase, emit original remain_text")
            
            # TODO: Fix json format issue, clean comma before '}' before sending
            if '}' in delta_to_emit:
                delta_to_emit = re.sub(r',(\s*\n\s*)?}', r'}', delta_to_emit)
            
            final_content += delta_to_emit
            self.event_handler.pub_message_delta(self.message.id, index, delta_to_emit, "assistant")
            index += 1
            self.remain_text = ""

        # === 拼尾逻辑 ===
        tail = ""
        if self._matched_prefix in self._phase_a_prefixes:
            logging.info("llm response message _matched_prefix with %s", self._matched_prefix)
            trimmed = re.sub(r'[\s\\]+', '', final_content)
            if trimmed.endswith(",") or trimmed.endswith("}]}"):
                tail = ""
            else:
                tail = ","
            if self._two_phase_detected:
                if not final_content.strip().endswith("]}"):
                    tail = "]}"
                else:
                    tail = ""

        elif self._matched_prefix in self._phase_b_prefixes.values() or self._matched_prefix in self._special_tail_only_prefixes:
            trimmed = re.sub(r'[\s\\]+', '', final_content)
            tail = "" if trimmed.endswith("]}") else "]}"

        else:
            logging.info("llm response message: %s", raw_content)
            logging.info("llm response format message: %s", final_content)
            message.content = raw_content
            return message

        final_content += tail
        if tail:
            # TODO: Fix json format issue, clean comma before '}' before sending
            tail_to_send = tail
            if '}' in tail_to_send:
                tail_to_send = re.sub(r',(\s*\n\s*)?}', r'}', tail_to_send)
            self.event_handler.pub_message_delta(self.message.id, index, tail_to_send, "assistant")

        logging.info("llm response message: %s", raw_content)
        feishu_notifier.send_notify(self.run_id, f"{final_content}", self.assistant_id, self.thread_id)
        logging.info("llm response format message: %s", final_content)

        message.content = raw_content
        return message