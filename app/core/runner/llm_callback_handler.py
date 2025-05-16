import logging
import re
from openai import Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from app.core.runner.pub_handler import StreamEventHandler
from app.core.runner.utils import message_util


class LLMCallbackHandler:
    """
    LLM chat callback handler, handling message sending and message merging
    """

    def __init__(
        self, run_id: str, on_step_create_func, on_message_create_func, event_handler: StreamEventHandler
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.final_message_started = False
        self.on_step_create_func = on_step_create_func
        self.step = None
        self.on_message_create_func = on_message_create_func
        self.message = None
        self.event_handler: StreamEventHandler = event_handler
        self._buffer_prefix = ""
        self._matched_prefix = None  
        self.remain_text = ""  
        self._sr0002_started = False  

    def _process_prefix_buffer(self, incoming_text: str) -> str:
        self._buffer_prefix += incoming_text

        scan_window = self._buffer_prefix[:150]

        match = re.search(r'([A-Z]{2}\d{4})', scan_window)

        if match:
            self._matched_prefix = match.group(1)
            cut_index = match.end()
            suffix = self._buffer_prefix[cut_index:]

            if self._matched_prefix in ("RC0001", "CS0001", "SR0001", "AD0001", "NC0001", "OD0001"):
                result = '{ "msgs": [' + suffix
            elif self._matched_prefix in ("CS0002", "SR0002", "RC0002","OD0002"):
                result = suffix
            else:
                self._matched_prefix = None
                result = self._buffer_prefix

            self._buffer_prefix = ""
            return result

        if len(self._buffer_prefix) > 150:
            logging.warning("Prefix not found, forcing passthrough")
            result = self._buffer_prefix
            self._buffer_prefix = ""
            return result

        return ""

    def handle_llm_response(
        self,
        response_stream: Stream[ChatCompletionChunk],
    ) -> ChatCompletionMessage:
        """
        Handle LLM response stream
        :param response_stream: ChatCompletionChunk stream
        :return: ChatCompletionMessage
        """
        message = ChatCompletionMessage(content="", role="assistant", tool_calls=[])
        index = 0
        raw_content = ""
        final_content = ""

        try:
            for chunk in response_stream:
                # logging.debug(chunk)

                if chunk.usage:
                    self.event_handler.pub_message_usage(chunk)
                    continue

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

                    if self._matched_prefix == "SR0001":
                            self.remain_text += delta.content

                            if self._sr0002_started:
                                continue

                            match = re.search(r"S[A-Z]\d{4}", self.remain_text)
                            if match and match.group() == "SR0002":
                                self._sr0002_started = True
                                logging.info("Detected SR0002 after SR0001, starting to buffer SR0002 block")
                                continue  
                            elif match:
                                delta_to_send = self.remain_text
                                self.remain_text = ""
                            else:
                                continue
                    elif self._sr0002_started:
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
                    # final_content += delta_to_send
                    # self.event_handler.pub_message_delta(self.message.id, index, delta_to_send, delta.role)
                    # index += 1
                    if delta_to_send:
                        final_content += delta_to_send
                        self.event_handler.pub_message_delta(self.message.id, index, delta_to_send, delta.role)
                        index += 1

        except Exception as e:
            logging.error("handle_llm_response error: %s", e)
            raise e

        if self._buffer_prefix:
            final_content += self._buffer_prefix
            self.event_handler.pub_message_delta(self.message.id, index, self._buffer_prefix, "assistant")
            index += 1
            self._buffer_prefix = ""
        logging.info("llm response message _matched_prefix: %s", self._matched_prefix)

        if self.remain_text:
            if self._sr0002_started:
                logging.info("llm response message SR0001 started, remain_text: %s", self.remain_text)
                delta_to_emit = re.sub(r"SR0002", ",", self.remain_text)
            else:
                logging.info("llm response message only SR0001 started, remain_text: %s", self.remain_text)
                delta_to_emit = self.remain_text

            final_content += delta_to_emit
            self.event_handler.pub_message_delta(self.message.id, index, delta_to_emit, "assistant")
            index += 1
            self.remain_text = ""
        if self._matched_prefix in ("RC0001", "CS0001", "SR0001", "OD0001"):
            logging.info("llm response message _matched_prefix with %s", self._matched_prefix)
            trimmed = re.sub(r'[\s\\]+', '', final_content)
            if trimmed.endswith(",") or trimmed.endswith("}]}"):
                logging.info("llm response message ends with ','")
                tail = ""
            else:
                tail = ","
            if self._sr0002_started:
                logging.info("llm response message _matched_prefix with SR0002 started final_content: %s", final_content)
                if not final_content.strip().endswith("]}"):
                    tail = "]}"
                else:
                    tail = ""
        elif self._matched_prefix in ("CS0002", "SR0002", "AD0001", "NC0001","RC0002", "OD0002"):
            trimmed = re.sub(r'[\s\\]+', '', final_content)
            if trimmed.endswith("]}"):
                logging.info("llm response message ends with ']}'")
                tail = ""
            else:
                tail = "]}"
        else:
            logging.info("llm response message: %s", raw_content)
            logging.info("llm response format message: %s", final_content)
            message.content = raw_content
            return message

        final_content += tail
        self.event_handler.pub_message_delta(self.message.id, index, tail, "assistant")
        logging.info("llm response message: %s", raw_content)
        logging.info("llm response format message: %s", final_content)

        message.content = raw_content

        return message
