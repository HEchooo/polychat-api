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

    def _process_prefix_buffer(self, incoming_text: str) -> str:
        self._buffer_prefix += incoming_text

        visible_chars = re.findall(r"[^\s]", self._buffer_prefix)
        if len(visible_chars) < 6:
            return ""

        prefix_chars = []
        prefix_indexes = []
        visible_count = 0
        for idx, c in enumerate(self._buffer_prefix):
            if not c.isspace():
                visible_count += 1
                prefix_chars.append(c)
                prefix_indexes.append(idx)
                if visible_count == 6:
                    break

        prefix = ''.join(prefix_chars)
        cut_index = prefix_indexes[-1] + 1  
        suffix = self._buffer_prefix[cut_index:]

        self._matched_prefix = prefix

        if prefix in ("RC0001", "CS0001", "SR0001", "AD0001", "NC0001"):
            result = self._buffer_prefix[:cut_index - 6] + '{ "msgs": [' + suffix
        elif prefix in ("CS0002", "SR0002"):
            result = suffix
        else:
            self._matched_prefix = None
            result = self._buffer_prefix

        self._buffer_prefix = ""
        return result
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

        try:
            for chunk in response_stream:
                logging.debug(chunk)

                if chunk.usage:
                    self.event_handler.pub_message_usage(chunk)
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if not delta:
                    continue

                # merge tool call delta
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        message_util.merge_tool_call_delta(message.tool_calls, tool_call_delta)

                elif delta.content:
                    # call on delta message received
                    if not self.final_message_started:
                        self.final_message_started = True

                        self.message = self.on_message_create_func(content="")
                        self.step = self.on_step_create_func(self.message.id)
                        logging.debug("create message and step (%s), (%s)", self.message, self.step)

                        self.event_handler.pub_run_step_created(self.step)
                        self.event_handler.pub_run_step_in_progress(self.step)
                        self.event_handler.pub_message_created(self.message)
                        self.event_handler.pub_message_in_progress(self.message)

                    if len(message.content) < 6:
                        processed = self._process_prefix_buffer(delta.content)
                        if not processed:
                            continue  
                        delta_to_send = processed
                    else:
                        delta_to_send = delta.content

                    message.content += delta_to_send
                    self.event_handler.pub_message_delta(self.message.id, index, delta_to_send, delta.role)
                    index += 1

        except Exception as e:
            logging.error("handle_llm_response error: %s", e)
            raise e

        if self._buffer_prefix:
            message.content += self._buffer_prefix
            self.event_handler.pub_message_delta(self.message.id, index, self._buffer_prefix, "assistant")
            index += 1
            self._buffer_prefix = ""
        
        if self._matched_prefix in ("RC0001", "CS0001", "SR0001"):
            tail = ","
        elif self._matched_prefix in ("CS0002", "SR0002", "AD0001", "NC0001"):
            tail = "\t] \t}"
        else:
            logging.info("llm response message: %s", message.content)
            return message

        message.content += tail
        self.event_handler.pub_message_delta(self.message.id, index, tail, "assistant")
        logging.info("llm response message: %s", message.content)
        return message
