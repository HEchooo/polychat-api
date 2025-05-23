import logging
import os
import urllib.parse
from typing import Dict, Iterator

import requests

from app.schemas.tool.authentication import Authentication, AuthenticationType
from app.schemas.tool.action import ActionMethod, ActionBodyType, ActionParam


# This function code from the Open Source Project TaskingAI.
# The original code can be found at: https://github.com/TaskingAI/TaskingAI
def _prepare_headers(authentication: Authentication, extra_headers: Dict) -> Dict:
    """
    Prepares the headers for an HTTP request including authentication and additional headers.

    :param authentication: An Authentication object containing the authentication details.
    :param extra_headers: A dictionary of additional headers to include in the request.
    :return: A dictionary of headers for the HTTP request.
    """

    headers = {}

    if extra_headers:
        headers.update(extra_headers)

    if authentication:
        if authentication.type == AuthenticationType.basic:
            # Basic Authentication: Assume the secret is a base64 encoded string
            headers["Authorization"] = f"Basic {authentication.secret}"

        elif authentication.type == AuthenticationType.bearer:
            # Bearer Authentication: Add the secret as a bearer token
            headers["Authorization"] = f"Bearer {authentication.secret}"

        elif authentication.type == AuthenticationType.custom:
            # Custom Authentication: Return the custom content as headers
            headers.update(authentication.content)

    return headers


# This function code from the Open Source Project TaskingAI.
# The original code can be found at: https://github.com/TaskingAI/TaskingAI
def _process_parameters(schema: Dict[str, ActionParam], parameters: Dict) -> Dict:
    """
    Processes parameters based on their schema and provided values.

    :param schema: A dictionary representing the parameter schema.
    :param parameters: A dictionary of provided parameter values.
    :return: A dictionary of processed parameters.
    """
    processed_params = {}
    for param_name, action_param in schema.items():
        param_value = None
        # Check for the presence of 'enum' in the parameter schema
        if action_param.is_single_value_enum():
            param_value = action_param.enum[0]
        elif param_name in parameters:
            param_value = parameters[param_name]

        if param_value is not None:
            processed_params[param_name] = param_value

        # todo: check if required
        # if action_param.required and param_value is None:
        #     raise_http_error(ErrorCode.REQUEST_VALIDATION_ERROR, message=f"Missing required parameter {param_name}")

    return processed_params


# This function code from the Open Source Project TaskingAI.
# The original code can be found at: https://github.com/TaskingAI/TaskingAI
def call_action_api(
    url: str,
    method: ActionMethod,
    path_param_schema: Dict[str, ActionParam],
    query_param_schema: Dict[str, ActionParam],
    body_type: ActionBodyType,
    body_param_schema: Dict[str, ActionParam],
    parameters: Dict,
    headers: Dict,
    authentication: Authentication,
) -> Dict:
    """
    Call an API according to OpenAPI schema.
    :param url: the URL of the API call
    :param method: the method of the API call
    :param path_param_schema: the path parameters schema
    :param query_param_schema: the query parameters schema
    :param body_type: the body type
    :param body_param_schema: the body parameters schema
    :param parameters: the parameters input by the user
    :param headers: the extra headers to be included in the API call
    :param authentication: the authentication of the action
    :return: Response from the API call
    """
    authentication.decrypt()
    # Update URL with path parameters
    if path_param_schema:
        path_params = _process_parameters(path_param_schema, parameters)
        for param_name, param_value in path_params.items():
            url = url.replace(f"{{{param_name}}}", urllib.parse.quote(str(param_value)))

    # Prepare query parameters
    query_params = {}
    if query_param_schema:
        query_params = _process_parameters(query_param_schema, parameters)
        # cast boolean values to string
        for param_name, param_value in query_params.items():
            if isinstance(param_value, bool):
                query_params[param_name] = str(param_value).lower()

    # Prepare body
    body = None
    if body_type != ActionBodyType.NONE:
        body = _process_parameters(body_param_schema, parameters)

    # Prepare headers
    prepared_headers = _prepare_headers(authentication, headers)

    # Making the API call
    try:
        request_kwargs = {"headers": prepared_headers}

        if query_params:
            request_kwargs["params"] = query_params

        if os.environ.get("HTTP_PROXY_URL"):
            request_kwargs["proxy"] = os.environ.get("HTTP_PROXY_URL")

        if body_type == ActionBodyType.JSON:
            request_kwargs["json"] = body
            prepared_headers["Content-Type"] = "application/json"
        elif body_type == ActionBodyType.FORM:
            request_kwargs["data"] = body
            prepared_headers["Content-Type"] = "application/x-www-form-urlencoded"

        logging.info(f"call_action_api url={url} request kwargs: {request_kwargs}")

        with requests.request(method.value, url, **request_kwargs) as response:
            response_content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" in response_content_type:
                data = response.json()
            else:
                data = response.text
            if response.status_code == 500:
                error_message = f"API call failed with status {response.status_code}"
                if data:
                    error_message += f": {data}"
                return {"status": response.status_code, "error": error_message}
            return {"status": response.status_code, "data": data}
    except requests.exceptions.RequestException as e:
        return {"status": 500, "error": f"Failed to make the API call: {e}"}
    except Exception:
        return {"status": 500, "error": "Failed to make the API call"}


def call_action_api_stream(
    url: str,
    method: ActionMethod,
    path_param_schema: Dict[str, ActionParam],
    query_param_schema: Dict[str, ActionParam],
    body_type: ActionBodyType,
    body_param_schema: Dict[str, ActionParam],
    parameters: Dict,
    headers: Dict,
    authentication: Authentication,
) -> Iterator[str]:
    """
    Stream-compatible API caller. Yields response chunks as strings.
    """
    authentication.decrypt()
    if path_param_schema:
        path_params = _process_parameters(path_param_schema, parameters)
        for param_name, param_value in path_params.items():
            url = url.replace(f"{{{param_name}}}", urllib.parse.quote(str(param_value)))

    query_params = {}
    if query_param_schema:
        query_params = _process_parameters(query_param_schema, parameters)
        for param_name, param_value in query_params.items():
            if isinstance(param_value, bool):
                query_params[param_name] = str(param_value).lower()

    body = None
    if body_type != ActionBodyType.NONE:
        body = _process_parameters(body_param_schema, parameters)

    prepared_headers = _prepare_headers(authentication, headers)

    try:
        request_kwargs = {
            "headers": prepared_headers,
            "stream": True,
        }

        if query_params:
            request_kwargs["params"] = query_params
        if os.environ.get("HTTP_PROXY_URL"):
            request_kwargs["proxy"] = os.environ.get("HTTP_PROXY_URL")
        if body_type == ActionBodyType.JSON:
            request_kwargs["json"] = body
            prepared_headers["Content-Type"] = "application/json"
        elif body_type == ActionBodyType.FORM:
            request_kwargs["data"] = body
            prepared_headers["Content-Type"] = "application/x-www-form-urlencoded"

        logging.info(f"[STREAM] call_action_api_stream url={url} kwargs={request_kwargs}")

        with requests.request(method.value, url, **request_kwargs) as response:
            logging.info(f"[STREAM] response status: {response.status_code}")
            if response.status_code != 200:
                logging.warning(f"[STREAM] error response: {response.text}")
                yield f"[ERROR {response.status_code}] {response.text}"
                return

            response_content_type = response.headers.get("Content-Type", "").lower()
            logging.debug(f"[STREAM] content-type: {response_content_type}")

            for chunk in response.iter_content(chunk_size=256):
                if chunk:
                    try:
                        text = chunk.decode("utf-8")
                        logging.debug(f"[STREAM] raw chunk: {text}")
                        yield text  
                    except Exception as e:
                        logging.exception("[STREAM] decode error")
                        yield f"[DecodeError] {e}"

            logging.debug("[STREAM] yield done")

    except requests.exceptions.RequestException as e:
        logging.exception("[STREAM] request exception")
        yield f"[RequestException] {e}"
    except Exception as e:
        logging.exception("[STREAM] general exception")
        yield f"[Exception] {e}"
