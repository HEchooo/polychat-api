<div align="center">

# Polychat API

_✨ An out-of-the-box AI intelligent assistant API ✨_

</div>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

## Introduction

Polychat API is an open-source, self-hosted AI intelligent assistant API, compatible with the official OpenAI
interface. It can be used directly with the official OpenAI [Client](https://github.com/openai/openai-python) to build
LLM applications.

It supports [One API](https://github.com/songquanpeng/one-api) for integration with more commercial and private models.

It supports [R2R](https://github.com/SciPhi-AI/R2R) RAG engine。

## Usage

Below is an example of using the official OpenAI Python `openai` library:

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8086/api/v1",
    api_key="xxx"
)

assistant = client.beta.assistants.create(
    name="demo",
    instructions="You are a helpful assistant.",
    model="gpt-4-1106-preview"
)
```

## Why Choose Polychat API

| Feature                  | Polychat  API                      | OpenAI Assistant API |
|--------------------------|------------------------------------|----------------------|
| Ecosystem Strategy       | Open Source                        | Closed Source        |
| RAG Engine               | Support R2R/Faiss(WIP) etc.        | Supported            |
| Internet Search          | Supported                          | Not Supported        |
| Custom Functions         | Supported                          | Supported            |
| Built-in Tool            | Extendable                         | Not Extendable       |
| Code Interpreter         | Supports(WIP)                      | Supported            |
| Multimodal               | Supported                          | Supported            |
| LLM Endpoint Switch      | Endpoints Dynamic Changeable (WIP) | Only GPT             |
| LLM Support              | Supports More LLMs                 | Only OpenAI          |
| Message Streaming Output | Supports                           | Supported            |
| Local Deployment         | Supported                          | Not Supported        |


- **LLM Support**: Compared to the official OpenAI version, more models can be supported through different LLM serving APIs (oenAPI, Ollama or vLLM or openAI).
- **Tool**: Currently supports online search; can easily expand more tools.
- **RAG Engine**: Multimodel suport, the currently supported file types are txt, html, markdown, pdf, docx, pptx, xlsx, png, mp3, mp4, etc.
  We provide a preliminary. We support up-to-date R2R engine and Faiss support and more are working in progress.
- **Message Streaming Output**: Support message streaming output for a smoother user experience.
- **Ecosystem Strategy**: Open source, you can deploy the service locally and expand the existing features.

## Quick Start

The easiest way to start the Polychat API is to run the docker-compose.yml file. Make sure Docker and Docker
Compose are installed on your machine before running.

### Configuration

Go to the project root directory, open `docker-compose.yml`, fill in the openai api_key and bing search key (optional).

```sh
# openai api_key (supports OneAPI api_key)
OPENAI_API_KEY=<openai_api_key>

# bing search key (optional)
BING_SUBSCRIPTION_KEY=<bing_subscription_key>
```

It is recommended to configure the R2R RAG engine to replace the default RAG implementation to provide better RAG capabilities.
You can learn about and use R2R through the [R2R Github repository](https://github.com/SciPhi-AI/R2R).

```sh
# RAG config
# FILE_SERVICE_MODULE=app.services.file.impl.oss_file.OSSFileService
FILE_SERVICE_MODULE=app.services.file.impl.r2r_file.R2RFileService
R2R_BASE_URL=http://<r2r_api_address>
R2R_USERNAME=<r2r_username>
R2R_PASSWORD=<r2r_password>
```

### Run

#### Run with Docker Compose:

 ```sh
docker compose up -d
 ```

### Access API

Api Base URL: http://127.0.0.1:8086/api/v1

Interface documentation address: http://127.0.0.1:8086/docs

### Complete Usage Example

In this example, an AI assistant is created and run using the official OpenAI client library. If you need to explore other usage methods,
such as streaming output, tools (web_search, retrieval, function), etc., you can find the corresponding code under the examples directory.
Before running, you need to run `pip install openai` to install the Python `openai` library.

```sh
# !pip install openai
export PYTHONPATH=$(pwd)
python examples/run_assistant.py
```


### Permissions
Simple user isolation is provided based on tokens to meet SaaS deployment requirements. It can be enabled by configuring `APP_AUTH_ENABLE`.

![](docs/imgs/user.png)

1. The authentication method is Bearer token. You can include `Authorization: Bearer ***` in the header for authentication.
2. Token management is described in the token section of the API documentation. Relevant APIs need to be authenticated with an admin token, which is configured as `APP_AUTH_ADMIN_TOKEN` and defaults to "admin".
3. When creating a token, you need to provide the base URL and API key of the large model. The created assistant will use the corresponding configuration to access the large model.

### Tools
According to the OpenAPI/Swagger specification, it allows the integration of various tools into the assistant, empowering and enhancing its capability to connect with the external world.

1. Facilitates connecting your application with other systems or services, enabling interaction with the external environment, such as code execution or accessing proprietary information sources.
2. During usage, you need to create tools first, and then you can integrate them with the assistant. Refer to the test cases for more details.[Assistant With Action](tests/tools/assistant_action_test.py)
3. If you need to use tools with authentication information, simply add the authentication information at runtime. The specific parameter format can be found in the API documentation. Refer to the test cases for more details. [Run With Auth Action](tests/tools/run_with_auth_action_test.py)

## Special Thanks

We mainly referred to and relied on the following projects:

- [Open Assistant API]([https://github.com/transitive-bullshit/OpenOpenAI](https://github.com/MLT-OSS/open-assistant-api)): Open Assistant API
- [OpenOpenAI](https://github.com/transitive-bullshit/OpenOpenAI): Assistant API implemented in Node
- [One API](https://github.com/songquanpeng/one-api): Multi-model management tool
- [R2R](https://github.com/SciPhi-AI/R2R): RAG engine
- [OpenAI-Python](https://github.com/openai/openai-python): OpenAI Python Client
- [OpenAI API](https://github.com/openai/openai-openapi): OpenAI interface definition
- [LangChain](https://github.com/langchain-ai/langchain): LLM application development library
- [OpenGPTs](https://github.com/langchain-ai/opengpts): LangChain GPTs
- [TaskingAI](https://github.com/TaskingAI/TaskingAI): TaskingAI Client SDK

## Contributing

Please read our [contribution document](./docs/CONTRIBUTING.md) to learn how to contribute.

## Open Source License

This repository follows the MIT open source license. For more information, please see the [LICENSE](./LICENSE) file.
