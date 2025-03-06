import httpx
from fastapi import APIRouter, Request, Response
from app.api.v1 import assistant, assistant_file, thread, message, files, runs, token, action

api_router = APIRouter(prefix="/v1")
proxy_router = APIRouter()

@proxy_router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(full_path: str, request: Request):
    # Rebuild the URL for the remote service including any query parameters.
    query = request.url.query
    print(query, full_path)
    remote_url = f"http://172.31.19.187:11343/v1/{full_path}"
    if query:
        remote_url = f"{remote_url}?{query}"

    # Read the request body (if any)
    body = await request.body()

    # Forward the request to the remote service using httpx.
    async with httpx.AsyncClient() as client:
        forwarded_response = await client.request(
            method=request.method,
            url=remote_url,
            headers=request.headers.raw,
            content=body,
        )

    # Construct and return a response with the forwarded content.
    return Response(
        content=forwarded_response.content,
        status_code=forwarded_response.status_code,
        headers=dict(forwarded_response.headers)
    )

def router_init():
    api_router.include_router(assistant.router, prefix="/assistants", tags=["assistants"])
    api_router.include_router(assistant_file.router, prefix="/assistants", tags=["assistants"])
    api_router.include_router(thread.router, prefix="/threads", tags=["threads"])
    api_router.include_router(message.router, prefix="/threads", tags=["messages"])
    api_router.include_router(runs.router, prefix="/threads", tags=["runs"])
    api_router.include_router(files.router, prefix="/files", tags=["files"])
    api_router.include_router(token.router, prefix="/tokens", tags=["tokens"])
    api_router.include_router(action.router, prefix="/actions", tags=["actions"])
    api_router.include_router(proxy_router)
