import logging
import sys
import os
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

mcp = FastMCP("PolychatTools")

@mcp.tool()
async def file_search(indexes: List[int], query: str) -> Dict[str, Any]:
    """
    搜索文件中的内容
    
    Args:
        indexes: 要搜索的文件索引列表
        query: 搜索查询
        
    Returns:
        搜索结果
    """
    from app.services.file.file import FileService
    
    try:
        file_ids = []
        
        result = FileService.search_in_files(query=query, file_keys=file_ids)
        return result
    except Exception as e:
        logging.error(f"MCP file_search error: {e}")
        return {"error": str(e)}

@mcp.tool()
async def web_search(query: str) -> Dict[str, Any]:
    from app.core.tools.web_search import WebSearchTool
    
    try:
        tool = WebSearchTool()
        
        result = tool.run(query=query)
        return result
    except Exception as e:
        logging.error(f"MCP web_search error: {e}")
        return {"error": str(e)}

def run_server():
    logging.info("Starting MCP server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_server()