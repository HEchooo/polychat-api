from typing import Type
import subprocess
import sys
import importlib.util
from pydantic import BaseModel, Field

from app.core.tools.base_tool import BaseTool
from config.llm import tool_settings

def ensure_package_installed(package_name):
    package_import_name = package_name.replace('-', '_')
    if importlib.util.find_spec(package_import_name) is None:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} 安装完成")
    else:
        print(f"{package_name} 已安装")


ensure_package_installed('duckduckgo_search')

from duckduckgo_search import DDGS

class WebSearchToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Search query. Use a format suitable for duckduckgo and, if necessary, "
    )

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "A tool for performing web searches and extracting snippets of web pages,"
        "Use when you need to search for unknown information or when your information is not up to date."
        "The input should be a search query."
    )
    
    args_schema: Type[BaseModel] = WebSearchToolInput
    
    def run(self, query: str) -> dict:
        ensure_package_installed('duckduckgo-search')
        
        ddgs = DDGS()
        results = ddgs.text(
            query, 
            region='wt-wt',
            safesearch='moderate', 
            timelimit=None, 
            max_results=tool_settings.WEB_SEARCH_NUM_RESULTS
        )
        
        return results