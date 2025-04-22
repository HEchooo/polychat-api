import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.tools.base_tool import BaseTool
from app.models.run import Run
from sqlalchemy.orm import Session
from config.config import settings

from .manager import MCPServerManager


class MCPToolBase(BaseTool):
    
    name = ""
    description = ""
    _mcp_client_initialized = False
    
    def __init__(self) -> None:
        self.mcp_client = None
        self.run = None
        self.session = None
        self.reader = None
        self.writer = None
        self.client_session = None
        
    def configure(self, session: Session, run: Run, **kwargs):
        """配置 MCP 工具"""
        self.run = run
        self.session = session
    
    async def initialize_mcp_client(self, server_name: str, script_path: str):
        """初始化 MCP 客户端
        
        Args:
            server_name: 服务器名称
            script_path: 服务器脚本路径
        """
        # 获取服务器管理器实例
        server_manager = MCPServerManager()
        
        # 确保服务器已启动
        server_manager.start_server_if_not_running(server_name, script_path)
        
        # 创建 MCP 客户端
        server_params = StdioServerParameters(
            command="python",
            args=[script_path],
        )
        
        try:
            # 创建连接
            read, write = await stdio_client(server_params)
            self.reader = read
            self.writer = write
            
            # 创建会话
            self.client_session = ClientSession(read, write)
            await self.client_session.initialize()
            
            self._mcp_client_initialized = True
            logging.info(f"MCP client for {self.name} initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize MCP client for {self.name}: {e}")
            self._mcp_client_initialized = False
            raise
    
    async def call_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        """调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if not self.client_session:
            raise ValueError("MCP client not initialized")
            
        try:
            # 调用工具
            result = await self.client_session.call_tool(tool_name, kwargs)
            return result
        except Exception as e:
            logging.error(f"Error calling MCP tool {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        if self.client_session:
            try:
                await self.client_session.__aexit__(None, None, None)
                self.client_session = None
            except Exception as e:
                logging.error(f"Error cleaning up MCP client session: {e}")

        if self.reader or self.writer:
            try:
                if self.writer:
                    self.writer.close()
                self.reader = None
                self.writer = None
            except Exception as e:
                logging.error(f"Error cleaning up MCP client transport: {e}")


class MCPFileSearchTool(MCPToolBase):
    """MCP 文件搜索工具"""
    
    name = "mcp_file_search"
    description = "可以用于搜索已上传到助手的文件内容。如果用户引用特定文件，这通常是一个很好的提示，信息可能在此处。"
    
    def __init__(self) -> None:
        super().__init__()
        self.__filenames = []
        self._keys = []  # 使用 _keys 以便在调用中访问
        
    def configure(self, session: Session, run: Run, **kwargs):
        super().configure(session, run, **kwargs)
        
        from app.services.file.file import FileService
        
        files = FileService.get_file_list_by_ids(session=session, file_ids=run.file_ids)
        # 预缓存数据以防止后续线程冲突
        for file in files:
            self.__filenames.append(file.filename)
            self._keys.append(file.key)
    
    async def _initialize_client(self):
        """初始化 MCP 客户端"""
        if self._mcp_client_initialized:
            return
            
        script_path = os.path.join(settings.BASE_PATH, "app/core/tools/mcp/server.py")
        await self.initialize_mcp_client("polychat_tools", script_path)
    
    def run(self, indexes: List[int], query: str) -> Dict[str, Any]:
        """运行文件搜索工具
        
        Args:
            indexes: 文件索引列表
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        # 获取文件路径
        file_keys = []
        for index in indexes:
            if index < len(self._keys):
                file_key = self._keys[index]
                file_keys.append(file_key)
            else:
                logging.warning(f"索引 {index} 超出文件键值范围")
        
        try:
            # 尝试通过 MCP 调用
            async def _call_tool():
                # 延迟初始化
                await self._initialize_client()
                # 返回结果
                result = await self.call_mcp_tool("file_search", indexes=indexes, query=query)
                return result
            
            # 运行 MCP 调用
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_call_tool())
            finally:
                loop.close()
                
            # 检查结果
            if isinstance(result, dict) and "error" in result:
                # 出错了，回退到原始实现
                raise Exception(result["error"])
                
            return result
            
        except Exception as e:
            logging.warning(f"MCP file_search failed: {e}, falling back to direct implementation")
            # 回退到原始实现
            from app.services.file.file import FileService
            return FileService.search_in_files(query=query, file_keys=file_keys)
    
    def _fallback_to_original(self, indexes: List[int], query: str) -> Dict[str, Any]:
        """回退到原始文件搜索工具"""
        file_keys = []
        for index in indexes:
            if index < len(self._keys):
                file_key = self._keys[index]
                file_keys.append(file_key)
        
        from app.services.file.file import FileService
        return FileService.search_in_files(query=query, file_keys=file_keys)
    
    def instruction_supplement(self) -> str:
        """提供文件选择信息"""
        if len(self.__filenames) == 0:
            return ""
        else:
            filenames_info = [f"({index}){filename}" for index, filename in enumerate(self.__filenames)]
            return (
                '你可以使用 "mcp_file_search" 工具从以下附加文件中检索相关上下文。'
                + '每行表示一个文件，格式为"(索引)文件名":\n'
                + "\n".join(filenames_info)
                + "\n使用附加文件时请尽量简洁。"
            )


class MCPWebSearchTool(MCPToolBase):
    """MCP 网络搜索工具"""
    
    name = "mcp_web_search"
    description = (
        "用于执行 Bing 搜索并提取片段和网页，适用于需要搜索未知内容或信息不是最新时。"
        "输入应为搜索查询。"
    )
    
    async def _initialize_client(self):
        """初始化 MCP 客户端"""
        if self._mcp_client_initialized:
            return
            
        script_path = os.path.join(settings.BASE_PATH, "app/core/tools/mcp/server.py")
        await self.initialize_mcp_client("polychat_tools", script_path)
    
    def run(self, query: str) -> Dict[str, Any]:
        """运行网络搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果
        """
        try:
            # 尝试通过 MCP 调用
            async def _call_tool():
                # 延迟初始化
                await self._initialize_client()
                # 返回结果
                return await self.call_mcp_tool("web_search", query=query)
            
            # 运行 MCP 调用
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_call_tool())
            finally:
                loop.close()
                
            # 检查结果
            if isinstance(result, dict) and "error" in result:
                # 出错了，回退到原始实现
                raise Exception(result["error"])
                
            return result
            
        except Exception as e:
            logging.warning(f"MCP web_search failed: {e}, falling back to direct implementation")
            # 回退到原始实现
            return self._fallback_to_original(query)
    
    def _fallback_to_original(self, query: str) -> Dict[str, Any]:
        """回退到原始网络搜索工具"""
        from app.core.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        return tool.run(query=query)