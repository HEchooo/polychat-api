import logging
import os
from config.config import settings

def register(app):
    from app.core.tools.mcp.manager import MCPServerManager
    
    @app.on_event("startup")
    def startup_mcp():
        """应用启动时初始化 MCP 服务器管理器"""
        logging.info("Initializing MCP server manager")
        # 创建服务器管理器实例
        server_manager = MCPServerManager()
        
        # 启动工具服务器
        script_path = os.path.join(settings.BASE_PATH, "app/core/tools/mcp/server.py")
        server_manager.start_server("polychat_tools", script_path)
        
    @app.on_event("shutdown")
    def shutdown_mcp():
        logging.info("Stopping all MCP servers")
        server_manager = MCPServerManager()
        server_manager.stop_all_servers()