import logging
import os
import subprocess
import atexit
import signal
from typing import Dict, Optional, Any

from config.config import settings


class MCPServerManager:
    """管理 MCP 服务器进程"""
    
    _instance = None
    _servers: Dict[str, subprocess.Popen] = {}
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPServerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._script_paths = {}
        atexit.register(self.stop_all_servers)
    
    def start_server(self, server_name: str, script_path: str) -> Optional[subprocess.Popen]:
        """启动 MCP 服务器进程
        
        Args:
            server_name: 服务器名称
            script_path: 脚本路径
            
        Returns:
            服务器进程实例
        """
        if server_name in self._servers:
            logging.info(f"MCP server '{server_name}' already running")
            return self._servers[server_name]
            
        try:
            server_process = subprocess.Popen(
                ["python", script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=0 
            )
            
            self._servers[server_name] = server_process
            self._script_paths[server_name] = script_path
            logging.info(f"Started MCP server '{server_name}' with PID {server_process.pid}")
            return server_process
            
        except Exception as e:
            logging.error(f"Failed to start MCP server '{server_name}': {e}")
            return None
    
    def get_server_process(self, server_name: str) -> Optional[subprocess.Popen]:
        return self._servers.get(server_name)
    
    def stop_server(self, server_name: str) -> bool:
        if server_name not in self._servers:
            logging.warning(f"MCP server '{server_name}' not found")
            return False
            
        server_process = self._servers[server_name]
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            
        except subprocess.TimeoutExpired:
            server_process.kill()
            logging.warning(f"Force killed MCP server '{server_name}'")
            
        except Exception as e:
            logging.error(f"Error stopping MCP server '{server_name}': {e}")
            return False
            
        del self._servers[server_name]
        logging.info(f"Stopped MCP server '{server_name}'")
        return True
    
    def stop_all_servers(self):
        server_names = list(self._servers.keys())
        for server_name in server_names:
            self.stop_server(server_name)
    
    def restart_server(self, server_name: str) -> Optional[subprocess.Popen]:
        if server_name not in self._script_paths:
            logging.error(f"Cannot restart server '{server_name}': script path not found")
            return None
            
        self.stop_server(server_name)
        return self.start_server(server_name, self._script_paths[server_name])
    
    def start_server_if_not_running(self, server_name: str, script_path: str) -> Optional[subprocess.Popen]:
        server_process = self.get_server_process(server_name)
        if server_process is not None:
            if server_process.poll() is None:
                return server_process
            else:
                del self._servers[server_name]
        
        self._script_paths[server_name] = script_path
        return self.start_server(server_name, script_path)