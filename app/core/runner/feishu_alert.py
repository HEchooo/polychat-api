#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import logging
import urllib
import requests
import hmac
import hashlib
import base64
from datetime import datetime

from queue import Empty, Queue
from threading import Thread

class FeishuApi:
   
    API = 'https://open.feishu.cn/open-apis/bot/v2/hook/{0}'

    def __init__(self, access_token, access_secret, environment = 'dev'):
        self.logger = logging.getLogger(__name__)
        self.api = self.API.format(access_token)
        self.access_secret = access_secret
        self.environment = environment
        self.session = requests.session()


    def get_timestamp(self):
        return str(int(time.time()))

    def __build_sign(self):
        timestamp = self.get_timestamp()
        string_to_sign = '{}\n{}'.format(timestamp, self.access_secret)
        hmac_code = hmac.new(string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return timestamp, sign

    def send_text(self, content):
        timestamp, sign = self.__build_sign()
        headers = {"Content-Type": "application/json"}
        message = {
            'timestamp': str(timestamp),
            'sign': sign,
            'msg_type':'text',
            'content':{
                'text': f'[{self.environment}]-{content}'
            }
        }
        response = self.http_request('POST', self.api, json = message, add_to_headers = headers)
        self.logger.info(f'response={response}')

    def send_post(self, title, content):
        timestamp, sign = self.__build_sign()
        headers = {"Content-Type": "application/json"}
        message = {
            'msg_type':'post',
            'timestamp': str(timestamp),
            'sign': sign,
            'content':{
                'post': {
                    "zh_cn": {
                        'title': f'[{self.environment}]-{title}',
                        'content': [
                            [
                                {
                                    'tag': 'text',
                                    'text': f'{content}'
                                }
                            ]
                        ]
                    }
                }
            }
        }
        
        response = self.http_request('POST', self.api, json = message, add_to_headers = headers)
        self.logger.info(f'response={response}')

    def http_request(self, method_type, url, params=None, data=None, json=None, add_to_headers=None):
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;',
            "Content-type": "application/json",
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36'
        }
        if params is not None:
            params = urllib.parse.urlencode(params)
        if add_to_headers is not None:
            headers.update(add_to_headers)

        if self.session is None:
            self.session = requests.session()
        try:
            response = self.session.request(method_type, url=url, params=params, data=data, json=json, headers=headers, timeout=100  )# verify = False
            if not str(response.status_code).startswith('2'):
                return {
                    "success": False,
                    "code": response.status_code,
                    "msg": "服务器异常{}".format(response.text)
                }
            else:
                res = response.json()
                return {"success": True, "code": 0, "data": res}
        except BaseException as e:
            print("%s , %s" % (url, e))
            return {"success": False, "code": -1, "msg": e}

class NotifyMsg(object):

    def __init__(self, title, message) -> None:
        self.title = title
        self.message = message

class Notifier(object):

    def send_notify(self, title, content) -> None:
        pass

class FeishuNotifier(Notifier):

    def __init__(self, access_token = None, access_secret = None, environment = None):
        # Import here to avoid circular imports
        from config.config import settings
        
        # Use provided values or fall back to settings
        self.access_token = access_token or settings.FEISHU_ACCESS_TOKEN
        self.access_secret = access_secret or settings.FEISHU_ACCESS_SECRET
        self.environment = environment or settings.FEISHU_ENVIRONMENT
        
        self.thread: Thread = Thread(name = 'feishu_notify', target=self.run)
        self.queue: Queue = Queue()
        self.active: bool = False
        
        # Only create FeishuApi if we have valid credentials
        if self.access_token and self.access_secret:
            self.feishu_api = FeishuApi(self.access_token, self.access_secret, self.environment)
        else:
            self.feishu_api = None

    def is_assistant_enabled(self, assistant_id):
        """Check if the assistant is enabled for Feishu notifications"""
        from config.config import settings
        
        if not settings.FEISHU_ENABLED_ASSISTANTS:
            return True
        
        enabled_assistants = [aid.strip() for aid in settings.FEISHU_ENABLED_ASSISTANTS.split(',') if aid.strip()]
        return assistant_id in enabled_assistants

    def send_notify(self, title, content, assistant_id=None) -> None:
        if not self.feishu_api:
            return
        
        if assistant_id and not self.is_assistant_enabled(assistant_id):
            logging.info(f"Assistant {assistant_id} is not enabled for notifications")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_with_timestamp = f"{title}-{timestamp}"
            
        if not self.active:
            self.start()
        self.queue.put(NotifyMsg(title_with_timestamp, content))

    def run(self) -> None:
        while self.active:
            try:
                msg = self.queue.get(block=True, timeout=1)
                if self.feishu_api:
                    self.feishu_api.send_post(msg.title, msg.message)
            except Empty:
                pass

    def start(self) -> None:
        self.active = True
        self.thread.start()

    def close(self) -> None:
        if not self.active:
            return

        self.active = False
        self.thread.join()

feishu_notifier = FeishuNotifier()