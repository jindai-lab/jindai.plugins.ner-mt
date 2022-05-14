"""Machine Translation
@chs 机器翻译
"""
import hashlib
import json
import random
import re
import time
import uuid
from hashlib import md5

import requests
from opencc import OpenCC
from jindai import PipelineStage, Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


def split_chunks(content, max_len=5000):
    """Split content into chunks of no more than `max_len` characters"""
    while len(content) > max_len:
        res = re.match(r'.*[\.\?!。？！”’\'\"]', content)
        if res:
            yield res.group()
            content = content[:len(res.group())]
        else:
            yield content[:max_len]
            content = content[max_len:]
    yield content


class RemoteTranslation(PipelineStage):
    """Machine Translation with Remote API Calls
    @chs 调用远程 API 进行机器翻译
    """

    def __init__(self, server='http://localhost/translate', to_lang='chs') -> str:
        """
        :param to_lang:
            Target language
            @chs 目标语言标识
        :type to_lang: LANG
        :param server:
            Entrypoint for remote calls
            @chs 远程调用的入口点
        :type server: str
        :return: Translated text
        :rtype: str
        """
        super().__init__()
        self.server = server
        self.to_lang = to_lang if to_lang not in ('chs', 'cht') else 'zh'
        self.convert = OpenCC('s2t' if to_lang == 'cht' else 't2s').convert

    def resolve(self, paragraph):
        """Translate the paragraph
        """
        result = ''
        try:
            for chunk in split_chunks(paragraph.content):
                chunk = chunk.strip()
                if not chunk:
                    continue

                resp = requests.post(self.server, json={
                    'text': chunk,
                    'source_lang': paragraph.lang.upper() if paragraph.lang != 'auto' else 'auto',
                    'target_lang': self.to_lang.upper()
                })

                resp = resp.json()
                if resp.get('code') != 200:
                    self.logger('Error while translating:',
                                resp.get('msg', 'null message'))
                    return

                content = resp.json()['data']
                if self.to_lang == 'zh':
                    content = self.convert(content)
                if not self.to_lang in ('zh', 'jp', 'kr'):
                    result += ' '
                result += content

                time.sleep(1 + random.random())

        except ValueError:
            self.logger('Error while reading from remote server')
            return

        if result:
            paragraph.content = result.strip()

        return paragraph


class YoudaoTranslation(PipelineStage):
    """Machine Translation via Youdao
    @chs 有道云机器翻译（付费）
    """

    def __init__(self, api_id, api_key, to_lang='chs'):
        """
        :param api_id: Youdao API ID
        :type api_id: str
        :param api_key: Youdao API Key
        :type api_key: str
        :param to_lang: Target language code
            @chs 目标语言
        :type to_lang: LANG
        """
        super().__init__()
        self.api_id, self.api_key = api_id, api_key
        self.to_lang = to_lang

    def resolve(self, paragraph: Paragraph):

        def _regulate_lang(lang):
            if lang in ('chs', 'cht'):
                return 'zh-' + lang.upper()
            else:
                return lang

        translate_text = paragraph.content.strip()

        if not translate_text:
            return

        youdao_url = 'https://openapi.youdao.com/api'
        input_text = ""

        if(len(translate_text) <= 20):
            input_text = translate_text
        elif(len(translate_text) > 20):
            input_text = translate_text[:10] + \
                str(len(translate_text)) + translate_text[-10:]

        time_curtime = int(time.time())
        uu_id = uuid.uuid4()
        sign = hashlib.sha256(
            f'{self.api_id}{input_text}{uu_id}{time_curtime}{self.api_key}'.encode('utf-8')).hexdigest()
        data = {
            'q': translate_text,
            'from': _regulate_lang(paragraph.lang),   # 源语言
            'to': _regulate_lang(self.to_lang),
            'appKey': self.api_id,
            'salt': uu_id,
            'sign': sign,
            'signType': "v3",
            'curtime': time_curtime,
        }

        resp = requests.get(youdao_url, params=data).json()
        paragraph.content = resp['translation'][0]
        return paragraph


class BaiduTranslation(PipelineStage):
    """Machine Translation via Baidu API
    @chs 百度云机器翻译
    """

    def __init__(self, to_lang='chs', api_key='', api_id=''):
        """
        :param to_lang: Target language
            @chs 目标语言
        :type to_lang: LANG
        :param api_key: API Key
        :type api_key: str, optional
        :param api_id: API ID
        :type api_id: str, optional
        """
        super().__init__()
        self.to_lang = to_lang
        self.api_id, self.api_key = api_id, api_key

    def resolve(self, paragraph: Paragraph):
        api_endpoint = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

        def _regulate_lang(lang):
            if lang == 'chs':
                return 'zh'
            return lang

        result = ''

        for query in split_chunks(paragraph.content, max_len=2000):
            salt = random.randint(32768, 65536)
            sign = md5(f'{self.api_id}{query}{salt}{self.api_key}'.encode(
                'utf-8')).hexdigest()

            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {'appid': self.api_id,
                       'q': query,
                       'from': _regulate_lang(paragraph.lang),
                       'to': _regulate_lang(self.to_lang),
                       'salt': salt,
                       'sign': sign}

            time.sleep(1)
            resp = requests.post(api_endpoint, params=payload,
                                 headers=headers).json()
            if 'error_msg' in resp:
                raise ValueError(resp['error_msg'])
            result += ' '.join([_['dst'] for _ in resp['trans_result']])

        paragraph.content = result
        return paragraph


class MachineTranslation(PipelineStage):
    """Machine Translation
    @chs 机器翻译"""

    def __init__(self, to_lang='chs', model='opus-mt') -> None:
        """
        Args:
            to_lang (LANG):
                Target language
                @chs 目标语言标识
            model (opus-mt|mbart50_m2m):
                Model for translation
                @chs 机器翻译所使用的模型 (opus-mt 较快速度, mbart50_m2m 较高准确度)
        """
        super().__init__()

        self.model = safe_import('easynmt').EasyNMT(model)

        self.opencc = None
        if to_lang == 'chs':
            to_lang = 'zh'
        elif to_lang == 'cht':
            to_lang = 'zh'
            self.opencc = safe_import(
                'opencc', 'opencc-python-reimplemented').OpenCC('s2t')

        self.to_lang = to_lang

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        """处理段落"""
        translated = self.model.translate(
            paragraph.content,
            source_lang=paragraph.lang if paragraph.lang not in (
                'chs', 'cht') else 'zh',
            target_lang=self.to_lang)
        if self.opencc:
            translated = self.opencc.convert(translated)
        paragraph.content = translated
        return paragraph


class MachineTranslationPlugin(Plugin):
    """Plugin for machin translations
    """

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
