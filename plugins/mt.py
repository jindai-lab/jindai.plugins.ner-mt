"""Machine Translation
@chs 机器翻译
"""
import requests
import time
import random
import re
from opencc import OpenCC

from jindai.helpers import safe_import
from jindai.models import Paragraph
from jindai import PipelineStage, Plugin


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

    def split_chunks(self, content):
        """Split content into chunks of no more than 5000 characters"""
        while len(content) > 5000:
            res = re.match(r'.*[\.\?!。？！”’\'\"]', content)
            if res:
                yield res.group()
                content = content[:len(res.group())]
            else:
                yield content[:5000]
                content = content[5000:]
        yield content

    def resolve(self, paragraph):
        """Translate the paragraph
        """
        result = ''
        try:
            for chunk in self.split_chunks(paragraph.content):
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
