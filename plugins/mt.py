"""Machine Translation
@chs 机器翻译
"""
import requests
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

    def resolve(self, paragraph):
        """Translate the paragraph
        """
        if not paragraph.content:
            return

        resp = requests.post(self.server, json={
            'text': paragraph.content,
            'source_lang': paragraph.lang,
            'target_lang': self.to_lang.upper()
        })
        try:
            resp = resp.json()
            paragraph.content = resp['data']
            if self.to_lang == 'zh':
                paragraph.content = self.convert(paragraph.content)
        except ValueError:
            self.logger('Error while reading from remote server')
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
