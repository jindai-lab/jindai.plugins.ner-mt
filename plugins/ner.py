"""
Named Entity Recognition
@chs 命名实体识别
"""
import re

from jindai import PipelineStage, Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


class HanLPModel(PipelineStage):
    """
    Process with HanLP Models
    @chs 使用 HanLP 的预训练模型进行处理
    """

    def __init__(self, pretrained, result=''):
        """
        Args:
            result (tok|ner|srl|sdp/dm|sdp/pas|sdp/psd|con|lem|pos|fea|dep):
                Results to fetch
                @chs 要提取的结果
            pretrained (ZH|JA|OT):
                Pretrained model set
                @chs 预训练模型选择
        """
        hanlp = safe_import('hanlp')
        self.result = result
        pretrained = {
            'ZH': 'CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH',
            'JA': 'NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA',
            'OT': 'UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE'
        }.get(pretrained, 'ZH')
        self.model = hanlp.load(getattr(hanlp.pretrained.mtl, pretrained))
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        setattr(paragraph, self.result, self.model(
            paragraph.content, tasks=self.result)[self.result])
        return paragraph


class HanlpNerZh(PipelineStage):
    """
    NER on Modern Chinese texts with HanLP
    @chs 使用 HanLP 的预训练模型进行现代汉语命名实体识别（NER）
    """

    def __init__(self) -> None:
        """
        Results will be in form of [<name>, <type>]
        @chs 提取的结果格式为 [名称, 类型]
        """
        super().__init__()
        hanlp = safe_import('hanlp')
        self.model = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.ner = self.model(paragraph.content, tasks='ner').get('ner/msra', [])
        return paragraph


class HanlpNerJa(PipelineStage):
    """
    NER on Japanese texts with HanLP
    @chs 使用 HanLP 的预训练模型进行日语命名实体识别（NER）
    """

    def __init__(self) -> None:
        super().__init__()
        hanlp = safe_import('hanlp')
        self.model = hanlp.load(
            hanlp.pretrained.mtl.hanlp.pretrained.mtl.NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA)

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.ner = self.model(re.sub(r'\s', '', paragraph.content),
                           tasks='ner').get('ner', [])
        return paragraph


class SpaCyNER(PipelineStage):
    """
    NER with SpaCy
    @chs 使用 SpaCy 的预训练模型进行命名实体识别（NER）
    """

    def __init__(self, lang) -> None:
        """ 
        Results will be in form of [<name>, <type>]
        @chs 提取的结果格式为 [名称, 类型]
        
        Args:
            lang (LANG):
                Language
                @chs 语言
        """
        spacy = safe_import('spacy')
        model = {
            'en': 'en_core_web_sm',
            'chs': 'zh_core_web_sm',
            'de': 'de_core_news_sm',
            'ja': 'ja_core_news_sm',
            'fr': 'fr_core_news_sm',
            'ru': 'ru_core_news_sm',
        }.get(lang, 'en_core_web_sm')
        self.model = spacy.load(model)
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.ner = [(e.text, {'ORG': 'ORGANIZATION'}.get(e.label_, e.label_))
                 for e in self.model(paragraph.content).ents]
        return paragraph


class AutoNER(PipelineStage):
    """
    Auto NER based on language code
    @chs 根据输入语段自动进行对应语言的命名实体识别
    """

    def __init__(self, ifchs='hanlp', ifja='hanlp') -> None:
        """
        Args:
            ifchs (hanlp|spacy):
                Method for Chinese texts
                @chs 中文命名实体识别方法
            ifja (hanlp|spacy):
                Method for Japanese texts
                @chs 日文命名实体识别方法
        """
        self.models_meta = {
            'chs': HanlpNerZh if ifchs == 'hanlp' else lambda: SpaCyNER('chs'),
            'ja': HanlpNerJa if ifja == 'hanlp' else lambda: SpaCyNER('ja'),
            'de': lambda: SpaCyNER('de'),
            'fr': lambda: SpaCyNER('fr'),
            'ru': lambda: SpaCyNER('ru'),
            'en': lambda: SpaCyNER('en')
        }
        self.models = {}
        super().__init__()

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        if paragraph.lang not in self.models and paragraph.lang in self.models_meta:
            self.models[paragraph.lang] = self.models_meta[paragraph.lang]()
        return self.models[paragraph.lang].resolve(paragraph)


class NERAsTokens(PipelineStage):
    """
    Take NER results as tokens
    @chs 将命名实体识别的结果作为词袋
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        paragraph.tokens = [token[0] + '/' + token[1]
                    for token in paragraph.ner] if hasattr(paragraph, 'ner') else []
        return paragraph


class NERPlugin(Plugin):
    """NER Plugin"""

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
