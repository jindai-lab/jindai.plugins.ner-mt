"""
Sentiment Analysis
@chs 情感分析
"""

from jindai import PipelineStage, Plugin
from jindai.helpers import safe_import
from jindai.models import Paragraph


class AutoSentimentAnalysis(PipelineStage):
    """
    Sentiment Analysis with SnowNLP
    @chs 使用 SnowNLP 进行自动情感分析
    """

    def resolve(self, paragraph: Paragraph) -> Paragraph:
        snow = safe_import('snownlp').SnowNLP
        paragraph.sentiment = (snow(paragraph.content).sentiments-0.5) * 2
        return paragraph


class SentimentAnalysisPlugin(Plugin):
    """Sentiment Analysis Plugin"""

    def __init__(self, pmanager, **config):
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())
