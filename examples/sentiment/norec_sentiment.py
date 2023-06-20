from skweak.base import CombinedAnnotator
from sentiment_lexicons import LexiconAnnotator, NRC_SentAnnotator, VADAnnotator, SocalAnnotator, BUTAnnotator
from sentiment_models import DocBOWAnnotator, MultilingualAnnotator, MBertAnnotator
import os
from spacy.tokens import Doc #type: ignore
from typing import Sequence, Tuple, Optional, Iterable
from collections import defaultdict


class FullSentimentAnnotator(CombinedAnnotator):
    """Annotation based on the heuristic"""

    def add_all(self, root):
        """Adds all implemented annotation functions, models and filters"""

        print("Loading lexicon functions")
        self.add_lexicons(root)
        print("Loading learned sentiment model functions")
        self.add_ml_models(root)

        return self

    def add_lexicons(self, root):
        """Adds Spacy NER models to the annotator"""

        self.add_annotator(LexiconAnnotator("norsent_forms", root + "data/sentiment/lexicons/norsentlex/Fullform"))
        self.add_annotator(LexiconAnnotator("norsent_lemma", root + "data/sentiment/lexicons/norsentlex/Lemma"))

        self.add_annotator(VADAnnotator("NRC_VAD", root + "data/sentiment/lexicons/NRC_VAD_Lexicon/Norwegian-no-NRC-VAD-Lexicon.txt"))

        self.add_annotator(SocalAnnotator("Socal-adj", root + "data/sentiment/lexicons/socal/no_adj.txt"))

        self.add_annotator(SocalAnnotator("Socal-adv", root + "data/sentiment/lexicons/socal/no_adv.txt"))

        self.add_annotator(SocalAnnotator("Socal-google", root + "data/sentiment/lexicons/socal/no_google.txt"))

        self.add_annotator(SocalAnnotator("Socal-int", root + "data/sentiment/lexicons/socal/no_int.txt"))

        self.add_annotator(SocalAnnotator("Socal-noun", root + "data/sentiment/lexicons/socal/no_noun.txt"))
        self.add_annotator(SocalAnnotator("Socal-verb", root + "data/sentiment/lexicons/socal/no_verb.txt"))

        self.add_annotator(SocalAnnotator("IBM", root + "data/sentiment/lexicons/IBM_Debater/no_unigram.txt"))

        self.add_annotator(NRC_SentAnnotator("NRC-Sent-Emo", root + "/data/sentiment/lexicons/NRC_Sentiment_Emotion/no_sent.txt"))

        self.add_annotator(BUTAnnotator("norsent_forms-BUT", root + "/data/sentiment/lexicons/norsentlex/Fullform"))

        self.add_annotator(BUTAnnotator("norsent_lemma-BUT", root + "/data/sentiment/lexicons/norsentlex/Lemma"))

        return self

    def add_ml_models(self, root):
        self.add_annotator(DocBOWAnnotator("doc-level-norec", root + "/data/sentiment/models/doc"))
        self.add_annotator(MultilingualAnnotator("nlptown-bert-multilingual-sentiment"))
        self.add_annotator(MBertAnnotator("mbert-sst"))
        return self

