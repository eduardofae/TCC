import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from metrics import register_metric

nltk.download('wordnet')
nltk.download('omw-1.4')

class PortugueseWordNetWrapper:
    def __init__(self):
        self.wn = wordnet

    def synsets(self, word, pos=None):
        return self.wn.synsets(word, pos=pos, lang='por')
    
pt_wordnet_oficial = PortugueseWordNetWrapper()
pt_stemmer = SnowballStemmer('portuguese')

@register_metric
def add_meteor(prediction, reference, results):
    metric_name = 'meteor'
    t_ref = word_tokenize(reference, language='portuguese')
    t_pred = word_tokenize(prediction, language='portuguese')
    value = single_meteor_score(hypothesis=t_pred, reference=t_ref, stemmer=pt_stemmer, wordnet=pt_wordnet_oficial)
    results[metric_name] = value