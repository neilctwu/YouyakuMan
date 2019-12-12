import re
from pyknp import Juman
from configparser import ConfigParser
from pytorch_pretrained_bert import BertTokenizer
import pdb

config = ConfigParser()
config.read('./config.ini')


class JumanTokenizer:
    def __init__(self):
        self.juman = Juman(command=config['Juman']['command'],
                           option=config['Juman']['option'])

    def __call__(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class LangFactory:
    def __init__(self, lang):
        self.support_lang = ['en', 'jp']
        self.lang = lang
        self.stat = 'valid'
        if self.lang not in self.support_lang:
            print('Language not supported, will activate Translation.')
            self.stat = 'Invalid'
        self._toolchooser()

    def _toolchooser(self):
        if self.lang == 'jp':
            self.toolkit = JapaneseWorker()
        elif self.lang == 'en':
            self.toolkit = EnglishWorker()
        else:
            self.toolkit = EnglishWorker()


class JapaneseWorker:
    def __init__(self):
        self.juman_tokenizer = JumanTokenizer()
        self.bert_tokenizer = BertTokenizer(config['DEFAULT']['vocab_path'],
                                            do_basic_tokenize=False)
        self.cls_id = self.bert_tokenizer.vocab['[CLS]']
        self.mask_id = self.bert_tokenizer.vocab['[MASK]']
        self.bert_model = './model/Japanese'

        self.cp = 'checkpoint/jp/cp_step_710000.pt'
        self.opt = 'checkpoint/jp/opt_step_710000.pt'

    @staticmethod
    def linesplit(src):
        """
        :param src: type str, String type article
        :return: type list, punctuation seperated sentences
        """
        def remove_newline(x):
            x = x.replace('\n', '')
            return x

        def remove_blank(x):
            x = x.replace(' ', '')
            return x

        def remove_unknown(x):
            unknown = ['\u3000']
            for h in unknown:
                x = x.replace(h, '')
            return x
        src = remove_blank(src)
        src = remove_newline(src)
        src = remove_unknown(src)
        src_line = re.split('。(?<!」)|！(?<!」)|？(?!」)', src)
        src_line = [x for x in src_line if x is not '']
        return src_line

    def tokenizer(self, src):
        """
        :param src: type list, punctuation seperated sentences
        :return: token: type list, numberized tokens
                 token_id: type list, tokens
        """
        token = []
        token_id = []

        def _preprocess_text(text):
            return text.replace(" ", "")  # for Juman

        for sentence in src:
            preprocessed_text = _preprocess_text(sentence)
            juman_tokens = self.juman_tokenizer(preprocessed_text)
            tokens = self.bert_tokenizer.tokenize(" ".join(juman_tokens))
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            token += tokens
            token_id += ids
        return token, token_id


class EnglishWorker:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cls_id = self.bert_tokenizer.vocab['[CLS]']
        self.mask_id = self.bert_tokenizer.vocab['[MASK]']
        self.bert_model = 'bert-base-uncased'

        self.cp = 'checkpoint/en/stdict_step_300000.pt'
        self.opt = 'checkpoint/en/opt_step_300000.pt'

    @staticmethod
    def linesplit(src):
        def remove_newline(x):
            x = x.replace('\n', ' ')
            return x

        def replace_honorifics(x):
            honors = ['Mr', 'Mrs']
            for h in honors:
                x = x.replace(h + '. ', h + ' ')
            return x

        src = remove_newline(src)
        src = replace_honorifics(src)
        src_line = re.split('\.', src)
        src_line = [x for x in src_line if x is not '']
        return src_line

    def tokenizer(self, src):
        token = []
        token_id = []

        for sentence in src:
            tokens = self.bert_tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            token += tokens
            token_id += ids
        return token, token_id
