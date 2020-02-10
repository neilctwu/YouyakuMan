import re

from pyknp import Juman
from sumeval.metrics.rouge import RougeCalculator
from configparser import ConfigParser
from pytorch_pretrained_bert import BertTokenizer

config = ConfigParser()
config.read('config.ini')


class JumanTokenizer:
    def __init__(self):
        self.juman = Juman(command=config['Juman']['command'],
                           option=config['Juman']['option'])

    def __call__(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class RougeNCalc:
    def __init__(self):
        self.rouge = RougeCalculator(stopwords=True, lang="ja")

    def __call__(self, summary, reference):
        score = self.rouge.rouge_n(summary, reference, n=1)
        return score


class Preprocess:
    def __init__(self):
        self.juman_tokenizer = JumanTokenizer()
        self.rouge_calculator = RougeNCalc()
        self.bert_tokenizer = BertTokenizer(config['DEFAULT']['vocab_path'],
                                            do_lower_case=False, do_basic_tokenize=False)
        self.trim_input = 0
        self.trim_clss = 0

    def __call__(self, data_dic, length):
        self.src_title = data_dic['name']
        self.src_body = data_dic['body']
        self.src_summary = data_dic['summary'].split('<sep>')
        self._init_data()

        if self.src_body is '':
            raise ValueError('Empty data')

        # step 1. article to lines
        self._split_line()
        # step 2. pick extractive summary by rouge
        self._rougematch()
        # step 3. tokenize
        self._tokenize()
        # step 4. clss process
        self._prep_clss()
        # step 5. segs process
        self._prep_segs()
        # step 6. trim length for input
        self._set_length(length)

        return {'title': self.src_title,
                'src': self.tokenid,
                'labels': self.label,
                'segs': self.segs,
                'mask': self.mask,
                'mask_cls': self.mask_cls,
                'clss': self.clss,
                'src_str': self.src_line}

    def _init_data(self):
        self.src_line = []
        self.label = []
        self.tokenid = []
        self.token = []
        self.clss = []
        self.segs = []
        self.mask = []
        self.mask_cls = []

    # step 1.
    def _split_line(self):
        # regex note: (?!...) Negative Lookahead
        # e.g. /foo(?!bar)/ for "foobar foobaz" get "foobaz" only
        self.src_line = re.split('。(?<!」)|！(?<!」)|？(?!」)', self.src_body)
        self.src_line = [x for x in self.src_line if x is not '']

    # step 2.
    def _rougematch(self):
        self.label = [0]*len(self.src_line)
        for summ in self.src_summary:
            scores = [self.rouge_calculator(x, summ) for x in self.src_line]
            self.label[scores.index(max(scores))] = 1

    # step 3.
    def _tokenize(self):
        def _preprocess_text(text):
            return text.replace(" ", "")  # for Juman
        for sentence in self.src_line:
            preprocessed_text = _preprocess_text(sentence)
            juman_tokens = self.juman_tokenizer(preprocessed_text)
            tokens = self.bert_tokenizer.tokenize(" ".join(juman_tokens))
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            self.token += tokens
            self.tokenid += ids

    # step 4.
    def _prep_clss(self):
        self.clss = [i for i, x in enumerate(self.tokenid) if x == self.bert_tokenizer.vocab['[CLS]']]

    # step 5.
    def _prep_segs(self):
        flag = 1
        for idx in self.tokenid:
            if idx == self.bert_tokenizer.vocab['[CLS]']:
                flag = not flag
            self.segs.append(int(flag))

    # step 6.
    def _set_length(self, n):
        self.__trim_data(n)
        self.__add_mask(n)

    def __trim_data(self, n):
        if len(self.tokenid) > n:
            # If last sentence starts after 512
            if self.clss[-1] > 512:
                for i, idx in enumerate(self.clss):
                    if idx > n:
                        # Index of last [SEP] in length=n
                        self.trim_input = self.clss[i-1] - 1
                        # Index of last [CLS] index in clss
                        self.trim_clss = i - 2
                        break
            # If src longer than 512 but last sentence start < 512
            else:
                self.trim_input = self.clss[len(self.clss) - 1] - 1
                self.trim_clss = len(self.clss) - 2
        # Do nothing if length < n
        if self.trim_clss*self.trim_input == 0:
            return
        self.tokenid = self.tokenid[:(self.trim_input+1)]
        self.segs = self.segs[:(self.trim_input+1)]
        self.clss = self.clss[:(self.trim_clss+1)]
        self.label = self.label[:(self.trim_clss+1)]
        self.src_line = self.src_line[:(self.trim_clss+1)]

    def __add_mask(self, n):
        # from index to len: +1
        pad_len = (n - len(self.tokenid))
        self.tokenid = self.tokenid + ([self.bert_tokenizer.vocab['[MASK]']] * pad_len)
        self.segs = self.segs + ([int(not self.segs[-1])] * pad_len)

