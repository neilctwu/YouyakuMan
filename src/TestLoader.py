from transformers import BertTokenizer

from src.LangFactory import LangFactory


class TestLoader:
    def __init__(self, path, super_long, lang, translator=None):
        self.path = path
        self.data = []
        self.super_long = super_long
        self.langfac = LangFactory(lang)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._load_data()
        # If rawdata isnt modelized, use google translation to translate to English
        if self.langfac.stat is 'Invalid':
            self.translator = translator
            self._translate()
        # Outsource suitable line splitter
        self.texts = self.langfac.toolkit.linesplit(self.rawtexts)
        # Outsource suitable tokenizer
        self.token, self.token_id = self.langfac.toolkit.tokenizer(self.texts)
        self._generate_results()

    def _generate_results(self):

        if not self.super_long:
            _, _ = self._add_result(self.fname, self.token_id)
        else:
            # Initialize indexes for while loop
            src_start, token_start, src_end = 0, 0, 1
            while src_end != 0:
                token_end, src_end = self._add_result(self.fname, self.token_id,
                                                      src_start, token_start)
                token_start = token_end
                src_start = src_end

    def _add_result(self, fname, token_all, src_start=0, token_start=0):
        results, (token_end, src_end) = self._all_tofixlen(token_all, src_start, token_start)
        token, clss, segs, labels, mask, mask_cls, src = results
        self.data.append({'fname': fname,
                          'src': token,
                          'labels': labels,
                          'segs': segs,
                          'mask': mask,
                          'mask_cls': mask_cls,
                          'clss': clss,
                          'src_str': src})
        return token_end, src_end

    def _load_data(self):
        self.fname = self.path.split('/')[-1].split('.')[0]
        with open(self.path, 'r', encoding='utf-8_sig', errors='ignore') as f:
            self.rawtexts = f.readlines()
        self.rawtexts = ' '.join(self.rawtexts)

    def _translate(self):
        texts = self.rawtexts
        self.texts = self.translator.input(texts)

    def _all_tofixlen(self, token, src_start, token_start):
        # Tune All shit into 512 length
        token_end = 0
        src_end = 0
        token = token[token_start:]
        src = self.texts[src_start:]
        clss = [i for i, x in enumerate(token) if x == self.langfac.toolkit.cls_id]
        if len(token) > 512:
            clss, token, token_stop, src, src_stop = self._length512(src, token, clss)
            token_end = token_start + token_stop
            src_end = src_start + src_stop
        labels = [0] * len(clss)
        mask = ([True] * len(token)) + ([False] * (512 - len(token)))
        mask_cls = [True] * len(clss)
        token = token + ([self.langfac.toolkit.mask_id] * (512 - len(token)))
        segs = []
        flag = 1
        for idx in token:
            if idx == self.langfac.toolkit.cls_id:
                flag = not flag
            segs.append(int(flag))
        return (token, clss, segs, labels, mask, mask_cls, src), (token_end, src_end)

    @staticmethod
    def _length512(src, token, clss):
        if max(clss) > 512:
            src_stop = [x > 512 for x in clss].index(True) - 1
        else:
            src_stop = len(clss) - 1
        token_stop = clss[src_stop]
        clss = clss[:src_stop]
        src = src[:src_stop]
        token = token[:token_stop]
        return clss, token, token_stop, src, src_stop
