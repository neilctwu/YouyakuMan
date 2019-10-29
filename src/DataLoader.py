from pytorch_pretrained_bert import BertTokenizer


class DataLoader:
    def __init__(self, path, super_long=False):
        self.path = path
        self.data = []
        self.super_long = super_long
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self._load_data()
        self._generate_results()

    def _generate_results(self):
        src_all = self._text_tolist()
        if not self.super_long:
            token_all = self._list_tokenize(src_all)
            _, _ = self._add_result(self.fname, src_all, token_all)
        else:
            # Initialize indexes for while loop
            src_start, token_start, src_end = 0, 0, 1
            token_all = self._list_tokenize(src_all)
            while src_end != 0:
                token_end, src_end = self._add_result(self.fname, src_all,
                                                      token_all, src_start, token_start)
                token_start = token_end
                src_start = src_end

    def _add_result(self, fname, src_all, token_all, src_start=0, token_start=0):
        results, (token_end, src_end) = self._all_tofixlen(src_all, token_all, src_start, token_start)
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

    def _text_tolist(self):
        def remove_newline(x):
            x = x.replace('\n', ' ')
            return x

        def replace_honorifics(x):
            honors = ['Mr', 'Mrs']
            for h in honors:
                x = x.replace(h + '. ', h + ' ')
            return x

        srcs = []
        for text in self.rawtexts:
            text = remove_newline(text)
            text = replace_honorifics(text)
            msrcs = text.split('. ')
            srcs += [x for x in msrcs if x not in ['', ' ']]
        return srcs

    def _list_tokenize(self, srcs):
        src_tokenize = []
        for src in srcs:
            src_subtokens = self.tokenizer.tokenize(src)
            src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
            src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
            src_tokenize += src_subtoken_idxs
        return src_tokenize

    def _all_tofixlen(self, src, token, src_start, token_start):
        # Tune All shit into 512 length
        # pdb.set_trace()
        token_end = 0
        src_end = 0
        token = token[token_start:]
        src = src[src_start:]
        clss = [i for i, x in enumerate(token) if x == 101]
        if len(token) > 512:
            clss, token, token_stop, src, src_stop = self._length512(src, token, clss)
            token_end = token_start + token_stop
            src_end = src_start + src_stop
        labels = [0] * len(clss)
        mask = ([True] * len(token)) + ([False] * (512 - len(token)))
        mask_cls = [True] * len(clss)
        token = token + ([0] * (512 - len(token)))
        segs = []
        flag = 1
        for idx in token:
            if idx == 101:
                flag = not flag
            segs.append(int(flag))
        return (token, clss, segs, labels, mask, mask_cls, src), (token_end, src_end)

    @staticmethod
    def _length512(src, token, clss):
        src_stop = [x > 512 for x in clss].index(True) - 1
        token_stop = clss[src_stop]
        clss = clss[:src_stop]
        src = src[:src_stop]
        token = token[:token_stop]
        return clss, token, token_stop, src, src_stop

