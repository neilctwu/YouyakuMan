from googletrans import Translator
import sys
import pdb


class TranslatorY(Translator):
    def __init__(self):
        super(TranslatorY, self).__init__()
        self.input_lang = 'en'
        self.check_lang = True

    def input(self, text):
        pdb.set_trace()
        if self.check_lang:
            self.input_lang = self.detect(text).lang
            sys.stdout.write('<Translator: Input article is wrote in [{}] language>\n'.format(self.input_lang.upper()))
            self.check_lang = False
        trans = self._translation(text, self.input_lang, 'en')
        return trans.text

    def output(self, texts_list):
        transed_text = []
        for text in texts_list:
            trans = self._translation(text, 'en', self.input_lang)
            transed_text.append(trans.text)
        return transed_text

    def _translation(self, text, input_lang, output_lang):
        return self.translate(text, src=input_lang, dest=output_lang)
