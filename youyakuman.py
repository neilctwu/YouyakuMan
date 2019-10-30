import pdb
import argparse
from argparse import RawTextHelpFormatter

from src.DataLoader import DataLoader
from src.ModelLoader import ModelLoader
from src.Summarizer import Summarizer
from src.Translator import TranslatorY

cp = 'checkpoint/stdict_step_300000.pt'
opt = 'checkpoint/opt_step_300000.pt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description="""
    Intro:   This is an one-touch extractive summarization machine.
             using BertSum as summatization model, extract top N important sentences.
             
    Note:    Since Bert only takes 512 length as inputs, this summarizer crop articles >512 length. 
             If --super_long option is used, summarizer automatically parse to numbers of 512 length
             inputs and summarize per inputs. Number of extraction might slightly altered with --super_long used.
    
    Example: youyakuman.py -txt_file YOUR_FILE -n 3
    """)

    parser.add_argument("-txt_file", default='test.txt',
                        help='Text file for summarization (encoding:"utf-8_sig")')
    parser.add_argument("-n", default=3, type=int,
                        help='Numbers of extraction summaries')
    parser.add_argument("--super_long", action='store_true',
                        help='If length of article >512, this option is needed')
    parser.add_argument("--not_en", action='store_true',
                        help='If language of article isn\'t Englisth, will automatically translate by google')

    args = parser.parse_args()
    if args.super_long:
        print('\n<Warning: Number of extractions might slightly altered since with --super_long option>\n')

    translator = TranslatorY() if args.not_en else None
    data = DataLoader(args.txt_file, args.super_long, translator).data
    model = ModelLoader(cp, opt)
    summarizer = Summarizer(data, model, args.n, translator)