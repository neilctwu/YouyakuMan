import random
import torch
import os

import argparse

from src.models.trainloader import TrainLoader
from src.models.model_builder import Summarizer, build_optim
from src.models.trainer import build_trainer

os.chdir('./')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_folder", default='', type=str)
    parser.add_argument("-batch_size", default=5, type=int)
    parser.add_argument("-train_from", default='', type=str)
    parser.add_argument("-save_path", default='', type=str)

    parser.add_argument("-train_steps", default=1200000, type=int)
    parser.add_argument("-report_every", default=10, type=int)
    parser.add_argument("-save_checkpoint_steps", default=1000, type=int)
    parser.add_argument("-accum_count", default=2, type=int)

    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-learning_rate", default=5e-5, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='no', type=str)
    parser.add_argument("-warmup_steps", default=10000, type=int)

    parser.add_argument("-ff_size", default=2048, type=int)
    parser.add_argument("-heads", default=8, type=int)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-param_init", default=0.0, type=float)
    parser.add_argument("-param_init_glorot", default=True, type=bool)
    parser.add_argument("-max_grad_norm", default=0, type=float)
    parser.add_argument("-inter_layers", default=2, type=int)

    parser.add_argument("-seed", default='')

    args = parser.parse_args()

    model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

    device = "cuda"
    device_id = -1

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    def train_loader_fct():
        return TrainLoader(args.data_folder, 512, args.batch_size, device=device, shuffle=True)

    model = Summarizer(args, './model/Japanese/', device, train=True)
    if args.train_from != '':
        print('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = dict(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        model.load_cp(checkpoint['model'])
        optim = build_optim(args, model, checkpoint)
    else:
        optim = build_optim(args, model, None)

    trainer = build_trainer(args, model, optim)
    trainer.train(train_loader_fct, args.train_steps)

