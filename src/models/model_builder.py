import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.init import xavier_uniform_

from src.models.encoder import TransformerInterEncoder
from src.models.optimizers import Optimizer


def build_optim(args, model, checkpoint):
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.learning_rate, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        optim.learning_rate = args.learning_rate
        for param_group in optim.optimizer.param_groups:
            param_group['lr'] = args.learning_rate

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, bert_model):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(bert_model)

    def forward(self, x, segs, mask):
        top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
        return top_vec


class Summarizer(nn.Module):
    def __init__(self, args, bert_model, device='cpu', train=False):
        super(Summarizer, self).__init__()
        self.device = device
        self.bert = Bert(bert_model)
        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size,
                                               args.ff_size, args.heads,
                                               args.dropout, args.inter_layers)

        if train:
            if args.param_init:
                for p in self.encoder.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

            if args.param_init_glorot:
                for p in self.encoder.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        self.to(device)

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
