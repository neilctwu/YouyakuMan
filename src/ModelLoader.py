import torch
from torch import nn
from pytorch_pretrained_bert import BertConfig

from src.models.model_builder import Bert
from src.models.encoder import TransformerInterEncoder


class Summarizer(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(Summarizer, self).__init__()
        self.bert = Bert('../model', False, bert_config)
        self.encoder = TransformerInterEncoder(self.bert.model.config.hidden_size,
                                               opt['ff_size'],
                                               opt['heads'],
                                               opt['dropout'],
                                               opt['inter_layers'])

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


class ModelLoader(Summarizer):
    def __init__(self, cp, opt):
        config = BertConfig.from_json_file('model/bert_config_uncased_base.json')
        cp_statedict = torch.load(cp, map_location=lambda storage, loc: storage)
        opt = vars(torch.load(opt))
        super(ModelLoader, self).__init__(opt, bert_config=config)
        self.load_cp(cp_statedict)
        self.eval()
