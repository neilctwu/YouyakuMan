import torch

from src.utils.utils import DictX
from src.models.model_builder import Summarizer


class ModelLoader(Summarizer):
    def __init__(self, cp, opt, bert_model):
        cp_statedict = torch.load(cp, map_location=lambda storage, loc: storage)
        opt = DictX(torch.load(opt))
        super(ModelLoader, self).__init__(opt, bert_model)
        self.load_cp(cp_statedict)
        self.eval()
