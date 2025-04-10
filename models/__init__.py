from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .CDCo_MambaNet import CDCo_MambaNet


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
