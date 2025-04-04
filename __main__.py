from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from dataset import dataset_factory
from models import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import numpy as np


def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)
    model = model_factory(cfg)

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)

    max_process = training.train()
    return max_process

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    acc, spe, sen, f1, auc = [], [], [], [], []
    for _ in range(cfg.repeat_time):
        run = wandb.init(reinit=True, tags=[f"{cfg.dataset.name}"])
        max_process = model_training(cfg)
        acc.append(max_process[0])
        spe.append(max_process[1])
        sen.append(max_process[2])
        auc.append(max_process[3])
        f1.append(max_process[4])
        run.finish()
    print(f"ACC: {np.mean(acc)*100:.2f}±{np.std(acc, ddof=1)*100:.2f}")
    print(f"SPE: {np.mean(spe)*100:.2f}±{np.std(spe, ddof=1)*100:.2f}")
    print(f"SEN: {np.mean(sen)*100:.2f}±{np.std(sen, ddof=1)*100:.2f}")
    print(f"AUC: {np.mean(auc)*100:.2f}±{np.std(auc, ddof=1)*100:.2f}")
    print(f"F1: {np.mean(f1)*100:.2f}±{np.std(f1, ddof=1)*100:.2f}")


if __name__ == '__main__':
    main()
