# Contrastive Distillation and Cross-Modal Mamba Fusion for Multimodal Brain Disease Diagnosis

This repository contains source code for "Contrastive Distillation and Cross-Modal Mamba Fusion for Multimodal Brain Disease Diagnosis". 

The complete code files related to this paper will be uploaded, but some time is required for organization.

## Dependencies
- python=3.8
- torchvision=0.15.0+cu118
- torch=2.0.0+cu118
- torchaudio=2.0.0+cu118
- wandb=0.17.2
- scikit-learn=1.3.2
- pandas=2.0.3
- mamba-ssm 2.2.2

## Usage
Run the following command to train the model.

`python -m source --multirun datasz=100p model=CDCo-MambaNet dataset=UCLA repeat_time=5 preprocess=mixup`
