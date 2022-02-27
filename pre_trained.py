import os
import argparse

import torch
from torchvision import models

# imports for data pipelines
from common import image_net_config
from utils.image_net_evaluator import ImageNetEvaluator
from utils.image_net_trainer import ImageNetTrainer

def preTrain(_config: argparse.Namespace, model: torch.nn.Module):
    """pre-train a model and then save that.

    Args:
        _config (argparse.Namespace): the pre-train config.
        model (torch.nn.Module): the model that want to be pre trained.

    Returns:
        model (torch.nn.Module): the model after pre-trained.
    """
    trainer = ImageNetTrainer(_config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                        batch_size=image_net_config.train['batch_size'],
                        num_workers=image_net_config.train['num_workers'])
    trainer.train(model, max_epochs = _config.epoches, learning_rate=_config.learning_rate,
            learning_rate_schedule=_config.learning_rate_schedule, use_cuda=_config.use_cuda)
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-train args')
    parser.add_argument('--dataset_dir', type=str, required=True, 
        help="Path to a directory containing ImageNet dataset.\n\
            This folder should conatin at least 2 subfolders:\n'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true', required=True, help='Add this flag to run the test on GPU.')
    parser.add_argument('--epoches', type=int, default=50, help='Number of epoches pretrained')
    parser.add_argument('--model_dir', type=str, default='./preTrainedModel', help='Number of epoches pretrained')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help="A float type learning rate for model finetuning.\n  Default is 0.01")
    parser.add_argument('--learning_rate_schedule', type=list, default=[5, 10],
                        help="A list of epoch indices for learning rate schedule used in finetuning.\n\
                                Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
                                Default is [5, 10]")

    _config = parser.parse_args()
    os.makedirs(_config.model_dir, exist_ok=True)
    if _config.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    resNet18 = models.resnet18()
    resNet50 = models.resnet50()
    mobileNetV2 = models.mobilenet_v2()
    myModels = [resNet18, resNet50, mobileNetV2]
    modelName  = ['resNet18', 'resNet50', 'mobileNetv2']
    for i, model in enumerate(myModels):
        model = preTrain(_config, model)
        torch.save(model, os.path.join(_config.model_dir, 'pre_trained_' + modelName[i] + '.pth'))