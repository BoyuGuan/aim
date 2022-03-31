import os
import argparse
import logging
from datetime import datetime


import torch
from torchvision import models

# imports for data pipelines
from common import image_net_config
from utils.image_net_trainer import ImageNetTrainer


logger = logging.getLogger()
logger.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


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
    parser = argparse.ArgumentParser(description='pre-train args, to pre-train the model')
    parser.add_argument('--dataset_dir', type=str, required=True, 
        help="Path to a directory containing ImageNet dataset.\n\
            This folder should conatin at least 2 subfolders:\n'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true', required=True, help='Add this flag to run the test on GPU.')
    parser.add_argument('--epoches', type=int, default=100, help='Number of epoches pretrained')
    parser.add_argument('--model_dir', type=str, default='./preTrainedModel', help='Number of epoches pretrained')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                    help="A float type learning rate for model finetuning.\n  Default is 0.1")
    parser.add_argument('--learning_rate_schedule', type=list, default=[40, 70],
                        help="A list of epoch indices for learning rate schedule used in finetuning.\n\
                                Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
                                Default is [40, 70]")

    _config = parser.parse_args()
    _config.logdir = os.path.join("pretrian_benchmark_output", "resnet50" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "pre_train_resnet50.log"))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)

    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    model = models.resnet50()
    model = preTrain(_config, model)

    torch.save(model, os.path.join(_config.model_dir, 'test_pre_trained_resnet50.pth'))