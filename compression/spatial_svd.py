# !/usr/bin/env python
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""
This file demonstrates the use of compression using AIMET spatial SVD
technique followed by fine tuning.
"""

import argparse
import logging
import os
from datetime import datetime
from decimal import Decimal
from torchvision import models
import torch
import time

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# imports for AIMET
import aimet_common.defs
from aimet_common.defs import CompressionScheme
from aimet_common.defs import CostMetric
import aimet_torch
from aimet_torch.compress import ModelCompressor

# imports for data pipelines
from common import image_net_config
from utils.image_net_evaluator import ImageNetEvaluator
from utils.image_net_trainer import ImageNetTrainer

logger = logging.getLogger('myCompression')
logger.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')



###
# data_pipelines_pytorch is a package internally developed by the CR&D Morpheus team.
# It provides utilities with which you can easily create training pipelines using
# the PyTorch framework. It also provides visualization of computational graph meta-data like
# number of parameters involved and MAC count estimate.

# This script utilize AIMET do perform spatial svd compression (50% compression ratio) on a resnet18 pretrained model
# with the ImageNet data set. It should re-create the same performance numbers as published in the
# AIMET release for the particular scenario as described below.
#
# Scenario parameters:
#    - AIMET Spatial SVD compression using auto mode
#    - Ignored model.conv1 (this is the first layer of the model)
#    - Target compression ratio: 0.5 (or 50%)
#    - Number of compression ration candidates: 10
#    - Input shape: [1, 3, 224, 224]
#    - Learning rate: 0.01
#    - Learning rate schedule: [5,10]
###

class ImageNetDataPipeline:
    """
    Provides APIs for model compression using AIMET weight SVD, evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.
        AIMET's compress_model() expects the function with this signature to its eval_callback
        parameter.

        :param model: The model to be evaluated.
        :param iterations: The number of batches of the dataset.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """

        # your code goes here instead of the example from below
        evaluator = ImageNetEvaluator(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                      batch_size=image_net_config.evaluation['batch_size'],
                                      num_workers=image_net_config.evaluation['num_workers'])
        return evaluator.evaluate(model, iterations, use_cuda)
    

    def finetune(self, model: torch.nn.Module):
        """
        Finetunes the model.  The implemtation provided here is just an example,
        provide your own implementation if needed.

        :param model: The model to finetune.
        :return: None
        """

        # Your code goes here instead of the example from below

        trainer = ImageNetTrainer(self._config.dataset_dir, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])
        trainer.train(model, max_epochs=self._config.epochs, learning_rate=self._config.learning_rate,
                      learning_rate_schedule=self._config.learning_rate_schedule, use_cuda=self._config.use_cuda)
        torch.save(model, os.path.join(self._config.logdir, 'svd_finetuned_model.pth'))
        
        # Calculate and log the accuracy of compressed-finetuned model
        accuracy = self.evaluate(model, use_cuda=self._config.use_cuda)
        logger.info("Finetuned Compressed Model top-1 accuracy = %.2f", accuracy)
        logger.info("Model Finetuning Complete")

def aimet_spatial_svd(model: torch.nn.Module,compressionRatio: float, metric_mac: bool,
                      evaluator: aimet_common.defs.EvalFunction):
    """
    Compresses the model using AIMET's Spatial SVD auto mode compression scheme.

    :param model: The model to compress
    :param compressionRatio: compression ratio of teh model 
    :param metric_mac: whether use mac as the metric, True is mac, False is mem 
    :param evaluator: Evaluator used during compression
    :param data_loader: DataLoader used during compression
    :return: A tuple of compressed model and its statistics
    """

    # create the parameters for AIMET to compress on auto mode.
    # please refer to the API documentation for other schemes (i.e weight svd & channel prunning)
    # and mode (manual)
    logger.info('compression ratio is %.4f' , compressionRatio)

    greedy_params = aimet_torch.defs.GreedySelectionParameters(target_comp_ratio=Decimal(compressionRatio),
                                                               num_comp_ratio_candidates=10)
    auto_params = aimet_torch.defs.SpatialSvdParameters.AutoModeParams(greedy_params,
                                                                       modules_to_ignore=[model.conv1])
    params = aimet_torch.defs.SpatialSvdParameters(aimet_torch.defs.SpatialSvdParameters.Mode.auto,
                                                   auto_params)

    scheme = CompressionScheme.spatial_svd

    metric = None
    if metric_mac:
        metric = aimet_common.defs.CostMetric.mac  # mac or memory
        logger.info('the metric is mac')
    else:
        metric = aimet_common.defs.CostMetric.memory  
        logger.info('the metric is memory')
    beginCompressionTime = time.time()
    results = ModelCompressor.compress_model(model=model,
                                             eval_callback=evaluator,
                                             eval_iterations=10,
                                             input_shape=(1, 3, 32, 32),
                                             compress_scheme=scheme,
                                             cost_metric=metric,
                                             parameters=params)
    endCompressionTime = time.time()
    logger.info('cost time to compression is ' + str(endCompressionTime - beginCompressionTime))

    return results



def spatial_svd_example(config: argparse.Namespace):
    """
    1. Instantiate Data Pipeline for evaluation and training
    2. Load the pretrained resnet18 model
    3. Calculate floating point accuracy
    4. Compression
        4.1. Compress the model using AIMET Spatial SVD
        4.2. Log the statistics
        4.3. Save the compressed model
        4.4. Calculate and log the accuracy of compressed model
    5. Finetuning
        5.1 Finetune the compressed model
        5.2 Calculate and log the accuracy of compressed-finetuned model

    :param config: This argparse.Namespace config expects following parameters:
                   dataset_dir: Path to a directory containing ImageNet dataset.
                                This folder should conatin at least 2 subfolders:
                                'train': for training dataset and 'val': for validation dataset.
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
                   epochs: Number of epochs (type int) for finetuning.
                   learning_rate: A float type learning rate for model finetuning
                   learning_rate_schedule: A list of epoch indices for learning rate schedule used in finetuning. Check
                                           https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR
                                           for more details.
    """
    # Instantiate Data Pipeline for evaluation and training
    data_pipeline = ImageNetDataPipeline(config)
    # Loads the pretrained resnet18 model
    
    #???????????? 
    modelNames = ['resnet18', 'resnet50', 'vgg19', 'mobilenetv2']
    model = torch.load('./preTrainedModel/' + modelNames[config.model] + '.pth')
    logger.info("This is my"+ modelNames[config.model] +" model")

    # #??????fc????????????????????? 
    # fc_features = model.fc.in_features 
    # #???????????????10 
    # model.fc = torch.nn.Linear(fc_features, 10)

    if config.use_cuda:
        model.to(torch.device('cuda'))
    model.eval()

    # Calculates floating point accuracy
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)

    logger.info("Original Model top-1 accuracy = %.2f", accuracy)

    # Compress the model using AIMET Weight SVD
    logger.info("Starting Spatial SVD")
    compressed_model, stats = aimet_spatial_svd(model=model, compressionRatio=config.compression_ratio,
                                                metric_mac=config.metric_mac,
                                                evaluator=data_pipeline.evaluate)
    logger.info(stats)

    # Calculate and log the accuracy of compressed model
    accuracy = data_pipeline.evaluate(compressed_model, use_cuda=config.use_cuda)
    logger.info("After SVD, Model Top-1 accuracy = %.2f", accuracy)

    logger.info("Spatial SVD Complete")

    torch.save(compressed_model, os.path.join(config.logdir,  "spatial_svd_" +\
            modelNames[config.model]+'_metricMac_'+str(config.metric_mac)+'_'+\
                str(config.compression_ratio)+ '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_compressed_model_not_finetuned.pth'))


    if config.epochs:
        # Finetune the compressed model
        logger.info("Starting Model Finetuning")
        data_pipeline.finetune(compressed_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply Spatial SVD on pretrained ResNet18 model and finetune it for ImageNet dataset')

    parser.print_help()

    parser.add_argument('--dataset_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should conatin at least 2 subfolders:\n\
                              'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true',
                        required=True,
                        help='Add this flag to run the test on GPU.')
    parser.add_argument('--metric_mac', action='store_true',
                        help='wheter use mac as metric, default is False.')
    parser.add_argument('--compression_ratio', type=float,
                        default=0.5,
                        help='compression ratrio of the model, default is 0.5.')
    parser.add_argument('--epochs', type=int,
                        default=80,
                        help="Number of epochs for finetuning.\n\
                              Default is 80")
    parser.add_argument('--learning_rate', type=float,
                        default=0.1,
                        help="A float type learning rate for model finetuning.\n\
                              Default is 0.1")
    parser.add_argument('--learning_rate_schedule', type=list,
                        default=[30, 60 ],
                        help="A list of epoch indices for learning rate schedule used in finetuning.\n\
                              Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR for more details.\n\
                              Default is [30, 60 ]]")
    parser.add_argument('--model', type=int, required=True,
                    help="The model you want to compress, \n\
                        0 means resnet18, \n\
                        1 means resnet50, \n\
                        2 means vgg19, \n\
                        3 means mobilenetv2")
    parser.add_argument('--logdir', type=str, default='./',
                    help="Path to a directory for logging.\
                        you don't need to give it a value, whatever you input, it will be 'benchmark_output/model_channel_pruning_<Y-m-d-H-M-S>'")

    _config = parser.parse_args()
    modelNames = ['resnet18', 'resnet50', 'vgg19', 'mobilenetv2']

    _config.logdir = os.path.join("benchmark_output", "spatial_SVD", modelNames[_config.model] +  "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, modelNames[_config.model]+'_SVD_metricMac_'+str(_config.metric_mac)+'_'+str(_config.compression_ratio)+'_' +".log"))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")
    spatial_svd_example(_config)
