import argparse
import copy
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple
from torchvision import models
import torch
import torch.utils.data as torch_data
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# imports for AIMET
import aimet_common
from aimet_common.defs import QuantScheme
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel

# imports for data pipelines
from common import image_net_config
from utils.image_net_data_loader import ImageNetDataLoader
from utils.image_net_evaluator import ImageNetEvaluator

logger = logging.getLogger('TorchAdaround')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


###
# This script utilizes AIMET to apply Adaround on a resnet18 pretrained model with
# the ImageNet data set. It should re-create the same performance numbers as
# published in the AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET quantization accuracy using simulation model
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - default_output_bw: 8, default_param_bw: 8
#       - Encoding compution with or without encodings file
#       - Encoding computation using 5 batches of data
#    - AIMET Adaround
#       - num of batches for adarounding: 5
#       - bitwidth for quantizing layer parameters: 4
#       - Quant Scheme: 'tf_enhanced'
#       - Remaining Parameters: default
#    - Input shape: [1, 3, 224, 224]
###

class ImageNetDataPipeline:
    """
    Provides APIs for model quantization using evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.

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


def apply_adaround_and_find_quantized_accuracy(model: torch.nn.Module, evaluator: aimet_common.defs.EvalFunction,
                                               data_loader: torch_data.DataLoader, use_cuda: bool = False,
                                               logdir: str = '', bit_width:int = 8) -> Tuple[torch.nn.Module, float]:
    """
    Quantizes the model using AIMET's adaround feature, and saves the model.

    :param model: The loaded model
    :param evaluator: The Eval function to use for evaluation
    :param dataloader: The dataloader to be passed into the AdaroundParameters api
    :param num_val_samples_per_class: No of samples to use from every class in
                                      computing encodings. Not used in pascal voc
                                      dataset
    :param use_cuda: The cuda device.
    :param logdir: Path to a directory for logging.
    :param bit_width: bit width after quantize
    :return: A tuple of quantized model and accuracy of model on this quantsim
    """

    bn_folded_model = copy.deepcopy(model)
    _ = fold_all_batch_norms(bn_folded_model, input_shapes=(1, 3, 32, 32))

    input_shape = (1, image_net_config.dataset['image_channels'],
                   image_net_config.dataset['image_width'],
                   image_net_config.dataset['image_height'],)
    if use_cuda:
        dummy_input = torch.rand(input_shape).cuda()

    else:
        dummy_input = torch.rand(input_shape)

    # Number of batches to use for computing encodings
    # Only 20 batches are used here to speed up the process, also the
    # number of images in these 20 batches should be sufficient for
    # compute encodings
    iterations = 10000

    params = AdaroundParameters(data_loader=data_loader, num_batches=16)

    timeBeforeAdaRound = time.time()
    ada_model = Adaround.apply_adaround(bn_folded_model, dummy_input, params,
                                        path=logdir, filename_prefix='adaround', default_param_bw= bit_width,
                                        default_quant_scheme=QuantScheme.post_training_tf_enhanced)
    torch.save(ada_model, os.path.join(logdir, 'adamodel.pth'))

    timeAfterAdaRound = time.time()
    logger.info('Time of adaround is ' + str(timeAfterAdaRound - timeBeforeAdaRound))

    quantsim = QuantizationSimModel(model=ada_model, dummy_input=dummy_input,
                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest', default_output_bw=bit_width, default_param_bw=bit_width,
                                    in_place=False)

    # Set and freeze parameter encodings. These encodings are associated with the Adarounded parameters.
    # This will make sure compute_encodings() doesn't alter the parameter encodings.
    quantsim.set_and_freeze_param_encodings(encoding_path=os.path.join(logdir, 'adaround.encodings'))
    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)
    quantsim.export(path=logdir, filename_prefix='adaround_resnet', dummy_input=dummy_input.cpu())
    torch.save(quantsim.model, os.path.join(logdir, 'test.pth'))

    beginEvalTimeAfterQuantization = time.time()
    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)
    endEvalTimeAfterQuantization = time.time()
    logger.info('After Quantization the eval time is ' + str(endEvalTimeAfterQuantization - beginEvalTimeAfterQuantization))

    return accuracy


def adaround_example(config: argparse.Namespace):
    """
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE and BC
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Applies AIMET BC and calculates QuantSim accuracy

    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should conatin files starting with:
                                'train*': for training records and 'validation*': for validation records
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
    """

    # Instantiate Data Pipeline for evaluation
    data_pipeline = ImageNetDataPipeline(config)

    # Load the config model
    modelNames = ['resnet18', 'resnet50', 'mobilenet_v2']
    model = torch.load('./preTrainedModel/' + modelNames[_config.model] + '.pth')

    if config.use_cuda:
        model.to(torch.device('cuda'))
    model = model.eval()

    # Calculate FP32 accuracy
    beginEvalTimeBeforeQuantization = time.time()
    accuracy = data_pipeline.evaluate(model, use_cuda=config.use_cuda)
    endEvalTimeBeforeQuantization = time.time()
    logger.info('Before Quantization the eval time is ' + str(endEvalTimeBeforeQuantization - beginEvalTimeBeforeQuantization))


    logger.info("Original Model top-1 accuracy = %.2f", accuracy)
    logger.info("Applying Adaround")

    # Applying Adaround
    # Optimally rounds the parameters of the model
    data_loader = ImageNetDataLoader(is_training=False, images_dir=config.dataset_dir,
                                     image_size=image_net_config.dataset['image_size']).data_loader
    accuracy = apply_adaround_and_find_quantized_accuracy(model=model, evaluator=data_pipeline.evaluate,
                                                          data_loader=data_loader, use_cuda=config.use_cuda,
                                                          logdir=config.logdir, bit_width = config.bit_width)

    logger.info("After applying Adaround, top-1 accuracy = %.2f", accuracy)
    logger.info("Adaround Complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Apply Adaround on pretrained ResNet18 model and evaluate on ImageNet dataset')

    parser.add_argument('--dataset_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should conatin at least 2 subfolders:\n\
                              'train': for training dataset and 'val': for validation dataset")
    parser.add_argument('--use_cuda', action='store_true', required=True,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default='./',
                        help="Path to a directory for logging.\
                        you don't need to give it a value, whatever you input, it will be 'benchmark_output/model_name_adaround_<Y-m-d-H-M-S>'")
    
    parser.add_argument('--model', type=int, required=True,
                help="The model you want to quantize, \n\
                    0 means resnet18, \n\
                    1 means resnet50, \n\
                    2 means mobilenet_v2")

    parser.add_argument('--bit_width', type=int, required=False, default = 8,
            help="The bit width you want to quantize,  4, 8(default), 16")

    _config = parser.parse_args()

    modelNames = ['resnet18', 'resnet50', 'mobilenet_v2']
    _config.logdir = os.path.join("benchmark_output", 'adaround', modelNames[_config.model] +  "_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, modelNames[_config.model]+"_adaround.log"))
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

    adaround_example(_config)
