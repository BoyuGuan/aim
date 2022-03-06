from operator import mod
import torch
from torchvision import transforms

from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.load('./preTrainedModel/pre_trained_resnet18.pth')
model.to('cpu')
model.eval()
example = torch.rand((1,3,32,32))
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)

traced_script_module_optimized._save_for_lite_interpreter("resnet18_original_model.ptl")
