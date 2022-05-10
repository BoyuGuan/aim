import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.load('./preTrainedModel/resnet18_1.pth')
model.to('cpu')
model.eval()
example = torch.rand((1,3,32,32))
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)

traced_script_module_optimized._save_for_lite_interpreter("./18_1.pt")
