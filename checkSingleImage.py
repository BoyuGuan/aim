import torch
from PIL import Image
from torchvision import transforms
from torchvision import models

# model = torch.load('./benchmark_output/resNet18_channel_prunning_2022-02-27-18-41-03/compressed_model_finetuned.pth')
model = torch.load('./benchmark_output/resNet18_channel_prunning_2022-02-27-18-41-03/compressed_model_finetuned.pth')
model.eval()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

for i in range (100):

    image  = Image.open(f'./cifar/val/airplane/airplane_{i}.jpg')
    # to tensor
    image = val_transforms(image).cuda()
    image = torch.unsqueeze(image,0)
    out = model(image)
    _, indices = torch.sort(out, descending=True)
    # _, indices = torch.sort(out, descending=True)
    print(f'\n{indices[0][0]}\n')