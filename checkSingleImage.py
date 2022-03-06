from select import select
from turtle import forward
import torch
from PIL import Image
from torchvision import transforms
# from torchvision import models

# model = torch.load('./benchmark_output/resNet18_channel_prunning_2022-02-27-18-41-03/compressed_model_finetuned.pth')
model = torch.load('./benchmark_output/resNet18_channel_prunning_2022-02-27-18-41-03/compressed_model_finetuned.pth')
model.eval()
model.to("cpu")

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])


count = 0
for i in range (100):
    
    image  = Image.open(f'./cifar/val/bird/bird_{i}.jpg')
    # to tensor
    image = val_transforms(image)
    image = torch.unsqueeze(image,0)
    out = model(image)
    _, indices = torch.sort(out, descending=True)
    if indices[0][0] == 2:
        count += 1 
    # print(f'\n{indices[0][0]}\n')
print(f'\n {count}\n\n')
