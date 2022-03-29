import torch
import torch.distributed as dist
import os
import torch.nn as nn
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys

"""
输入的4个参数依次是 rank, world_size, master, local_rank
"""

rank = int(sys.argv[1])
local_rank = int(sys.argv[4])
world_size = int(sys.argv[2])
print(rank, local_rank, world_size, sys.argv[3])

dist.init_process_group(backend="nccl",
                        init_method="tcp://" + sys.argv[3],
                        rank=rank,
                        world_size=world_size)
print('ok')

def cifar_transform():
    # transform_list = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                         transforms.Pad(4, padding_mode='reflect'),
    #                                         transforms.RandomCrop(32, padding=0),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                                             (0.2023, 0.1994, 0.2010))])

    transform_list = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_list


dataset = torchvision.datasets.CIFAR10
trainset = dataset(root='data', train=True, download=True, transform=cifar_transform())
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=64,  # 这个batch size不需要手动切分
                                            shuffle=False,
                                            sampler=train_sampler)

testset = dataset(root='data', train=False,
                      transform=cifar_transform())
test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=64,
                                              shuffle=False)
resnet_50 = torchvision.models.resnet50()
resnet_50.load_state_dict(torch.load('./resnet50-19c8e357.pth'))

model = resnet_50.to(local_rank)
# vgg_19 = torchvision.models.vgg19(pretrained = False)

# vgg_19.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
# model = vgg_19.to(local_rank)
print("■", flush=True)
ddp_model = DDP(model, device_ids=[local_rank],bucket_cap_mb=25)
if torch.__version__.startswith("1.9."):
    import ddp_comm_hooks_new
    ddp_comm_hooks_new.default_hooks.bucket_number = 5
    ddp_comm_hooks_new.default_hooks.init_my_group()
    ddp_model.register_comm_hook(state=None, hook=ddp_comm_hooks_new.default_hooks.allreduce_hook)
elif torch.__version__.startswith("1.10."):
    import ddp_comm_hooks_110
    ddp_comm_hooks_110.default_hooks.init_my_group()
    ddp_model.register_comm_hook(state=None, hook=ddp_comm_hooks_110.default_hooks.allreduce_hook)
else:
    import ddp_comm_hooks
    ddp_comm_hooks.default_hooks.init_my_group_for_global_topk5()  # todo !!!要改的地方在这
    ddp_comm_hooks.default_hooks.init_layer_count(cfg.par_layer_count * cfg.local_rank)  # 树总共有几层×gpu rank
    ddp_model.register_comm_hook(state=None, hook=ddp_comm_hooks.default_hooks.allreduce_hook)  # 

loss_fn = torch.nn.CrossEntropyLoss()
print("★", flush=True)
optimizer = optim.SGD(ddp_model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)


print("◆", flush=True)
import time
facc = open('cifar_vgg19_bucketdrop_accuracy.txt','w')
floss = open('cifar_vgg19_bucketdrop_loss.txt','w')
testtime = 0
def test(epoch):
    global testtime
    t = time.time()
    ddp_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = ddp_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        facc.write(str(acc )+'\n' )
        facc.flush()
        print('epoch ',epoch,':',acc)
    testtime += time.time() - t
    return

t = time.time()

for e in range(0, 300):
    # create model and move it to GPU with id rank
    #from nets.cifar_vgg import vgg16
    #from nets.imgnet_resnet import resnet50
    # model = torchvision.models.vgg16().to(0)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #if batch_idx > 5: break
        #print('batch index=%s' % batch_idx)
        inputs, targets = inputs.to(local_rank), targets.to(local_rank)
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        floss.write(str(loss.item() )+'\n' )
        print('loss:',loss.item())
    test(e)
    floss.flush()
facc.close()
floss.close()
print(time.time()- t - testtime)
torch.save(model, 'vgg19-bucketdrop.pth')

if torch.__version__.startswith("1.10."):
    ddp_comm_hooks_110.default_hooks.finish()
elif torch.__version__.startswith("1.9."):
    ddp_comm_hooks_new.default_hooks.finish()
else:
    ddp_comm_hooks.default_hooks.finish()
