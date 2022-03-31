import pickle
import os
from PIL import Image

import numpy as np

def unpickle(file):
    # 返回的dict对象有这几个key dict_keys([b'batch_label', b'labels', b'data', b'filenames'])，每个batch 10000张图
    # b'batch_label' 返回b'training batch 1 of 5' 
    # b'labels' 返回list格式的标签    
    # b'data'返回numpy格式 (10000, 3072)维的数据
    # b'filenames'返回list格式的文件名称
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def handleCifar(dataSetDir):
    count = 0
    num2label = unpickle(os.path.join(dataSetDir,'batches.meta'))[b'label_names']
    for batches in os.listdir(dataSetDir):
        subSet = None
        if 'test' in batches :
            subSet = 'val'
        elif 'data' in batches: 
            subSet = 'train'
        else :
            continue
        batch = unpickle(os.path.join(dataSetDir,batches))
        pics = batch[b'data']
        picLabel = batch[b'labels']
        for imgNO in range(10000):
            picOrigin = np.array(pics[imgNO].reshape(3,-1))
            pic = np.zeros((32,32,3),dtype=np.uint8)
            imgLabel = num2label[picLabel[imgNO]].decode('UTF-8')
            for i in range(32):
                for j in range(32):
                    pic[i,j] = [picOrigin[0,i*32+j],picOrigin[1,i*32+j],picOrigin[2,i*32+j]]
            img = Image.fromarray(pic,'RGB')
            os.makedirs(os.path.join('./cifar',subSet,imgLabel),exist_ok=True)
            nowNumberInThisClass = len(os.listdir(os.path.join('./cifar',subSet,imgLabel)))
            img.save(os.path.join('./cifar',subSet,imgLabel,imgLabel+'_'+str(nowNumberInThisClass)+'.jpg'))
            count += 1
            print(count, os.path.join('./cifar',subSet,imgLabel,imgLabel+'_'+str(nowNumberInThisClass)+'.jpg'))



if __name__ == '__main__':
    dataSetDir = '../temp/cifar-10-batches-py'
    handleCifar(dataSetDir)