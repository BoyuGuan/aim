import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PLOTDIR = './handleExperimentData/plotImages'
os.makedirs(PLOTDIR, exist_ok=True)

def channel_resnet18():
    MACRatio = np.array([0.302908, 0.545076, 0.649888, 0.665484, 0.197297, 0.399185, 0.473236, 0.621336])
    memoryRatio = np.array([0.198605, 0.241760, 0.296652, 0.389077, 0.160678, 0.210540,  0.226964, 0.262622])
    accBeforeFinetune = np.array([43.869427, 74.343153, 84.056529, 83.588774, 29.319268, 57.314889, 48.029459, 82.931927])
    accAfterFinetune = np.array([86.17, 88.37, 88.66, 88.48, 84.45, 87.42, 87.82, 88.63])
    costTime = np.array([ 863.702305316925, 817.5897779464722,  839.0135326385498, 823.8756124973297, 858.6545453071594, 769.8132638931274, 862.3200585842133, 808.2758123874664])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/channel_resnet18.png')


def channel_resnet50():
    MACRatio = np.array([0.392186, 0.196126, 0.405257, 0.472652, 0.192403, 0.387988, 0.405073, 0.668017])
    memoryRatio = np.array([0.272202, 0.230461, 0.278040, 0.303741, 0.231501, 0.273090,  0.271780, 0.747886])
    accBeforeFinetune = np.array([49.512341, 20.292596, 76.761545, 85.061704, 18.083201, 76.144506, 77.756768, 73.278264])
    accAfterFinetune = np.array([89.37, 86.51, 88.92, 88.48, 87.63, 89.04, 88.78, 89.49])
    costTime = np.array([ 3309.1602478027344, 3944.231453895569,  3270.3850288391113, 3158.2146010398865, 4006.2553544044495, 3466.395381450653, 3259.653082847595, 2472.8123886585236])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/channel_resnet50.png')

def weightSVD_resnet18():
    MACRatio = np.array([0.471318, 0.156471, 0.282467, 0.461664])
    memoryRatio = np.array([0.174902, 0.098078, 0.112759, 0.165290])
    accBeforeFinetune = np.array([78.025478, 21.726, 36.276, 73.378])
    accAfterFinetune = np.array([87.99, 84.92, 87.0, 88.34])
    costTime = np.array([ 246.68, 248.4190,  245.428, 245.5164])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate( MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/weightSVD_resnet18.png')

def weightSVD_resnet50():
    MACRatio = np.array([0.396226, 0.175211, 0.365779])
    memoryRatio = np.array([0.142889, 0.096962, 0.130151])
    accBeforeFinetune = np.array([69.804936, 53.642516, 70.23288])
    accAfterFinetune = np.array([88.15, 86.81, 88.82])
    costTime = np.array([ 754.1109130382538, 747.9583733081818, 744.1413459777832])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate( MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/weightSVD_resnet50.png')

def spatialSVD_resnet18():
    MACRatio = np.array([0.411991, 0.197031, 0.338500, 0.401594, 0.790962])
    memoryRatio = np.array([0.178526, 0.137202, 0.151192, 0.164245, 0.955607])
    accBeforeFinetune = np.array([60.658838, 31.329618, 40.615048, 43.033439, 50.587182])
    accAfterFinetune = np.array([87.28, 86.27, 86.4, 86.85, 87.34])
    costTime = np.array([ 144.548, 146.545, 146.413, 146.243, 140.435])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate( MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/spatialSVD_resnet18.png')

def spatialSVD_resnet50():
    MACRatio = np.array([0.321798, 0.435391, 0.191963, 0.325387, 0.434307])
    memoryRatio = np.array([0.196, 0.214, 0.169, 0.194, 0.216])
    accBeforeFinetune = np.array([54.956210, 69.0386, 42.665207, 53.125000, 61.365446])
    accAfterFinetune = np.array([87.59, 87.79, 87.1, 87.43, 88.0])
    costTime = np.array([ 349.689, 348.9998, 353.03, 351.152, 349.424])

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    MACRatioSortedIndex, MACRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate( MACRatio) ], key=lambda x : x[1]) ) 
    axs[0, 0].plot( MACRatioSorted, costTime[np.array(MACRatioSortedIndex)] , '^k:')
    axs[0, 0].set_xlabel('compression ratio (MAC)')
    axs[0, 0].set_ylabel('algorithm time cost (s)')
    memoryRatioSortedIndex, memoryRatioSorted = zip(* sorted([(i, x) for (i ,x) in enumerate(memoryRatio) ] , key=lambda x : x[1]) ) 
    axs[0, 1].plot( memoryRatioSorted, costTime[np.array(memoryRatioSortedIndex)], '^k:')
    axs[0, 1].set_xlabel('compression ratio (memory)')
    axs[0, 1].set_ylabel('algorithm time cost (s)')
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baseline accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/spatialSVD_resnet50.png')

def adaround():
    accResNet18 = [10.52, 72.52, 67.58]
    accResNet50 = [76.77, 84.98, 85.33]
    # x = [ 1, 2, 3 ]
    x = [ '4bit', '8bit', '16bit' ]

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
    axs[0].plot( x , [86]*3, ',-r', label = 'baseline model accuracy')
    axs[0].plot( x , accResNet18, '*m' , label = 'AdaRound quantization model accuracy')
    axs[0].plot( x , [32.25, 68.72, 67.39], '^k', label = 'normal quantized model accuracy')
    axs[0].set_title('ResNet18 accuracy after quantization')
    axs[0].set_xlabel('bit number after quantization')
    axs[0].set_ylabel('accuracy on validation dataset(%)')
    axs[0].legend()

    axs[1].plot( x , [87]*3, ',-r', label = 'baseline model accuracy')
    axs[1].plot( x , accResNet50, '*m', label = 'AdaRound quantization model accuracy')
    axs[1].plot( x , [9.95, 84.68, 85.28], '^k', label = 'normal quantized model accuracy')

    axs[1].set_title('ResNet50 accuracy after quantization')
    axs[1].set_xlabel('bit number after quantization')
    axs[1].set_ylabel('accuracy on validation dataset(%)')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTDIR + '/adaround.png')

def cle_bc_resnet18():
    accResNet18BeforeCLE = np.array([32.25, 68.72, 67.39])
    accResNet18AfterCLE = np.array([71.97, 85.83, 85.90])
    accResNet18AfterBC = np.array([61.14, 85.83, 85.90])
    br0 = np.arange(len(accResNet18BeforeCLE))
    barWidth = 0.15
    br1 = [x + barWidth for x in br0]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.figure(figsize=(10, 6), dpi=180)

    plt.bar(br0, [86.02]*len(accResNet18AfterBC), color ='#9b95c9', width = barWidth, edgecolor ='grey', label ='baseline accuracy')
    plt.bar(br1, accResNet18BeforeCLE, color ='#faa755', width = barWidth, edgecolor ='grey', label ='accuracy after quantization before CLE')
    plt.bar(br2, accResNet18AfterCLE, color ='#444693', width = barWidth, edgecolor ='grey', label ='accuracy after CLE before BC')
    plt.bar(br3, accResNet18AfterBC, color ='#f2eada', width = barWidth, edgecolor ='grey', label ='accuracy after BC')

    plt.xticks([r + barWidth for r in range(3)], [ '4bit', '8bit', '16bit'])
    plt.xlabel('bit number after quantization')
    plt.ylabel('accuracy on validation dataset(%)')
    plt.title('The influence of Cross-Layer Equalization(CLE) and Bias Correction(BC) on ResNet18')
    plt.legend(loc='lower right', shadow=True )
    plt.savefig(PLOTDIR + '/cle_bc_resnet18.png')

def cle_bc_resnet50():
    accResNet50BeforeCLE = np.array([9.95, 84.68, 85.28])
    accResNet50AfterCLE = np.array([66.10, 86.99, 87.13])
    accResNet50AfterBC = np.array([57.29, 87.13, 87.13])
    br0 = np.arange(len(accResNet50BeforeCLE))
    barWidth = 0.15
    br1 = [x + barWidth for x in br0]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.figure(figsize=(10, 6), dpi=180)

    plt.bar(br0, [86.02]*len(accResNet50AfterBC), color ='#9b95c9', width = barWidth, edgecolor ='grey', label ='baseline accuracy')
    plt.bar(br1, accResNet50BeforeCLE, color ='#faa755', width = barWidth, edgecolor ='grey', label ='accuracy after quantization before CLE')
    plt.bar(br2, accResNet50AfterCLE, color ='#444693', width = barWidth, edgecolor ='grey', label ='accuracy after CLE before BC')
    plt.bar(br3, accResNet50AfterBC, color ='#f2eada', width = barWidth, edgecolor ='grey', label ='accuracy after BC')

    plt.xticks([r + barWidth for r in range(3)], [ '4bit', '8bit', '16bit'])
    plt.xlabel('bit number after quantization')
    plt.ylabel('accuracy on validation dataset(%)')
    plt.title('The influence of Cross-Layer Equalization(CLE) and Bias Correction(BC) on ResNet50')
    plt.legend(loc='lower right', shadow=True )
    plt.savefig(PLOTDIR + '/cle_bc_resnet50.png')

def compare_time():
    # plt.figure(figsize=(20, 6), dpi=180)
    channel_time_resnet18 = np.array([863.70, 817.59, 839.01, 823.88, 769.81, 862.32, 808.28])
    weight_svd_resnet18 = np.array([246.6894, 248.419, 245.428, 245.516])
    spatial_svd_resnet18 = np.array([144.548, 146.545, 146.413, 146.24, 140.435])
    adaround_resnet18 = np.array([425.87, 422.48, 429.46])
    CLE_time_ResNet18 = np.array([10.25, 10.25, 10.25])
    bc_time_ResNet18 = np.array([19.02, 18.92, 19.29])

    channel_time_resnet50 = np.array([3309.16, 3944.23, 3270.385, 3158.2, 4006.255, 3466.395, 3259.653, 2472.81])
    weight_svd_resnet50 = np.array([754.11, 747.958, 744.14])
    spatial_svd_resnet50 = np.array([144.548, 146.545, 146.413, 146.24, 140.435])
    adaround_resnet50 = np.array([1139.88, 1114.346, 1122.76])
    CLE_time_ResNet50 = np.array([26.34, 26.253, 26.35])
    bc_time_ResNet50 = np.array([53.9424, 53.7026, 53.73])

    colName = ['C P', 'W SVD', 'S SVD', 'AdaRound', 'CLE', 'BC','CLE + BC' ]
    barPositions = (np.arange(7) + 0.75) * (-1)
    tickPositions = barPositions
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

    resnet18_xerr = [channel_time_resnet18.max() -  channel_time_resnet18.min(),\
             weight_svd_resnet18.max()- weight_svd_resnet18.min(), spatial_svd_resnet18.max() - spatial_svd_resnet18.min(), adaround_resnet18.max() - adaround_resnet18.min(), \
                 CLE_time_ResNet18.max() - CLE_time_ResNet18.min(), bc_time_ResNet18.max() - bc_time_ResNet18.min(), (CLE_time_ResNet18 + bc_time_ResNet18).max() - (CLE_time_ResNet18 + bc_time_ResNet18).min()] 
    resnet18_xerr = [0.5*x for x in resnet18_xerr]
    axs[0].barh(barPositions, [ channel_time_resnet18.mean(), weight_svd_resnet18.mean(), spatial_svd_resnet18.mean(), adaround_resnet18.mean(),\
         CLE_time_ResNet18.mean(), bc_time_ResNet18.mean(), (CLE_time_ResNet18 + bc_time_ResNet18).mean()], 0.5, xerr = resnet18_xerr, align='center', color = '#905a3d')
    axs[0].set_yticks(tickPositions)
    axs[0].set_yticklabels(colName)
    axs[0].set_title('Time Cost on ResNet18')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Algorithm Name')

    resnet50_xerr = [channel_time_resnet50.max() -  channel_time_resnet50.min(),\
             weight_svd_resnet50.max()- weight_svd_resnet50.min(), spatial_svd_resnet50.max() - spatial_svd_resnet50.min(), adaround_resnet50.max() - adaround_resnet50.min(), \
                 CLE_time_ResNet50.max() - CLE_time_ResNet50.min(), bc_time_ResNet50.max() - bc_time_ResNet50.min(), (CLE_time_ResNet50 + bc_time_ResNet50).max() - (CLE_time_ResNet50 + bc_time_ResNet50).min() ]
    resnet50_xerr = [ 0.5 * x for x in resnet50_xerr]
    axs[1].barh(barPositions, [ channel_time_resnet50.mean(), weight_svd_resnet50.mean(), spatial_svd_resnet50.mean(), adaround_resnet50.mean(),\
         CLE_time_ResNet50.mean(), bc_time_ResNet50.mean(), CLE_time_ResNet50.mean() + bc_time_ResNet50.mean()], 0.5, xerr = resnet50_xerr, color = '#905a3d')
    axs[1].set_yticks(tickPositions)
    axs[1].set_yticklabels(colName)
    axs[1].set_title('Time Cost on ResNet50')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Algorithm Name')
    # plt.tight_layout()
    plt.savefig(PLOTDIR + '/compare_time.png')

def compare_memory():

    colName = ['C P', 'W SVD', 'S SVD', 'AdaRound', 'CLE + BC' ]
    
    channel_resnet18 = np.array(list(zip(*[(5.9877166748046875, 2347.75), (5.322246551513672, 2347.75), (5.932491302490234, 2347.75), (5.547443389892578, 2417.75), (5.955879211425781, 2347.75), (5.724559783935547, 2347.75), (5.965583801269531, 2457.75), (5.558692932128906, 2347.75)])))
    weightSVD_resnet18 = np.array(list(zip(*[(4.9183502197265625, 2660.75), (4.33245849609375, 2660.75), (4.370265960693359, 2660.75), (4.3753204345703125, 2660.75), (4.344554901123047, 2306.75), (4.312370300292969, 2306.75), (4.3216552734375, 2306.75), (4.323333740234375, 2306.75)])))
    spatialSVD_resnet18 = np.array(list(zip(*[(4.410438537597656, 2081.75), (4.446521759033203, 2081.75), (4.429775238037109, 2081.75), (4.4181671142578125, 2081.75), (4.425960540771484, 2081.75), (4.388862609863281, 2081.75), (4.4202728271484375, 2081.75), (4.461154937744141, 2081.75)])))
    adaround_resnet18 = np.array(list(zip(*[(4.282390594482422, 2017.75), (4.346527099609375, 2009.75), (4.343418121337891, 2007.75)])))
    cle_bc_resnet18 = np.array(list(zip(*[(3.9810256958007812, 2038.75), (3.9981689453125, 2038.75), (4.0495758056640625, 2038.75)])))
    
    channel_resnet50 = np.array(list(zip(*[(14.816150665283203, 2941.75), (12.694087982177734, 2941.75), (12.42123794555664, 2941.75), (8.4573974609375, 2941.75), (13.638538360595703, 2941.75), (11.165534973144531, 2941.75), (11.165771484375, 2941.75), (11.442974090576172, 2941.75)])))
    weightSVD_resnet50 = np.array(list(zip(*[(4.336017608642578, 2042.75), (4.314155578613281, 2042.75), (4.3668212890625, 2042.75), (4.343132019042969, 2042.75), (4.339973449707031, 2400.75), (4.356632232666016, 2400.75), (4.388874053955078, 2400.75), (4.241703033447266, 2400.75)])))
    spatialSVD_resnet50 = np.array(list(zip(*[(4.453243255615234, 2021.75), (4.461597442626953, 2021.75), (4.466957092285156, 2021.75), (4.4745941162109375, 2021.75), (4.477317810058594, 2021.75), (4.479930877685547, 2021.75), (4.4957427978515625, 2021.75), (4.508365631103516, 2021.75)])))
    adaround_resnet50 = np.array(list(zip(*[(4.403831481933594, 2317.75), (3.7180709838867188, 2321.75), (4.195941925048828, 2321.75)])))
    cle_bc_resnet50 = np.array(list(zip(*[(4.180061340332031, 2292.75), (4.1720733642578125, 2292.75), (4.167217254638672, 2222.75)])))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

    resnet18_memory_yerr =  [   channel_resnet18[0].max()-channel_resnet18[0].min(),\
    weightSVD_resnet18[0].max()-weightSVD_resnet18[0].min(), spatialSVD_resnet18[0].max()-spatialSVD_resnet18[0].min(),\
        adaround_resnet18[0].max() - adaround_resnet18[0].min(), cle_bc_resnet18[0].max() - cle_bc_resnet18[0].min()]
    resnet18_memory_yerr = [0.5*x for x in resnet18_memory_yerr]
    
    axs[0, 0].bar(colName, [channel_resnet18[0].mean(), weightSVD_resnet18[0].mean(), spatialSVD_resnet18[0].mean(), adaround_resnet18[0].mean(), cle_bc_resnet18[0].mean()], color = '#8f80ca', yerr = resnet18_memory_yerr)
    axs[0, 0].set_title('Algorithm Memory Cost on ResNet18')
    axs[0, 0].set_xlabel('Algorithm Name')
    axs[0, 0].set_ylabel('Memory Cost(GB)')

    resnet50_memory_yerr =  [   channel_resnet50[0].max()-channel_resnet50[0].min(),\
    weightSVD_resnet50[0].max()-weightSVD_resnet50[0].min(), spatialSVD_resnet50[0].max()-spatialSVD_resnet50[0].min(),\
        adaround_resnet50[0].max() - adaround_resnet50[0].min(), cle_bc_resnet50[0].max() - cle_bc_resnet50[0].min()]
    resnet50_memory_yerr = [0.5*x for x in resnet50_memory_yerr]
    axs[0, 1].bar(colName, [channel_resnet50[0].mean(), weightSVD_resnet50[0].mean(), spatialSVD_resnet50[0].mean(), adaround_resnet50[0].mean(), cle_bc_resnet50[0].mean()], color = '#686eba', yerr = resnet50_memory_yerr)
    axs[0, 1].set_title('Algorithm Memory Cost on ResNet50')
    axs[0, 1].set_xlabel('Algorithm Name')
    axs[0, 1].set_ylabel('Memory Cost(GB)')

    resnet18_video_memory_yerr =  [   channel_resnet18[1].max()-channel_resnet18[1].min(),\
    weightSVD_resnet18[1].max()-weightSVD_resnet18[1].min(), spatialSVD_resnet18[1].max()-spatialSVD_resnet18[1].min(),\
        adaround_resnet18[1].max() - adaround_resnet18[1].min(), cle_bc_resnet18[1].max() - cle_bc_resnet18[1].min()]
    resnet18_video_memory_yerr = [0.5*x for x in resnet18_video_memory_yerr]

    axs[1, 0].bar(colName, [channel_resnet18[1].mean(), weightSVD_resnet18[1].mean(), spatialSVD_resnet18[1].mean(), adaround_resnet18[1].mean(), cle_bc_resnet18[1].mean()], color = '#abc88b', yerr = resnet18_video_memory_yerr)
    axs[1, 0].set_title('Algorithm Video Memory Cost on ResNet18')
    axs[1, 0].set_xlabel('Algorithm Name')
    axs[1, 0].set_ylabel('Memory Cost(MB)')

    resnet50_video_memory_yerr =  [   channel_resnet50[1].max()-channel_resnet50[1].min(),\
    weightSVD_resnet50[1].max()-weightSVD_resnet50[1].min(), spatialSVD_resnet50[1].max()-spatialSVD_resnet50[1].min(),\
        adaround_resnet50[1].max() - adaround_resnet50[1].min(), cle_bc_resnet50[1].max() - cle_bc_resnet50[1].min()]
    resnet50_video_memory_yerr = [0.5*x for x in resnet50_memory_yerr]

    axs[1, 1].bar(colName, [channel_resnet50[1].mean(), weightSVD_resnet50[1].mean(), spatialSVD_resnet50[1].mean(), adaround_resnet50[1].mean(), cle_bc_resnet50[1].mean()], color = '#88b365', yerr = resnet50_video_memory_yerr)
    axs[1, 1].set_title('Algorithm Video Memory Cost on ResNet50')
    axs[1, 1].set_xlabel('Algorithm Name')
    axs[1, 1].set_ylabel('Memory Cost(MB)')

    plt.tight_layout()
    plt.savefig(PLOTDIR + '/compare_memory.png')




if __name__ == '__main__':
    # channel_resnet18()
    # channel_resnet50()
    # weightSVD_resnet18()
    # weightSVD_resnet50()
    # spatialSVD_resnet18()
    # spatialSVD_resnet50()
    adaround()
    # cle_bc_resnet18()
    # cle_bc_resnet50()
    # compare_time()
    # compare_memory()

