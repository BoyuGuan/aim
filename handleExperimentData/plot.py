import os
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
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
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
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
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
    accBeforeFinetune = np.array([78.025478, 21.725717, 36.275876, 73.377787])
    accAfterFinetune = np.array([87.99, 84.92, 87.0, 88.34])
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
    axs[1, 0].plot( MACRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
    axs[1, 0].plot( MACRatioSorted, accAfterFinetune[np.array(MACRatioSortedIndex)] , 's--y' , label='accuracy before finetune')
    axs[1, 0].plot( MACRatioSorted, accBeforeFinetune[np.array(MACRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 0].set_xlabel('compression ratio (MAC)')
    axs[1, 0].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 0].legend()
    axs[1, 1].plot( memoryRatioSorted, [90.2] * len(MACRatioSorted) , ',-r' , label='baselien accuracy')
    axs[1, 1].plot(memoryRatioSorted, accAfterFinetune[np.array(memoryRatioSortedIndex)] , 's--y', label='accuracy after finetune')
    axs[1, 1].plot( memoryRatioSorted, accBeforeFinetune[np.array(memoryRatioSortedIndex)] , '8-.c', label='accuracy before finetune')
    axs[1, 1].set_xlabel('compression ratio (memory)')
    axs[1, 1].set_ylabel('accuracy on validation dataset(%)')
    axs[1, 1].legend()
    fig.set(tight_layout = True)
    plt.savefig(PLOTDIR + '/channel_resnet50.png')





if __name__ == '__main__':
    # channel_resnet18()
    channel_resnet50()