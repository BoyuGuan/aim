import os
import matplotlib.pyplot as plt

PLOTDIR = './handleExperimentData/plotImages'
os.makedirs(PLOTDIR, exist_ok=True)

def channel_resnet18():
    MACRatio = [0.302908, 0.545076, 0.649888, 0.665484, 0.197297, 0.399185, 0.473236, 0.621336]
    memoryRatio = [0.198605, 0.241760, 0.296652, 0.389077, 0.160678, 0.210540,  0.226964, 0.262622]
    accBeforeFinetune = [43.869427, 74.343153, 84.056529, 83.588774, 29.319268, 57.314889, 48.029459, 82.931927]
    accAfterFinetune = [86.17, 88.37, 88.66, 88.48, 84.45, 87.42, 87.82, 88.63]
    costTime = [ 863.702305316925, 817.5897779464722,  839.0135326385498, 823.8756124973297, 858.6545453071594, 769.8132638931274, 862.3200585842133, 808.2758123874664]

    plt.plot( MACRatio, costTime )
    plt.savefig(PLOTDIR + '/channel_resnet18.png')


if __name__ == '__main__':
    channel_resnet18()