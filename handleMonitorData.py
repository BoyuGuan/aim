import pandas as pd
import numpy as np

def findMax(filen):
    m = -1
    g = -1
    for line in filen:
        line = line.split()
        m = max(float(line[-2]), m)
        g = max(float(line[-1]), g)
    return m, g 


if __name__ == '__main__':
    ans = []
    count = 0
    with  open('./monitorLog/resnet50.log', mode='r') as f:
        lines = f.readlines()
        for trueFilen in [lines[:995], lines[995:1927], lines[1927:2744], lines[2744:3617]]:
            ans.append(findMax(trueFilen))
        for falseFilen in [lines[3617:4652], lines[4652:5476], lines[5476:6348], lines[6348: 8200]]:
            ans.append(findMax(falseFilen))
    print(ans)
