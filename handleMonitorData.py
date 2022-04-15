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
    with  open('./monitorLog/channel_resnet50.txt', mode='r') as f:
        lines = f.readlines()
        sections = [ 1023, 1897, 2719, 3343, 4336, 5160, 5956]
        for trueFilen in [lines[:sections[0]], lines[sections[0]:sections[1]], lines[sections[1]:sections[2]], lines[sections[2]:sections[3]]]:
            ans.append(findMax(trueFilen))
        for falseFilen in [lines[sections[3]:sections[4]], lines[sections[4]:sections[5]], lines[sections[5]:sections[6]], lines[sections[6]:]]:
            ans.append(findMax(falseFilen))
    print(ans)
