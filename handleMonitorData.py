from audioop import findmax
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
    with  open('./monitorLog/adaround_resnet50.log', mode='r') as f:
        lines = f.readlines()
        sections = [ 722, 913, 1103, 1296, 1489, 1681, 1875]
        # for trueFile in [lines[ 530:sections[0]], lines[sections[0]:sections[1]], lines[sections[1]:sections[2]], lines[sections[2]:sections[3]]]:
        #     ans.append(findMax(trueFile))
        # for falseFile in [lines[sections[3]:sections[4]], lines[sections[4]:sections[5]], lines[sections[5]:sections[6]], lines[sections[6]:]]:
        #     ans.append(findMax(falseFile))
        ans.append(findMax(lines))
    print(ans)
