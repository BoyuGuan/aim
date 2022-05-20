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
    with  open('./monitorLog/cle_bc.txt', mode='r') as f:
        lines = f.readlines()
        sections = [ 32, 64, 96, 186, 276, 366]
        for trueFile in [lines[ 0:sections[0]], lines[sections[0]:sections[1]], lines[sections[1]:sections[2]]]:
            ans.append(findMax(trueFile))
        for falseFile in [lines[sections[2]:sections[3]], lines[sections[3]:sections[4]], lines[sections[4]:sections[5]]]:
            ans.append(findMax(falseFile))
    print(ans)
