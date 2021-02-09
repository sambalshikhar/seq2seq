import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import fasttext

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Acc:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.true = 0

    def addScore(self,score):
        self.count+=1
        self.true+=score

    def getAcc(self):
        if self.count!=0:
            return (self.true/self.count)*100
        else:
            return 0

class Multiple_acc:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.true = 0

    def addScore(self,score):
        self.count+=1
        self.true+=score

    def getAcc(self):
        if self.count!=0:
            return (self.true/self.count)*100
        else:
            return 0