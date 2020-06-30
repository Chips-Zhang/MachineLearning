import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def show(file_name):
    data = pd.read_csv(file_name)
    header=["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]
    sns.set(style="white", palette="muted", color_codes=True)
    for i in range(len(header)):
        ax1 =sns.kdeplot(data[header[i]][data['label']=='male'],color='r',shade=True,label='male')
        ax2 =sns.kdeplot(data[header[i]][data['label'] == 'female'], color='b',shade=True,label='female')
        plt.show()

if __name__ == "__main__":
    file_name = "voice.csv"
    show(file_name)