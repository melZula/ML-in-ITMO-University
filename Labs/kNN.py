import pandas as pd
from matplotlib import pyplot as plt
filename = "datasets/heart.csv"
dataset = pd.read_csv(filename)

def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax

def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            dif = minmax[i][1] - minmax[i][0]
            row[i] = ((row[i] - minmax[i][0]) / dif) if dif != 0 else 0

    return dataset

min_max = minmax(dataset.values)
normalized_dataset_values = normalize(dataset.values, min_max)
print(pd.DataFrame(normalized_dataset_values))