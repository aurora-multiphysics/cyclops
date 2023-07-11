from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from torch import nn
import pandas as pd
import torch



# Model overview
# 5 input neurons for 5 temperature sensors.
# 2,992 output neurons for 2,992 points in the temperature field.
# 




#Define important variables.
batch_size = 64
epochs = 10
loss_fn = nn.CrossEntropyLoss()


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\nUsing {device} device")




#Define custom dataset.
class FashionSet(Dataset):
    def __init__(self, points, faces):
        self.__length = len(points)
        self.__data = points
        self.__targets = faces
        

    def __len__(self):
        return self.__length


    def __getitem__(self, i):
        return self.__data[i], self.__targets[i]