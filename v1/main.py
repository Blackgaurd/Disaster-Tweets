import os
import time

from network import Network
from layer import *
from activation import *

print("Finished imports")

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

train_path = DIR_PATH + "/../input/train.csv"
test_path = DIR_PATH + "/../input/test.csv"
submission_path = DIR_PATH + "/sumbission.csv"

Utils.seed(1)

ann = Network(
    batch_size=100,
    learning_rate=0.1,
    layers=[Dense(100, Relu), Dense(100, Relu), Dense(2, Softmax)],
    train_file_path=train_path,
    normalization=Utils.normalize,
)

print("Finished setup")

epochs = 100
for i in ann.run(epochs):
    print(f"Epoch:\t\t{i[0]}/{epochs}\nAccuracy:\t{i[1]}\n")
    time.sleep(0.1)

print("Finished training")

ann.submit_csv(test_path, submission_path)
