import numpy as np
import pandas as pd
from layer_dense import Layer_Dense
import os


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


if __name__ == "__main__":
    data = pd.read_csv('/Users/jopauljoy/Brain/Ai/Sendex - Neural Networks/A. Digit Recognizer/datas/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_list = data[0:200]
    train_output_list = data_list[:][0]

    W1,b1,W2,b2 = init_params()

    layer1 = Layer_Dense(W1,b1)
    layer2 = Layer_Dense(W2,b2)

    for i,train_data in enumerate(data_list):
        print("Iteration", i)
        train_data = train_data[1:]
        layer1.forward(train_data)
        layer2.forward(layer1.output)
        layer2.update_weights(False,train_output_list)
        layer1.update_weights(True,[] ,layer2.delta_weight,layer2.weights)




