import math
import numpy as np


class Layer_Dense:
    def __init__(self, weights, biases):
        self.weights = weights#array with random values
        self.biases = biases #bias array with value zero
    def forward(self,inputs):
        outputs = []
        for weight, bias in zip(self.weights,self.biases):
            print(np.dot(inputs,weight))
            output = np.dot(inputs,weight) + bias[0]
            outputs.append(self.activation_funct(output))
        self.output = outputs
    def activation_funct(self, x):
        return 1/(1 + math.exp(-x)) 
    def ReLU(self, x):
        return max(0,x)
    def error(self):
        try:
            return (self.output - self.expected)**2
        except:
            return -1
    def update_weights(self,hidden_layer: bool, expected_output=[], next_layer_delta =[],next_layer_weights=[] ):
        if hidden_layer:
            self.delta_weight = []
            for i in range(len(next_layer_weights)):
                next_layer_weights[i] = list(map(lambda x: x-next_layer_delta[i],next_layer_weights[i]))
            next_layer_weights_transpose = np.array(next_layer_weights).T
            for output_index in range(len(self.output)): #for each nodes output_index
                tot = 0
                for weight, delta in zip(next_layer_weights_transpose[output_index],next_layer_delta):
                    tot += weight*delta
                dw = self.output[output_index]*(1-self.output[output_index])* tot
                self.delta_weight.append(dw)
                self.weights[output_index] = self.weights[output_index] + self.delta_weight[output_index]
            return
        delta_weights = []
        for i in range(len(self.weights)):
            dw = self.output[i]*(1-self.output[i])*(expected_output[i] - self.output[i])
            delta_weights.append(dw)
            self.weights[i] = list(map(lambda x: x + dw,self.weights[i]))
        self.delta_weight = delta_weights


