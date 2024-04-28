import numpy as np
import math

inputs = [.35,.9]
learning_rate = 1
expected_output = [.5]

class Layer_Dense:
    def __init__(self, weights, biases):
        self.weights = weights#array with random values
        self.biases = biases #bias array with value zero
    def forward(self,inputs):
        outputs = []
        for weight, bias in zip(self.weights,self.biases):
            print(np.dot(inputs,weight))
            output = np.dot(inputs,weight) + bias
            outputs.append(self.activation_funct(output))
        self.output = outputs
    def activation_funct(self, x):
        return 1/(1 + math.exp(-x)) 
    def error(self):
        try:
            return (self.output - self.expected)**2
        except:
            return -1
    def update_weights(self,hidden_layer: bool, next_layer_delta =[],next_layer_weights=[] ):
        if hidden_layer:
            # [-0.406]
            # [[0.2592401003575149, 0.859240100357515]] weights
            # self.output [0.5974857658270161, 0.6942363401080305]
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
                self.weights + self.delta_weight
            return
        delta_weights = []
        for i in range(len(self.weights)):
            dw = self.output[i]*(1-self.output[i])*(expected_output[i] - self.output[i])
            delta_weights.append(dw)
            self.weights[i] = list(map(lambda x: x + dw,self.weights[i]))
        self.delta_weight = delta_weights
layer1 = Layer_Dense([[0.1,0.8],[0.4,0.6]],[0,0])
layer2 = Layer_Dense([[.3,.9]],[0])

layer1.forward(inputs)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
layer2.update_weights(False)
layer1.update_weights(True, layer2.delta_weight,layer2.weights)
print("****")
print(layer1.delta_weight)
print(layer2.delta_weight)

