# 1.py using dot product
import numpy as np

inputs = [1,2,3,2.5] # i1, i2, i3 
weights = [
    [0.2,0.8, -0.5, 1], #weight a
    [0.5,-.91,.26,-.5], #weight b
    [-.26,-.27,.17,.87], #weight c
    ]

bias = [2, 3,.5] # ba, bb, bc

neuron_outputs = []
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = np.dot(inputs,neuron_weights) + neuron_bias
    neuron_outputs.append(neuron_output)
print(neuron_outputs)