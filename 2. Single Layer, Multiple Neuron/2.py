#improved code of 1.py
inputs = [1,2,3,2.5] # i1, i2, i3 
weights = [
    [0.2,0.8, -0.5, 1], #weight a
    [0.5,-.91,.26,-.5], #weight b
    [-.26,-.27,.17,.87], #weight c
    ]

bias = [2, 3,.5] # ba, bb, bc

input_len = len(inputs)
neuron_len = len(weights)

neuron_outputs = []
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = 0
    for neuron_input, neuron_input_weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input*neuron_input_weight
    neuron_output += neuron_bias
    neuron_outputs.append(neuron_output)
print(neuron_outputs)
