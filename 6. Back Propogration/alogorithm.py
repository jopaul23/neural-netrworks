from layer_dense import Layer_Dense

inputs = [.35,.9]
learning_rate = 1
expected_output = [.5]
layer1_weights = [
    [0.1,0.8], # weights of each inputs for first neuron in the layer
    [0.4,0.6]  # weights of each inputs in second neuron in the layer
    ]
layer1_baises = [[0] ,[0]] # biases of each neurons in the layer

layer2_weights =[
    [.3,.9] # weights of each inputs for first neuron in the layer
    ] 
layer2_baises = [[0]] # bias of first neuron

layer1 = Layer_Dense(layer1_weights,layer1_baises)
layer2 = Layer_Dense(layer2_weights,layer2_baises)

layer1.forward(inputs)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
layer2.update_weights(False, expected_output)
layer1.update_weights(True, [],layer2.delta_weight,layer2.weights)
print("****")
print(layer1.delta_weight)
print(layer2.delta_weight)