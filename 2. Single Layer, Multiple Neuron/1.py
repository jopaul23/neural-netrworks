inputs = [1,2,3,2.5] # i1, i2, i3 
weights = [
    [0.2,0.8, -0.5, 1], #weight a
    [0.5,-.91,.26,-.5], #weight b
    [-.26,-.27,.17,.87], #weight c
    ]

bias = [2, 3,.5] # ba, bb, bc

input_len = len(inputs)
neuron_len = len(weights)

outputs = []
for i in range(neuron_len):
    output = 0
    for j in range(input_len):
        output += inputs[j]*weights[i][j]
    output += bias[i]
    outputs.append(output)
print(outputs)
