import numpy as np

inputs = [[1,2,3,2.5],
          [1,2,3,2.5],
          [1,2,3,2.5],
          ] 
weights = [
    [0.2,0.8, -0.5, 1], #weight a
    [0.5,-.91,.26,-.5], #weight b
    [-.26,-.27,.17,.87], #weigh t c
]
biases = [2, 3,.5]

weights_transpose  = np.array(weights).T

output = np.dot(inputs, weights_transpose) + biases

print(output)