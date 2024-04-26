import numpy as np
import matplotlib.pyplot as plt

def create_graph():
    graph_min_val = -1
    graph_max_val = 1
    plt.xlim(graph_min_val, graph_max_val)
    plt.ylim(graph_min_val, graph_max_val)

    plt.plot([0,0],[graph_max_val,graph_min_val],'black')
    plt.plot([graph_max_val,graph_min_val],[0,0],'black')

    plt.style.use('ggplot')

    plt.xticks(np.arange(graph_min_val, graph_max_val, .10))
    plt.yticks(np.arange(graph_min_val, graph_max_val, .10))

class Layer_Dense:
    def __init__(self, weights, biases, hidden = True):
        self.hidden = hidden
        self.weights = weights
        self.biases = biases
    def forward(self,inputs):
        output = np.dot(inputs,self.weights)+self.biases
        print("out: ",output)
        if not self.hidden:
            self.output = output
            return
        if isinstance(output, (int, float)):
            self.output = max(output,0)
            return
        self.output = list(map(lambda x: max(x,0),output))


inputs = []
outputs1 = []
outputs2 = []
outputs3 = []


for i in range(10):
    layer_1 = Layer_Dense([6,3.5],[0,-.42]) 
    layer_1.forward(i*.1)

    layer_2_weights = np.identity(2)
    layer_2_weights[0][0] = -1
    layer_2_weights[1][1] = -1
    layer_2 = Layer_Dense(layer_2_weights,[.7,0]) #4 nodes, 4x4 different weights(as we are having 4 inputs each for 4 nodes. each weight for all 4 inputs an array is given as an array of length 4 )
    layer_2.forward(layer_1.output)

    layer_3 = Layer_Dense([-1,1],0,False) #single node, 4 different weights(as we are havind 4 inputs)
    layer_3.forward(layer_2.output)

    inputs.append(i*.1)
    outputs1.append(layer_1.output)
    outputs2.append(layer_2.output)
    outputs3.append(layer_3.output)

create_graph()

plt.plot(inputs,outputs3,"green")

plt.grid(True)
plt.show()