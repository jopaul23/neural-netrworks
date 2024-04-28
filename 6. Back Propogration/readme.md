# Back Propogation
Back propogation is used to update weights and bias
<a href="https://www.youtube.com/watch?v=tIeHLnjs5U8">youtube</a>
### Consider the case of where we have multiple layers with single neuron
![<img src>](<assets/derivation/0. neuron.png>)

#### activation function
represented as a(L) for last neuron, a(L-1) for second last neuro, and so on

#### desired output
the desired output is represented as y

#### cost (Co)
the cost is the square of difference of actual output a(L) and Desired output y
(a(L) - y)2

### Derivation
from the above details we can define following
![alt text](<assets/derivation/1. derived-values.png>)

from the chain rule we can say that
![alt text](<assets/derivation/2. chain-rule.png>)

Now substituting values from the defined eequations and applying derivative to it we get followin equations on the left of following image
![alt text](<assets/derivation/3. applying-derivatives.png>)

now substituting these values to equation that we derived using chain rule we get
![alt text](<assets/derivation/4. derivation.png>)

#### We can do the same to find difference in bias, activation
for bias
![alt text](<assets/derivation/6. bias.png>)
for activation
![alt text](<assets/derivation/7. activation.png>)

### Considering case of multiple neuron in single layer and applying the above equation we get
for activation function
![alt text](<assets/derivation/5. multiple-neuron.png>)
for weight


## final algorithm
![alt text](assets/algorithm.png)
## solved problem
![alt text](assets/solved-problem.png)