# A Simple Neural Network Implementation

## Weights update

Notation:
<!-- markdownlint-disable MD033 -->

<img src="https://latex.codecogs.com/svg.latex?$X&space;=&space;\text{input&space;matrix}$" title="$X = \text{input matrix}$"/>

<img src="https://latex.codecogs.com/svg.latex?$Y_{out}&space;=&space;\text{output&space;obtained&space;from&space;the&space;output&space;layer&space;during&space;the&space;forward&space;step}$" title="$Y_{out} = \text{output obtained from the output layer during the forward step}$" />

<img src="https://latex.codecogs.com/svg.latex?$Y_{hid}&space;=&space;\text{output&space;obtained&space;from&space;the&space;hidden&space;layer&space;during&space;the&space;forward&space;step}$" title="$Y_{hid} = \text{output obtained from the hidden layer during the forward step}$" />

<img src="https://latex.codecogs.com/svg.latex?$Y_{target}&space;=&space;\text{expected&space;values}$" title="$Y_{target} = \text{expected values}$" />

<img src="https://latex.codecogs.com/svg.latex?$Err&space;=&space;Y_{out}&space;-&space;Y_{target}$" title="$Err = Y_{out} - Y_{target}$" />


<img src="https://latex.codecogs.com/svg.latex?$f(x)=\frac{1}{1&space;&plus;&space;e^{-x}}\text{\space&space;(sigmoid)}$" title="$f(x)=\frac{1}{1 + e^{-x}}\text{\space (sigmoid)}$" />

<img src="https://latex.codecogs.com/svg.latex?$f'(x)=x*(1-x)$" title="$f'(x)=x*(1-x)$" />

<img src="https://latex.codecogs.com/svg.latex?$\eta=\text{learning&space;rate}$" title="$\eta=\text{learning rate}$" />

<img src="https://latex.codecogs.com/svg.latex?$\lambda=\text{regularization&space;rate}$" title="$\lambda=\text{regularization rate}$" />

<img src="https://latex.codecogs.com/svg.latex?$\alpha=\text{momentum&space;rate}$" title="$\alpha=\text{momentum rate}$" />

<img src="https://latex.codecogs.com/svg.latex?$W_{out}=\text{weights&space;matrix&space;relative&space;to&space;output&space;layer}$" title="$W_{out}=\text{weights matrix relative to output layer}$" />

<img src="https://latex.codecogs.com/svg.latex?$W_{hid}=\text{weights&space;matrix&space;relative&space;to&space;hidden&space;layer}$" title="$W_{hid}=\text{weights matrix relative to hidden layer}$" />

<img src="https://latex.codecogs.com/svg.latex?$b_{out}&space;=&space;\text{bias&space;vector&space;for&space;output&space;layer}$" title="$b_{out} = \text{bias vector for output layer}$" />

<img src="https://latex.codecogs.com/svg.latex?$b_{hid}&space;=&space;\text{bias&space;vector&space;for&space;hidden&space;layer}$" title="$b_{hid} = \text{bias vector for hidden layer}$" />

\
\
Backpropagation factors:

<img src="https://latex.codecogs.com/svg.latex?$$\delta_{out}&space;=&space;Err&space;*&space;f'(Y_{out})$$" title="$$\delta_{out} = Err * f'(Y_{out})$$" />

<img src="https://latex.codecogs.com/svg.latex?$$\delta_{hid}&space;=&space;W_{out}*\delta_{out}*f'(y_{hid})$$" title="$$\delta_{hid} = W_{out}*\delta_{out}*f'(y_{hid})$$" />
\
\
\
The correction factor is so
$$
\Delta W_{out} = \eta\space (Y_{hid}*\delta_{out}-\lambda*W_{out}+\alpha * \Delta W_{out \space old})
$$
\
\
\
Updating the wheigts
$$
W_{out \space new} = \Delta W_{out} + W_{out \space old}
$$

[]