# A Simple Neural Network Implementation

## Weights update

Notation:

![input](https://latex.codecogs.com/svg.latex%24X%20%3D%20%5Ctext%7Binput%20matrix%7D%24)

* $X = \text{input matrix}$
* $Y_{out} = \text{output obtained from the output layer during the forward step}$
* $Y_{hid} = \text{output obtained from the hidden layer during the forward step}$
* $Y_{target} = \text{expected values}$
* $Err = Y_{out} - Y_{target}$
* $f(x)=\frac{1}{1 + e^{-x}}\text{\space (sigmoid)}$
* $f'(x)=x*(1-x)$
* $\eta=\text{learning rate}$
* $\lambda=\text{regularization rate}$
* $\alpha=\text{momentum rate}$
* $W_{out}=\text{weights matrix relative to output layer}$
* $W_{hid}=\text{weights matrix relative to hidden layer}$
* $b_{out} = \text{bias vector for output layer}$
* $b_{hid} = \text{bias vector for hidden layer}$

\
\
Backpropagation factors:

$$\delta_{out} = Err * f'(Y_{out})$$

$$\delta_{hid} = W_{out}*\delta_{out}*f'(y_{hid})$$
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