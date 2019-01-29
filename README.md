# A Simple Neural Network Implementation

Basic functions for the training and use of a neural network, with a single hidden layer composed of a variable number of nodes, random weights inizialization, L2 regularization and momentum.\
Of course, it is not the most complete experience in term of NN on Python.
Only few parameters to play with, less flexibility, less options. Easier to use ;)

* ```NN_train.py``` contains the actual training functions, for regression and for classification tasks.

* ```NN_tools.py``` contains some useful functions both for the training phase and the visualization of results.

* ```examples_of_usages.ipynb``` contains some case studies for showing the usage of the main functions.

## NN_train.py

### binary classification

* ```train_model_classification(X, y, X_valid=None, y_valid=None, neurons_hidden=5, epochs=500, lr=0.1, reg_lambda=0.0, momentum_alpha=0.0, validation_split = 0.0, threshold=0.5)```\
\
It return a trained ```model``` for classification, an object that incapsulate all the informations for predicting new values, plot the training intermediate results, and retrieve the hyperparameters used for the training process.\
\
Parameters:

  * ```X``` = the input dataset to be used for training.
  * ```y``` = the vector of target variables.
  * ```X_valid``` = extra dataset used as validation set.
  * ```y_valid``` = vector of target variables for the validation set.
  * ```neurons_hidden``` = number of neurons in the hidden layer.
  * ```epochs``` = for how many iteration the training process will last.
  * ```lr``` = learning rate.
  * ```reg_lambda``` = regularization rate.
  * ```momentum_alpha``` = momentum rate.
  * ```validation_split```= the portion to be extracted from ```X``` for validation purpose (Ignored if ```X_valid``` and ```y_valid``` are not ```None```).
  * ```threshold``` = the thresold that a record need to achieve in order to be classified as positive. It is used during the prediction of values for the accuracy calculation.

### regression

* ```train_model_classification(X, y, X_valid=None, y_valid=None, neurons_hidden=5, epochs=500, lr=0.1, reg_lambda=0.0, momentum_alpha=0.0, validation_split = 0.0)```\
The regression version of ```train_model_classification```. Same parameters (except ```threshold```) and same type of object returned

*Tech* = the output nodes here use a linear function instead of the sigmoid as in all the other nodes.

## NN_tools.py

* ```predict_values(model, X, classification=True)```\
return a numpy array containing the results of the application of the ```X``` dataset to the ```model```. Specifiy if it is a classification task or not, by the last parameter ```classification``` (default = ```True```)
* ```plot_loss_accuracy(model, print_accuracy=True)```\
plot the graphs of the training process, in term of values of the loss function, and the accuracy result only in case of classification task. When the ```model``` is for regression, it must be call with ```print_accuracy=False```

# An Example
```
# dataset loading
monks2_train = pd.read_csv("input/monk2_oneofk.train", delimiter = " ", )
monks2_train_x = monks2_train.drop(["target"],axis = 1).values
monks2_train_y = monks2_train["target"].values

# training
monks2_model = train_model(X=monks2_train_x,
                           y=monks2_train_y,
                           neurons_hidden=4,
                           epochs= 150,
                           momentum_alpha=0.7, 
                           lr=0.1, reg_lambda=0.0, 
                           validation_split = 0.2)

plot_loss_accuracy(monks2_model)
```

![accuracy](NeuralNetwork/screenshots/loss_monks2.png)

## Theory: the Back-propagation algorithm

In this section is briefly explained how the neural network is trained.\
A neural network can be fully rappresented here with his matrices of weights for the hidden layer and the output layer, and the two bias vectors.
So, for each epochs, these four object will be updated in base at the error they do respect to the expected output.

### Notation

<!-- markdownlint-disable MD033 -->
<img src="https://latex.codecogs.com/svg.latex?$X&space;=&space;\text{input&space;data&space;matrix}$" title="$X = \text{input data matrix}$" />\
<img src="https://latex.codecogs.com/svg.latex?Y_{target}=\text{input&space;labels&space;vector}" title="Y_{target}=\text{input labels vector}" />\
<img src="https://latex.codecogs.com/svg.latex?$Y_{out}&space;=&space;\text{output&space;obtained&space;from&space;the&space;output&space;layer&space;during&space;the&space;forward&space;step}$" title="$Y_{out} = \text{output obtained from the output layer during the forward step}$" />\
<img src="https://latex.codecogs.com/svg.latex?$Y_{hid}&space;=&space;\text{output&space;obtained&space;from&space;the&space;hidden&space;layer&space;during&space;the&space;forward&space;step}$" title="$Y_{hid} = \text{output obtained from the hidden layer during the forward step}$" />\
<img src="https://latex.codecogs.com/svg.latex?$W_{out}=\text{weights&space;matrix&space;relative&space;to&space;output&space;layer}$" title="$W_{out}=\text{weights matrix relative to output layer}$" />\
<img src="https://latex.codecogs.com/svg.latex?$W_{hid}=\text{weights&space;matrix&space;relative&space;to&space;hidden&space;layer}$" title="$W_{hid}=\text{weights matrix relative to hidden layer}$" />\
<img src="https://latex.codecogs.com/svg.latex?$b_{out}&space;=&space;\text{bias&space;vector&space;for&space;output&space;layer}$" title="$b_{out} = \text{bias vector for output layer}$" />\
<img src="https://latex.codecogs.com/svg.latex?$b_{hid}&space;=&space;\text{bias&space;vector&space;for&space;hidden&space;layer}$" title="$b_{hid} = \text{bias vector for hidden layer}$" />\
<img src="https://latex.codecogs.com/svg.latex?$\eta=\text{learning&space;rate}$" title="$\eta=\text{learning rate}$" />\
<img src="https://latex.codecogs.com/svg.latex?$\lambda=\text{regularization&space;rate}$" title="$\lambda=\text{regularization rate}$" />\
<img src="https://latex.codecogs.com/svg.latex?$\alpha=\text{momentum&space;rate}$" title="$\alpha=\text{momentum rate}$" />

### Logistic function used

<img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}\space\text{\qquad&space;[sigmoid&space;function]}" title="f(x) = \frac{1}{1+e^{-x}}\space\text{\qquad [sigmoid function]}" />

### Error correction coefficent

<img src="https://latex.codecogs.com/svg.latex?$$\delta_{out}&space;=(Y_{target}&space;-&space;Y_{out})&space;*&space;f'(Y_{out})$$" title="$$\delta_{out} =(Y_{target} - Y_{out}) * f'(Y_{out})$$" />\
\
<img src="https://latex.codecogs.com/svg.latex?$$\delta_{hid}&space;=&space;W_{out}*\delta_{out}*f'(y_{hid})$$" title="$$\delta_{hid} = W_{out}*\delta_{out}*f'(y_{hid})$$" />

### The correction factor (including regularization and momentum)

<img src="https://latex.codecogs.com/svg.latex?$$\Delta&space;W_{out}&space;=&space;\eta\space&space;(Y_{hid}*\delta_{out}-\lambda*W_{out}&plus;\alpha&space;*&space;\Delta&space;W_{out,old})$$" title="$$\Delta W_{out} = \eta\space (Y_{hid}*\delta_{out}-\lambda*W_{out}+\alpha * \Delta W_{out,old})$$" />\
\
<img src="https://latex.codecogs.com/svg.latex?$$\Delta&space;W_{hid}&space;=&space;\eta\space&space;(X*\delta_{hid}-\lambda*W_{hid}&plus;\alpha&space;*&space;\Delta&space;W_{hid,old})$$" title="$$\Delta W_{hid} = \eta\space (X*\delta_{hid}-\lambda*W_{hid}+\alpha * \Delta W_{hid,old})$$" />

### Updating the weights

<img src="https://latex.codecogs.com/svg.latex?$$W_{out}&space;=&space;\Delta&space;W_{out}&space;&plus;&space;W_{out,&space;old}$$" title="$$W_{out} = \Delta W_{out} + W_{out, old}$$" />\
\
<img src="https://latex.codecogs.com/svg.latex?$$W_{hid}&space;=&space;\Delta&space;W_{hid}&space;&plus;&space;W_{hid,&space;old}$$" title="$$W_{hid} = \Delta W_{hid} + W_{hid, old}$$" />

### Updating the bias

<img src="https://latex.codecogs.com/svg.latex?$$b_{out}&space;=&space;\eta&space;*\sum&space;\delta_{out}$$" title="$$b_{out} = \eta *\sum \delta_{out}$$" />\
\
<img src="https://latex.codecogs.com/svg.latex?$$b_{hid}&space;=&space;\eta&space;*\sum&space;\delta_{hid}$$" title="$$b_{hid} = \eta *\sum \delta_{hid}$$" />
