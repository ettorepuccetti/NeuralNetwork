# coding: utf-8
from NN_tools import *

def train_model(X, y, X_valid=None, y_valid=None, neurons_hidden=5, epochs=500, lr=0.1,
                reg_lambda=0.0, momentum_alpha=0.0, validation_split = 0.0, threshold=0.5):
    
    X,y = datapreprocessing(X,y)

    if (X_valid is not None and y_valid is not None):
        X_valid, y_valid = datapreprocessing(X_valid, y_valid)
    else: 
        X,X_valid,y,y_valid = train_test_split(X,y,test_size=validation_split,shuffle=True)

    # adatto la rete alla dimensione di input e output
    n_input_layer = X.shape[1]
    n_hidden_layer = neurons_hidden 
    n_output_layer = y.shape[1] 

    # inizializzazione pesi e bias
    np.random.seed(0)
    mu, sigma = 0, 0.2
    Wh=np.random.normal(loc=mu, scale=sigma, size=(n_input_layer,n_hidden_layer)) / np.sqrt(n_input_layer)
    bh=np.random.normal(loc=mu, scale=sigma, size=(1,n_hidden_layer))
    Wout=np.random.normal(loc=mu, scale=sigma, size=(n_hidden_layer,n_output_layer)) / np.sqrt(n_hidden_layer)
    bout=np.random.normal(loc=mu, scale=sigma, size=(1,n_output_layer))
    
    accuracy_values_train = []
    loss_values_train = []
    accuracy_values_valid = []
    loss_values_valid = []
    model = {}
    
    # per il momentum
    dWout = 0
    dWh = 0
    
    for i in range(0, epochs):

        #passo Forward
        net_hidden_layer = np.dot(X,Wh) + bh
        out_hidden_layer = sigmoid(net_hidden_layer)
        net_output_layer = np.dot(out_hidden_layer,Wout) + bout
        out_output_layer = sigmoid(net_output_layer)
        
        #Backpropagation
            #calcolo dei delta
                #output
        d_output = (y - out_output_layer) * derivatives_sigmoid(out_output_layer)
                #hidden
        d_hiddenlayer = d_output.dot(Wout.T) * derivatives_sigmoid(out_hidden_layer)
        
        # calcolo DELTA_W e aggiornamento W e b
            # per il momentum
        dWout_old = dWout 
        dWh_old = dWh
        
            #output layer
        dWout = out_hidden_layer.T.dot(d_output) * lr
        dWout += -reg_lambda * lr * Wout                   # regolarizzazione
        dWout += momentum_alpha * dWout_old                # momentum
        Wout += dWout                                      # aggiornamento pesi
        bout += np.sum(d_output, axis=0,keepdims=True) *lr # aggiornamento bias 
        
            # hidden layer
        dWh = X.T.dot(d_hiddenlayer) * lr
        dWh += -reg_lambda * lr * Wh                          # regolarizzazione
        dWh += momentum_alpha * dWh_old                       # momentum
        Wh += dWh                                             # aggiornamento pesi
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr # aggiornamento bias 

        
        #### statistiche

        #compute the binary output for the training set
        output_binary = [0 if (x<threshold) else 1 for x in out_output_layer]
        accuracy_values_train.append(accuracy_score(y_pred=output_binary,y_true=y))
        loss_values_train.append(MSE(y_pred=out_output_layer, y_true=y))
        
            #predict the values for the validation set
        if (len(X_valid) > 0):
            output_validation = predict_values({'Wh': Wh, 'bh': bh, 'Wout': Wout, 'bout': bout}, X=X_valid)
            output_validation_binary = [0 if (x<threshold) else 1 for x in output_validation]
            accuracy_values_valid.append(accuracy_score(y_pred=output_validation_binary, y_true=y_valid))
            loss_values_valid.append(MSE(y_pred=output_validation, y_true=y_valid))
            
        
    model = { 'Wh': Wh, 'bh': bh, 'Wout': Wout, 'bout': bout,
            'accuracy_values_train': accuracy_values_train, 'loss_values_train': loss_values_train,
            'accuracy_values_valid': accuracy_values_valid, 'loss_values_valid': loss_values_valid,
            'hyperparams': [lr, reg_lambda, momentum_alpha] }
    
    return model





def train_model_regression(X, y, X_valid=None, y_valid=None, neurons_hidden=5, epochs=500, lr=0.1,
                reg_lambda=0.0, momentum_alpha=0.0, validation_split = 0.0):
    
    X,y = datapreprocessing(X,y)

    if (X_valid is not None and y_valid is not None):
        X_valid, y_valid = datapreprocessing(X_valid, y_valid)
    else: 
        X,X_valid,y,y_valid = train_test_split(X,y,test_size=validation_split,shuffle=True)

    # adatto la rete alla dimensione di input e output
    n_input_layer = X.shape[1]
    n_hidden_layer = neurons_hidden 
    n_output_layer = y.shape[1] 

    # inizializzazione pesi e bias
    np.random.seed(7)
    mu, sigma = 0, 0.1
    Wh=np.random.normal(loc=mu, scale=sigma, size=(n_input_layer,n_hidden_layer)) / np.sqrt(n_input_layer)
    bh=np.random.normal(loc=mu, scale=sigma, size=(1,n_hidden_layer))
    Wout=np.random.normal(loc=mu, scale=sigma, size=(n_hidden_layer,n_output_layer)) / np.sqrt(n_hidden_layer)
    bout=np.random.normal(loc=mu, scale=sigma, size=(1,n_output_layer))
    
    loss_values_train = []
    loss_values_valid = []
    model = {}
    
    # per il momentum
    dWout = 0
    dWh = 0
    
    for i in range(0, epochs):

        #passo Forward
        net_hidden_layer = np.dot(X,Wh) + bh
        out_hidden_layer = sigmoid(net_hidden_layer)
        net_output_layer = np.dot(out_hidden_layer,Wout) + bout
        out_output_layer = net_output_layer
 
               
        #Backpropagation
            #calcolo dei delta
                
                #output
                
        denomin = np.sqrt(np.sum(np.square(y - out_output_layer),axis=1, keepdims=True))
        # raddoppio la dimensione di "denomin" per poter dividere la matrice con i risultati di (target - output)
        denomin = np.insert(denomin, [1], denomin, axis = 1)
        
        d_output = (y - out_output_layer) / denomin
                
                #hidden
        d_hiddenlayer = d_output.dot(Wout.T) * derivatives_sigmoid(out_hidden_layer)
        
        
        #calcolo DELTA_W e aggiornamento W e b
        
        # per il momentum
        dWout_old = dWout 
        dWh_old = dWh
        
        dWout = out_hidden_layer.T.dot(d_output) * lr / X.shape[0]
        dWout += -reg_lambda * lr * Wout                   # regolarizzazione
        dWout += momentum_alpha * dWout_old                # momentum
        Wout += dWout                                      # aggiornamento pesi
        bout += np.sum(d_output, axis=0,keepdims=True) *lr / X.shape[0] # aggiornamento bias 
        
        dWh = X.T.dot(d_hiddenlayer) * lr / X.shape[0]
        dWh += -reg_lambda * lr * Wh                          # regolarizzazione
        dWh += momentum_alpha * dWh_old                       # momentum
        Wh += dWh                                             # aggiornamento pesi
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr / X.shape[0] # aggiornamento bias 
        
        
        #### statistiche ####
        # non calcolo il valore della funzione di loss per le prime 30 epoche,
        # altrimenti il grafico, per inserire i grandi valori iniziali, non mi fa apprezzare le differenze sulle successive epoche
        if i > 30:
            loss_values_train.append(MEE(y_pred=out_output_layer, y_true=y))
        else:
            loss_values_train.append(None)
        
        # risultati per il validation set, di cui calcolo i valori di output con il modello ottenuto in ogni epoca
        if (len(X_valid) > 0 and len(y_valid) > 0):
            if (i > 30):
                output_validation = predict_values({'Wh': Wh, 'bh': bh, 'Wout': Wout, 'bout': bout}, X=X_valid, classification=False)
                loss_values_valid.append(MEE(y_pred=output_validation, y_true=y_valid))
            else:
                loss_values_valid.append(None)
        
    model = { 'Wh': Wh, 'bh': bh, 'Wout': Wout, 'bout': bout,
             'loss_values_train': loss_values_train, 'loss_values_valid': loss_values_valid,
             'hyperparams': [lr, reg_lambda, momentum_alpha]}
    
    return model


#difinita qui perchè si appoggia sulla funzione 'train_model_regression' non importata nell'altro script.
def cross_validation(hyperparams, cup_tr_X, cup_tr_y, K_fold):
    model_loss_valid_values = []
    model_loss_train_values = []
    for k in range(1,K_fold+1):
        hyperparam = {'lr':hyperparams[0], 'momentum_alpha': hyperparams[1], 'reg_lambda':hyperparams[2]}
        X_train, X_valid, y_train, y_valid = split_cross_validation(cup_tr_X,cup_tr_y,k,K_fold)
        model = train_model_regression(X=X_train, y=y_train, epochs=1800, neurons_hidden=16,
                                        validation_split = 0.0, **hyperparam)
        y_pred = predict_values(model, X=X_valid, classification=False)
        model_loss_valid_values.append(MEE(y_pred=y_pred, y_true=y_valid))
        model_loss_train_values.append(model['loss_values_train'][-1])
    average_Mee_valid = np.mean(np.array(model_loss_valid_values))
    average_Mee_train = np.mean(np.array(model_loss_train_values))
    return ({'hyperparam': hyperparams, 'loss_valid': average_Mee_valid, 'loss_train': average_Mee_train})
    