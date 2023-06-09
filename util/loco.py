# LOCO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import get_scorer


def loco_importance(
    model, X, y, variables, scoring, n_repeats = 1
):
    # 1-Randomly split training data in 2 (Dn_1 and Dn_2)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    # 2-Train a classification algorithm to estimate f_hat_n_1 (function to make predictions, based on training with Dn_1)
    base_model = clone(model).fit(X_train, y_train)
    # 3- With this trained model on Dn_1, make predictions on Dn_2. 
    # 4-For each patient
    # 	-Calculate the loss{Y, f_hat_n_1} (Difference between True and obtained prediction)
    #    	This gives us all the “baseline” errors —> 300 estimated errors obtained by training a model USING ALL VARIABLES.
    loss_fn   = get_scorer(scoring)
    base_loss = loss_fn(base_model, X_test, y_test)
    # 5-For 10 times (one for each of the 10 selected variables): [[[we have to choose 10 potentially important variables]]]]
    # 	  -Train a classification algorithm to estimate f_hat_n_1 , (but this time REMOVE variable J from Dn_1 before training) 
    # 	  -With this trained model on Dn_1(-j), make predictions on Dn_2.
    #          For each patient in Dn_2 (300 patients):
    #             	-Calculate the loss{Y, f_hat_n_1(-j)} (Difference between True and obtained prediction by removing variable J)
    # 		-Calculate difference between: the “baseline" error on Patient [i] and the error obtained without var. J
    # 	  We obtain a dataset of differences between baseline erros and obtained errors without var J
    # 	  -Calculate the median of these dataset
    # 	  -For B bootstraped datasets on the dataset of differences obtained:
    # 		    -Calculate the median of these dataset of differences
    res = list()
    for var in variables:
        X_train_noj, X_test_noj = X_train.drop(columns=var), X_test.drop(columns=var)
        model_noj = clone(model).fit(X_train_noj, y_train)
        # by looking here https://elearning.uniroma1.it/mod/forum/discuss.php?d=212989#p309265 
        # I believe that maybeee we shouldn't be doing bootstrap
        for b in range(n_repeats):
            X_test_noj_sample , y_test_sample = sample(X_test_noj, y_test)
            loss_noj = loss_fn(model_noj, X_test_noj_sample, y_test_sample)
            res.append((var, b, loss_noj))
            
    # 6- Obtain a (bootstrapped) distribution of Medians of the differences between baseline errors and errors w/o J.	
    res = pd.DataFrame().from_records(res, columns = ['variable', 'repeat', 'loss'])
    res.loss -= base_loss
    
    res = res.groupby('variable').loss.agg(np.median)
    return res
    # 7- Build a confidence interval of this diestro ( account for FWER, so divide alpha/10 i think)
    
    
def sample(*args):
    df = args[0]
    idx = np.random.randint(df.shape[0], size=df.shape[0])
    return [df.iloc[idx] for df in args]