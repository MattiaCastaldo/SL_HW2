# LOCO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import get_scorer

def log_loss(model, X, y):
    y_hat = model.predict_proba(X)[:,1]
    return -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat))

def loco_importance(
    model, X, y, variables, scoring = 'unused', n_repeats = 1
):
    # 1-Randomly split training data in 2 (Dn_1 and Dn_2)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    # 2-Train a classification algorithm to estimate f_hat_n_1 (function to make predictions, based on training with Dn_1)
    base_model = clone(model).fit(X_train, y_train)
    # 3- With this trained model on Dn_1, make predictions on Dn_2. 
    #   This gives us all the “baseline” errors —> len(Dn-1) estimated errors obtained by training a model USING ALL VARIABLES.
    base_loss = log_loss(base_model, X_test, y_test)
    # 4-For 10 times (one for each of the 10 selected variables): [[[we have to choose 10 potentially important variables]]]]
    df_medians = pd.DataFrame() # (shape:bx10) colum J contains the b medians obtained by removing the Jth variable.
    for var in variables:
        median_var_pe = list() #median variations in prediction error of variable J
        X_train_noj, X_test_noj = X_train.drop(columns=var), X_test.drop(columns=var) #remove var J 
        model_noj = clone(model).fit(X_train_noj, y_train) #train model with new data
        loss_noj = log_loss(model_noj, X_test_noj, y_test) #obtain patient wise loss
        delta_loss = abs(loss_noj - base_loss) # obtain abs difference between base loss and loss w/o var J
        median_var_pe.append(delta_loss.median()) #compute the median of this distro and store it
        for b in range(n_repeats):
            b_data = np.random.choice(delta_loss, size=len(delta_loss), replace=True) # create bootstrapped dataset
            median_var_pe.append(b_data.median()) #compute the median of this distro and store it
    # 5- Obtain a (bootstrapped) distribution of Medians of the differences between baseline errors and errors w/o J.
        bootstrap_medians = pd.Series(median_var_pe)      
    	df_medians[var] = bootstrap_medians

    # 6- Build a confidence interval for each of this columns in df_medians (and account for FWER, so divide alpha/10)
    confidence_level = 0.95  # Desired confidence level
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha/10 * 100
    upper_percentile = (1 - alpha/10) * 100
    confidence_intervals = {}
    for i in df_medians.columns:
        confidence_interval = np.percentile(df_medians[i], [lower_percentile, upper_percentile])
        confidence_intervals[i] = confidence_interval
