from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import numpy as np 
from metrics import * 

def conformalRAPS(val_smx, y_val, test_smx, alpha, lam_reg, k_reg, rand=True, disallow_zero_sets=False, verbose=False): 
    reg_vec = np.array(k_reg*[0,] + (y_val.shape[1]-k_reg)*[lam_reg,])[None,:]
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    val_labels = y_val.argmax(axis=1)
    n = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_L = np.where(val_pi == val_labels[:,None])[1]
    val_scores = val_srt_reg.cumsum(axis=1)[np.arange(n),val_L] - np.random.rand(n)*val_srt_reg[np.arange(n),val_L]
    # Get the score quantile
    qhat = np.quantile(val_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    # Deploy 
    n_test = test_smx.shape[0]
    test_pi = test_smx.argsort(1)[:,::-1]
    test_srt = np.take_along_axis(test_smx, test_pi, axis=1)
    test_srt_reg = test_srt + reg_vec 
    test_srt_reg_cumsum = test_srt_reg.cumsum(axis=1)
    indicators = (test_srt_reg.cumsum(axis=1) - np.random.rand(n_test,1)*test_srt_reg) <= qhat if rand else test_srt_reg.cumsum(axis=1) - test_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,test_pi.argsort(axis=1),axis=1)
    
    return prediction_sets, qhat

def get_conformity_scores(val_smx, y_val, test_smx, y_test, lam_reg=.02, k_reg=1): 
    reg_vec = np.array(k_reg*[0,] + (y_val.shape[1]-k_reg)*[lam_reg,])[None,:]
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    val_labels = y_val.argmax(axis=1)
    n = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    
    val_L = np.where(val_pi == val_labels[:,None])[1]
    val_scores = val_srt_reg.cumsum(axis=1)[np.arange(n),val_L] - val_srt_reg[np.arange(n),val_L]
    # Deploy 
    test_labels = y_test.argmax(axis=1)
    n_test = test_smx.shape[0]
    test_pi = test_smx.argsort(1)[:,::-1]
    test_srt = np.take_along_axis(test_smx, test_pi, axis=1)
    test_srt_reg = test_srt + reg_vec 
    test_L = np.where(test_pi == test_labels[:,None])[1]
    test_scores = test_srt_reg.cumsum(axis=1)[np.arange(n_test),test_L] - test_srt_reg[np.arange(n_test),test_L]
    return val_scores, test_scores


