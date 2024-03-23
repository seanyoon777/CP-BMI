import numpy as np 
from sklearn.metrics import accuracy_score, cohen_kappa_score

def get_baseline_metrics(model, X_test, y_test, verbose=True): 
    y_pred = model.predict(X_test, verbose=verbose).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    baseline_accuracy = accuracy_score(y_true, y_pred)
    baseline_kohen = cohen_kappa_score(y_true, y_pred)
    return baseline_accuracy, baseline_kohen
    
def get_abstain_metrics(prediction_set, test_smx, y_test, verbose=False): 
    # If no abstaining or multiple predictions, what accuracy do we get?
    idx_single = np.sum(prediction_set, axis=1) < 2
    if np.sum(idx_single) == 0: 
        if verbose: 
            print('No single predictions, so abstaining model accuracy cannot be computed.')
        return 1
    return np.mean(test_smx[idx_single,...].argmax(1) == y_test[idx_single,...].argmax(1))
    
def get_set_size(prediction_sets, n_classes): 
    set_sizes = {} 
    for i in range(n_classes+1):
        set_sizes[i] = np.sum(np.sum(prediction_sets, axis=1)==i)
    return set_sizes 

def get_average_size(set_sizes): 
    weighted_sum = 0
    for k in set_sizes.keys(): 
        weighted_sum += set_sizes[k] * k
    return weighted_sum / sum(set_sizes.values())
        
def get_coverage(y_test, prediction_sets):
    coverage = 0
    total = 0
    for i, pred_set in enumerate(prediction_sets): 
        label = y_test.argmax(axis=1)[i]
        if pred_set[label]: 
            coverage += 1
        total += 1
    return coverage / total

def report(model, X_test, y_test, prediction_sets, n_classes): 
    coverage = get_coverage(y_test, prediction_sets)
    print(f'Empirical Coverage: {coverage}')
    
    baseline_accuracy, baseline_kohen = get_baseline_metrics(model, X_test, y_test)
    abstain_accuracy, abstain_kohen = get_abstain_metrics(model, prediction_sets, X_test, y_test)
    print(f'Baseline accuracy without abstaining: {baseline_accuracy}')
    print(f'Accuracy with abstaining: {abstain_accuracy}')
    print(f'Baseline Kohen\'s Kappa without abstaining: {baseline_kohen}')
    print(f'Kohen\'s Kappa with abstaining: {abstain_kohen}')
    set_sizes = get_set_size(prediction_sets, n_classes)
    single_proportion = set_sizes[1] / sum(set_sizes.values())
    print(f'Proportion of single prediction: {single_proportion}')
    print(f'Total Predictions:  {len(prediction_sets)}')
    average_set_size = get_average_size(set_sizes)    
    print(f'Average set size: {get_average_size(set_sizes)}')
    for i in set_sizes.keys(): 
        print(f'{i} Predictions: {set_sizes[i]}')
    
    return baseline_accuracy, abstain_accuracy, baseline_kohen, abstain_kohen, 1-single_proportion, coverage, average_set_size, set_sizes
