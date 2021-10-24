import numpy as np
from numpy.random import default_rng



def k_indices_split(k, rows, random_generator=default_rng()):
    """ Splitting indices into k fold randomly

    Args:
        k (int): k splits
        rows (int): Number of rows to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). [list1, list2, list3] that each contains the indices,
        where list1, list2, list3 respectively contains a list of indices
    """

    indices = random_generator.permutation(rows)
    k_indices = np.array_split(indices, k)
    return k_indices


def nested_k_fold_cross_validation(n_folds, n_instances):
    """ Generate train, test, and validation indices at each fold.
         Testing set stays the same at each fold = k
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of list of length n_folds. 
        [[train1, test1, val1], [train2, test2, val2], ...] 
        1. Main list contains each list k
        2. At each list k, contains train set, test set, and validation set
    """
    # split the dataset into k splits
    split_indices = k_indices_split(n_folds, n_instances)
    folds = []
    

    for k in range(n_folds):

        
        inner_folds = []
        # pick k as test
        test_indices = split_indices[k]
       
        # loop through inner list to find train and validation set
        for j in range(n_folds):
            if k == j:
                continue
            else:
                validation_indices = split_indices[j]
                train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
                inner_folds.append([train_indices, test_indices, validation_indices])
        folds.append(inner_folds)

    return folds

     

  
