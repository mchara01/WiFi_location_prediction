import numpy as np
import copy
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
        
        # create a new list and delete test set from that element
        non_test_split_indices = copy.deepcopy(split_indices)
        del non_test_split_indices[k]

        # loop through inner list to find train and validation set
        for j in range(n_folds - 1):
            validation_indices = non_test_split_indices[j]
            train_indices = np.hstack(non_test_split_indices[:j] + non_test_split_indices[j+1:])
            inner_folds.append([train_indices, test_indices, validation_indices])
        folds.append(inner_folds)

    return folds

def visualize_k_fold():
    k = 0
    for outer_fold in nested_k_fold_cross_validation(4, 30):
        print("K:" + str(k))
        for inner_fold in outer_fold:
            print("Train:") 
            print(inner_fold[0])
            print("Test:") 
            print(inner_fold[1])
            print("Validation:") 
            print(inner_fold[2])
        k+=1
    return None



  
