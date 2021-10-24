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

def evaluate(test_set, trained_tree):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        test_set (numpy.ndarray): test dataset
        trained_tree (object node): trained decision tree

    Returns:
        float : the accuracy
    """

    X_test, y_test = np.split(test_set,[-1],axis=1)
    y_prediction = trained_tree.predict(X_test)
    assert len(y_test) == len(y_prediction)  
    
    try:
        return np.sum(y_test == y_prediction) / len(y_test)
    except ZeroDivisionError:
        return 0.



def confusion_matrix(y_truth, y_prediction, class_labels=None):
    """ Compute the confusion matrix.
        
    Args:
        y_truth (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_truth, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row), 
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_truth == label)
        truth = y_truth[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    
        

    return confusion