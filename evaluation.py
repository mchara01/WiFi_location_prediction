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
    # Shuffle the indices
    indices = random_generator.permutation(rows)
    k_indices = np.array_split(indices, k)
    return k_indices

def k_fold_indices(n_folds,n_instances):
    """ Used for Step 3 - Evaluation, for cross validation. 
    Generates n_folds possible combinations of indices for training, testing, and validation.
    
    Args:
        n_folds (int): Number of outer folds
        n_instances (int): Total number of instances (i.e. rows of data)

    Returns:
        folds (list):
        [ [[test1], [train1]], [[test2], [train2]], ... ,[[test_nfold], [train_nfold]] ]

        1. Each row represents 1 fold.
        2. 1st element in each row: test indices.
        3. 2nd element in each row: train indices.
    """

    # Initialise
    folds = []

    # Split the dataset into n_folds
    split_indices = k_indices_split(n_folds, n_instances)

    for k in range(n_folds):

        # Pick k as test dataset
        test_indices = split_indices[k]

        # Remaining folds will belong to the training dataset
        training_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        # Append into folds
        folds.append([test_indices, training_indices])
    
    return folds


def nested_k_fold_indices(n_outer_folds, n_inner_folds, n_instances):
    """ Used for Step 4 - Pruning, for nested cross-validation 
    Generates nested n_folds possible combinations of indices 
    for training, testing, and validation.
    
    Args:
        n_outer_folds (int): Number of outer folds
        n_inner_folds (int): Number of inner folds
        n_instances (int): Total number of instances (i.e. rows of data)

    Returns:
        folds (list):
        [[test1, [ [[train1 indices], [val1 indices]], [[train2 indices],[val2 indices]], ... ],
         [test2, [ [[train1 indices], [val1 indices]], [[train2 indices],[val2 indices]], ... ],
         ... ]

        1. Each row represents the outer fold.
        2. 1st element in each row: test indices
        3. 2nd element in each row: appended list indices for each of the inner folds in this layer.
    """
    # Initialise 
    folds = []

    # Split the dataset into n_outer_folds
    outer_split_indices = k_indices_split(n_outer_folds, n_instances)

    for k in range(n_outer_folds):

        # Pick k as test dataset
        test_indices = outer_split_indices[k]
        
        # Remaining dataset
        remaining_indices = np.hstack(outer_split_indices[:k] + outer_split_indices[k+1:])
        inner_split_indices = k_indices_split(n_inner_folds, len(remaining_indices))

        # Generate indices to split the remaining dataset into training and validation sets
        inner_fold = []
        for j in range(n_inner_folds):
            
            val_indices = remaining_indices[inner_split_indices[j]]
            train_intermediate_idx = np.hstack(inner_split_indices[:j]+inner_split_indices[j+1:])
            train_indices = remaining_indices[train_intermediate_idx]

            inner_fold.append([train_indices,val_indices])
            
            
        folds.append([test_indices,inner_fold])

    return folds

def visualize_k_fold():
    """ Function prints the nested k-fold cross validation indices
    """
    
    for index, outer_fold in enumerate(nested_k_fold_indices(10, 10, 30)):
        print("K:", index+1)
        print("Test:", outer_fold[0])

        for inner_fold in outer_fold[1]:
            print("Train:", inner_fold[0])
            print("Validation:", inner_fold[1])
        
    return None

def evaluate(x_test,y_test, trained_tree):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        test_set (numpy.ndarray): test dataset
        trained_tree (object node): trained decision tree

    Returns:
        float : the accuracy
    """

    y_prediction = trained_tree.predict(x_test)
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