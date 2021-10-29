"""
This module contains all implementations of dataset-related methods,
such as reading, splitting and cross-validating which are imported
in other modules of this project.
"""
import numpy as np

from numpy.random import default_rng


def visualize_k_fold():
    """ Function prints the nested k-fold cross validation indices into terminal."""
    for index, outer_fold in enumerate(nested_k_fold_indices(10, 10, 30)):
        print("K:", index + 1)
        print("Test:", outer_fold[0])

        for inner_fold in outer_fold[1]:
            print("Train:", inner_fold[0])
            print("Validation:", inner_fold[1])


def read_dataset(file_path):
    """ Read the dataset from the specified file path.

    Read a dataset from a given file path, split it into training and test subsets and return them
    along with the unique classes in the dataset.

    Args:
        file_path (str): The relative or absolute path to the dataset file on the local file system.

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array. 
                - x is a numpy array with shape (N, K), 
                    where N is the number of instances
                    K is the number of features/attributes
                - y is a numpy array with shape (N, 1), and should be integers from 0 to C-1
                    where C is the number of classes 
                - classes : a numpy array with shape (C, ), which contains the 
                    unique class labels corresponding to the integers in y
    """
    # Load .txt file from file_path
    data = np.loadtxt(file_path)
    # Divide dataset into its features(x) and labels(y)
    x = data[:, :-1]
    y = data[:, -1]
    # Find classes and their indices from y
    classes, y = np.unique(y, return_inverse=True)
    return x, y, classes


def k_indices_split(k, rows, random_generator=default_rng()):
    """ Splitting indices into k folds randomly.

    Args:
        k (int): Number(k) of splits
        rows (int): Number of rows to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list containing k sub-lists (i.e., [list1, list2, list3] for k=3).
    """
    # Shuffle the indices
    indices = random_generator.permutation(rows)
    # Split the shuffled indices list into K sub-lists
    k_indices = np.array_split(indices, k)
    return k_indices


def k_fold_indices(n_folds, n_instances):
    """ Used for Step 3 - Evaluation, for cross validation.

    Generates n_folds possible combinations of indices for training, validation, and testing.
    
    Args:
        n_folds (int): Number of outer folds
        n_instances (int): Total number of instances (i.e. rows of data)

    Returns:
        folds (list): [ [[test1], [train1]], [[test2], [train2]], ... , [[test_nfold], [train_nfold]] ]
        1. Each row represents 1 fold.
        2. 1st element in each row: test indices.
        3. 2nd element in each row: train indices.
    """
    # Initialise folds list
    folds = list()

    # Split the dataset into n_folds
    split_indices = k_indices_split(n_folds, n_instances)

    # Iterate through the n_folds
    for k in range(n_folds):
        # Pick k as test dataset
        test_indices = split_indices[k]

        # Remaining (n_folds - n_instances) folds will belong to the training dataset
        training_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

        # Append into folds
        folds.append([test_indices, training_indices])

    return folds


def nested_k_fold_indices(n_outer_folds, n_inner_folds, n_instances):
    """ Used for Step 4 - Pruning, for nested cross-validation.

    Generates nested n_folds possible combinations of indices for training, testing, and validation.
    
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
    # Initialise folds list
    folds = list()

    # Split the dataset into n_outer_folds
    outer_split_indices = k_indices_split(n_outer_folds, n_instances)

    for k in range(n_outer_folds):
        # Pick k as test dataset
        test_indices = outer_split_indices[k]

        # Remaining dataset
        remaining_indices = np.hstack(outer_split_indices[:k] + outer_split_indices[k + 1:])
        inner_split_indices = k_indices_split(n_inner_folds, len(remaining_indices))

        # Generate indices to split the remaining dataset into training and validation sets
        inner_fold = list()
        for j in range(n_inner_folds):
            # Get the validation subset
            val_indices = remaining_indices[inner_split_indices[j]]
            # Calculate remaining rows
            train_intermediate_idx = np.hstack(inner_split_indices[:j] + inner_split_indices[j + 1:])
            # Get the train subset
            train_indices = remaining_indices[train_intermediate_idx]
            # Append the training and validation indices list
            inner_fold.append([train_indices, val_indices])

        folds.append([test_indices, inner_fold])

    return folds
