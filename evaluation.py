"""
The evaluation module contains the implementation of all metrics used
during evaluation (accuracy, confusion matrix, precision, recall, f1-score)
and pruning.
"""
import numpy as np

from decision_tree import DecisionTree
from dataset import *


def cross_validation(x, y, k_folds):
    """Step 3 - Evaluation with simple cross validation.
    
    Computes and appends the confusion matrix generated from each fold.
    At each fold, the function trains the decision tree with training data.
    Its performance is then evaluated with the test data, and appended.

    Args:
        x (np.ndarray): Test dataset label. 
        y (np.ndarray): Predicted label. 
        k_folds (int): Number of folds.
        
    Returns:
        list: results is a list of length = k_folds
            - Each element is the confusion matrix corresponding to a fold. 
        list: depth is a list of length = k_folds
            - Each element is the decision tree depth corresponding to a fold. 
    """
    indices_list = k_fold_indices(k_folds, len(x))

    # Initialise the list that will store the: (i) confusion matrix from each fold (ii) depth of the decision tree from each fold
    result_dt, depth = list(), list()

    # Iterate through each fold
    for k in indices_list:
        # Testing dataset
        test_idx = k[0]
        x_test = x[test_idx]
        y_test = y[test_idx]

        # Training dataset
        train_idx = k[1]
        x_train = x[train_idx]
        y_train = y[train_idx]

        # Training the decision tree
        k_decision_tree = DecisionTree()
        k_decision_tree.train(x_train, y_train)

        # Predict the label and evaluate performance
        y_predicted = k_decision_tree.predict(x_test)
        final_cm = confusion_matrix(y_test, y_predicted)

        # Append to the two lists the Confusion Matrix and the Depth
        result_dt.append(final_cm)
        depth.append(k_decision_tree.final_depth())

    return result_dt, depth


def pruning_nested_cross_validation(x, y, outer_fold, inner_fold):
    """Step 4 - Pruning with Nested Cross-Validation.

    Args:
        x (np.ndarray): Features subset.
        y (np.ndarray): Labels subset.
        outer_fold (int): Number of outer folds during cross-validation.
        inner_fold (int): Number of inner folds during cross-validation.

    Returns:
        result_dt: list of length = k_folds
            - Each element is the confusion matrix corresponding to a fold. 
        depth: list of length = k_folds
            - Each element is the decision tree depth corresponding to a fold. 
    """
    # Generates nested k-folds possible combinations of indices for training, testing, and validation
    indices_list = nested_k_fold_indices(outer_fold, inner_fold, len(x))
    result_dt, depth, acc_finals = list(), list(), list()
    counter = 1 # outer fold counter used for printing
    # Iterate through the indices
    for k in indices_list:
        # Testing fold
        test_idx = k[0]
        x_test = x[test_idx]
        y_test = y[test_idx]
        # Initialise best DecisionTree and its accuracy
        best_dt = None
        best_acc = None
        for j in k[1]:  # k[1]: [ [[train1 indices],[val1 indices]] , [[train2 indices],[val2 indices]] ...]
            # Extract train and validation indices
            train_indices = j[0]
            val_indices = j[1]
            x_train = x[train_indices]
            y_train = y[train_indices]
            x_val = x[val_indices]
            y_val = y[val_indices]
            # Create a DTree and train it
            current_decision_tree = DecisionTree()
            current_decision_tree.train(x_train, y_train)
            # Recursively prune the DTree
            recursive_pruning_simulation(current_decision_tree, current_decision_tree.root_node, x_val, y_val)
            # Make predictions on the pruned DTree
            y_predict = current_decision_tree.predict(x_val)
            # Calculate its accuracy
            acc = evaluate(y_val, y_predict)
            # Check if its the first time we check or it has the best accuracy up to now
            if best_acc is None or acc > best_acc:
                # New best accuracy and DTree
                best_acc = acc
                best_dt = current_decision_tree
        # Make predictions on test features using the most accurate DTree created above
        y_predicted = best_dt.predict(x_test)
        # Calculate Confusion Matrix
        final_cm = confusion_matrix(y_test, y_predicted)
        print('Confusion Matrix and Depth from Fold ', counter)
        print(final_cm)
        counter+=1
        # Calculate accuracy and append to list
        acc_finals.append(evaluate(y_test, y_predicted))
        # Append its confusion matrix
        result_dt.append(final_cm)
        print(best_dt.final_depth())
        # Append its depth after pruning
        depth.append(best_dt.final_depth())

    return result_dt, depth


def recursive_pruning_simulation(current_decision_tree, tree_node, x, y):
    """Step 4 - Recursive pruning simulation.

    Recursively find nodes directly connected to two leaves, evaluate the benefits
    on the validation error of substituting these nodes with a single leaves. If a single
    leaf reduces the validation error, then the node in pruned and replaced by a single leaf.

    Args:
        current_decision_tree (DecisionTree): Object of class Decision Tree.
        tree_node (TreeNode): Object of class Decision Tree.
        x (np.ndarray): Features subset.
        y (np.ndarray): Labels subset.

    Returns:
        list: list of length = k_folds
            - Each element is the confusion matrix corresponding to a fold.
        depth: list of length = k_folds
            - Each element is the decision tree depth corresponding to a fold.
    """
    # Recursion ending step. If we reached a leaf, its time to start pruning
    if tree_node.leaf:
        return
    # Check whether the current node's children are leafs and continue with recursion if not
    if not tree_node.left.leaf:
        recursive_pruning_simulation(current_decision_tree, tree_node.left, x, y)
    if not tree_node.right.leaf:
        recursive_pruning_simulation(current_decision_tree, tree_node.right, x, y)
    # Check if both children are leafs first
    if tree_node.left.leaf and tree_node.right.leaf:
        # Make predictions with current decision tree
        y_predict = current_decision_tree.predict(x)
        # Calculate the accuracy of predictions
        orig_val = evaluate(y, y_predict)
        # Get the label counts of each child
        left_counts = tree_node.left.label_counts
        right_counts = tree_node.right.label_counts
        # Get label and count of child with most label counts (majority class)
        if left_counts > right_counts:
            label = tree_node.left.label
            label_count = tree_node.left.label_counts
        else:
            label = tree_node.right.label
            label_count = tree_node.right.label_counts
        # Copy the examining node in a temp variable
        tmp_orig_node = tree_node.clone()
        # Make the node a leaf
        tree_node.set_leaf(label, label_count)
        # Make predictions with the pruned DTree
        y_predict_pruned = current_decision_tree.predict(x)
        # Evaluate the accuracy of pruned DTree
        pruned_val = evaluate(y, y_predict_pruned)
        # If the accuracy before (not pruned) was better, copy back its old attributes from temp variable
        if orig_val > pruned_val:
            tree_node.change_attribute(tmp_orig_node)


# Evaluation Metric 1: Confusion Matrix
def confusion_matrix(y_truth, y_prediction):
    """Compute the confusion matrix.
        
    Args:
        y_truth (np.ndarray): the actual data labels. shape (N,), where N is the number of datasets
        y_prediction (np.ndarray): the predicted labels. shape (N,), where N is the number of datasets

    Returns:
        np.array: confusion has shape (C, C), where C is the number of classes. Rows are ground truth per
         class, columns are predictions
    """

    # An array of all the possible class labels
    class_labels = np.unique(np.concatenate((y_truth, y_prediction)))

    # Initialise the confusion matrix
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # For each class (row), 
    for (i, label) in enumerate(class_labels):

        # Tabulate the model's predictions for that class (columns)
        indices = (y_truth == label)
        predictions = y_prediction[indices]

        # Counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # Convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # Populate confusion matrix for the current label (row)
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


# Evaluation Metric 2(i): Accuracy using y_test and y_predict
def evaluate(y_test, y_predict):
    """Compute the accuracy given the ground truth and predictions

    Args:
        y_test (numpy.ndarray): test dataset label. shape (N,), where N is the number of test datasets
        y_predict (numpy.ndarray): predicted label. shape (N,), where N is the number of test datasets

    Returns:
        float: the accuracy
    """
    # The two np.ndarrays must be the same size to make a comparison
    assert len(y_test) == len(y_predict)

    try:
        # Accuracy calculation
        return np.sum(y_test == y_predict) / len(y_test)
    except ZeroDivisionError:
        return 0.


# Evaluation Metric 2(ii): Accuracy using confusion matrix
def accuracy_cm(confusion_matrix):
    """Compute the accuracy given the ground truth and predictions

    Args:
        confusion_matrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions

    Returns:
        float: the accuracy
    """
    if np.sum(confusion_matrix) > 0:
        # Sum of diagonal array (correctly predictions) divided by the sum of the whole matrix
        return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    else:
        return 0.


# Evaluation Metric 3: Precision
def precision(confusion_matrix):
    """Compute the precision score per class, as well as the macro-averaged precision
    given a confusion matrix. 
        
    Args:
        confusion_matrix (np.ndarray): an array of shape (C, C), where C is the number of classes.
        Rows are ground truth per class, columns are predictions

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where:
            - precisions: np.ndarray of shape (C,), where each element is the precision for class c
            - macro-precision: a float of the macro-averaged precision 
    """
    # Calculate precision score per Class
    precision = np.zeros((len(confusion_matrix),))
    for c in range(confusion_matrix.shape[0]):
        if np.sum(confusion_matrix[:, c]) > 0:
            precision[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[:, c])

    # Calculate macro-averaged precision
    macro_precision = 0.
    if len(precision) > 0:
        macro_precision = np.mean(precision)

    return precision, macro_precision


# Evaluation Metric 4: Recall
def recall(confusion_matrix):
    """Computes the recall score per class, as well as the macro-averaged recall
    given a confusion matrix. 
        
    Args:
        confusion_matrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where:
        - recalls: a np.ndarray of shape (C,), where each element is the recall for class c
        - macro-recall: a float of the macro-averaged recall 
    """

    # Recall score per class
    recall = np.zeros((len(confusion_matrix),))
    for c in range(confusion_matrix.shape[0]):
        if np.sum(confusion_matrix[c, :]) > 0:
            recall[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[c, :])

    # Macro-averaged recall
    macro_recall = 0.
    if len(recall) > 0:
        macro_recall = np.mean(recall)

    return recall, macro_recall


# Evaluation Metric 5: F1-Score
def f1_score(confusion_matrix):
    """ Compute the F1-score per class, as well as the macro-averaged F1-score 
    given a confusion matrix.
        
    Args:
        confusion_matrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where:
        - f1s: a np.ndarray of shape (C,), where each element is the f1-score for class c
        - macro-f1: a float of the macro-averaged f1-score 
    """

    # Get the precision and recall 
    (precisions, macro_p) = precision(confusion_matrix)
    (recalls, macro_r) = recall(confusion_matrix)

    # Ensure precision and recall are of same length
    assert len(precisions) == len(recalls)

    # F1-Score per class
    f = np.zeros((len(precisions),))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)

    return f, macro_f
