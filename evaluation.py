import numpy as np
from numpy.random import default_rng
from decision_tree import decision_tree
from dataset import *


# Step 3 - Evaluation with simple cross validation
def cross_Validation(x,y,k_folds):
    """
    Computes and appends the confusion matrix generated from each fold.
    At each fold, the function trains the decision tree with training data.
    Its performance is then evaluated with the test data, and appended.

    Args:
        y_test (numpy.ndarray): test dataset label
        y_predict (numpy.ndarray): predicted label

    Returns:
        results: list of length k_folds
            - Each element is the confusion matrix corresponding to a fold. 
        depth: list of length k_folds
            - Each element is the confusion matrix corresponding to a fold. 
    """
    indices_list = dataset.k_fold_indices(k_folds,len(x))

    # Initialise the list that will store the:
    # (i) confusion matrix from each fold
    results = [] 
    # (ii) depth of the decision tree from each fold
    depth = []

    # Going through each fold
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
        k_decision_tree = decision_tree()
        data_tree , data_depth = k_decision_tree.train(x_train,y_train)

        # Predict the label and evaluate performance
        y_predicted = k_decision_tree.predict(x_test)
        final_cm = confusion_matrix(y_test,y_predicted)

        results.append(final_cm)
        depth.append(k_decision_tree.depth)
    
    return results, depth

# Step 4 - Pruning with Nested Cross-Validation
def pruning_nested_cross_Validation(x,y,outer_fold, inner_fold):
    indices_list = dataset.nested_k_fold_indices(outer_fold,inner_fold, len(x))
    result_dt = []
    depth = []
    for k in indices_list:
        test_indx = k[0]
        x_test = x[test_indx]
        y_test = y[test_indx]
        
        best_dt = None
        best_acc = None
        for j in k[1]: # k[1]: [ [[train1 indices],[val1 indices]] , [[train2 indices],[val2 indices]] ...]
            train_indices = j[0] 
            val_indices = j[1]
            x_train = x[train_indices]
            y_train = y[train_indices]
            x_val = x[val_indices]
            y_val = y[val_indices]

            current_decision_tree = decision_tree()
            data_tree , data_depth = current_decision_tree.train(x_train,y_train)
            
            pruning_simulation(current_decision_tree,x_val,y_val)
            y_predict = current_decision_tree.predict(x_val)
            acc = evaluate(y_val, y_predict)

            if best_acc is None or acc > best_acc:
                # print("changed best acc to {}".format(acc) )
                best_acc = acc
                best_dt = current_decision_tree

        y_predicted = best_dt.predict(x_test)

        final_cm = confusion_matrix(y_test,y_predicted)
        # print(final_cm)
        result_dt.append(final_cm) #best_dt,final_cm]
        depth.append(best_dt.depth)

    return result_dt, depth

def pruning_simulation(current_decision_tree,x,y):
    queue = list()
    queue.append(current_decision_tree.root_node)

    while queue:
        current_node = queue.pop(0)
        if current_node.left.leaf and current_node.right.leaf:
            y_predict = current_decision_tree.predict(x)
            orig_val = evaluate(y, y_predict)
            left_counts = current_node.left.label_counts
            right_counts = current_node.right.label_counts
            label = None
            label_count = None
            if left_counts > right_counts:
                label = current_node.left.label
                label_count = current_node.left.label_counts
            else:
                label = current_node.right.label
                label_count = current_node.right.label_counts
            tmp_orig_node = current_node.clone()
            orig_val.convert_leaf(label,label_count)
            y_predict_pruned = current_decision_tree.predict(x)
            pruned_val = evaluate(y, y_predict_pruned)


            if orig_val > pruned_val:
                current_node.change_attribute(tmp_orig_node)
        else:
            if current_node.left.leaf:
                queue.append(current_node.left)
            if current_node.right.leaf:
                queue.append(current_node.right)

# Evaluation Metric 1: Confusion Matrix
def confusion_matrix(y_truth, y_prediction):
    """ Compute the confusion matrix.
        
    Args:
        y_truth (np.ndarray): the actual data labels
        y_prediction (np.ndarray): the predicted labels 

    Returns:
        confusion (np.array): shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions
    """

    # An array of all the possible class labels
    class_labels = np.unique(np.concatenate((y_truth, y_prediction)))

    # Initialise the confusion matrix
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # For each class (row), 
    for (i, label) in enumerate(class_labels):

        # Tabulate the model's predictions for that class (columns)
        indices = (y_truth == label)
        truth = y_truth[indices]
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
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_test (numpy.ndarray): test dataset label
        y_predict (numpy.ndarray): predicted label

    Returns:
        float : the accuracy
    """
    assert len(y_test) == len(y_predict)  
    
    try:
        return np.sum(y_test == y_predict) / len(y_test)
    except ZeroDivisionError:
        return 0.


# Evaluation Metric 2(ii): Accuracy using confusion matrix
def accuracy_cm(confusionmatrix):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        confusionmatrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """
    if np.sum(confusionmatrix) > 0:
        return np.sum(np.diag(confusionmatrix)) / np.sum(confusionmatrix)
    else:
        return 0.


# Evaluation Metric 3: Precision
def precision(confusionmatrix):
    """ Compute the precision score per class, as well as the macro-averaged precision
    given a confusion matrix. 
        
    Args:
        confusionmatrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where:
            - precisions: np.ndarray of shape (C,), where each element is the precision for class c
            - macro-precision: a float of the macro-averaged precision 
    """
    # Precision Score per Class
    precision = np.zeros((len(confusionmatrix), ))
    for c in range(confusionmatrix.shape[0]):
        if np.sum(confusionmatrix[:, c]) > 0:
            precision[c] = confusionmatrix[c, c] / np.sum(confusionmatrix[:, c]) 

    # Macro-averaged precision
    macro_precision = 0.
    if len(precision) > 0:
        macro_precision = np.mean(precision)
    
    return (precision, macro_precision)

# Evaluation Metric 4: Recall
def recall(confusionmatrix):
    """ Computes the recall score per class, as well as the macro-averaged recall
    given a confusion matrix. 
        
    Args:
        confusionmatrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where:
        - recalls: a np.ndarray of shape (C,), where each element is the recall for class c
        - macro-recall: a float of the macro-averaged recall 
    """

    # Recall score per class
    recall = np.zeros((len(confusionmatrix), ))
    for c in range(confusionmatrix.shape[0]):
        if np.sum(confusionmatrix[c, :]) > 0:
            recall[c] = confusionmatrix[c, c] / np.sum(confusionmatrix[c, :])

    # Macro-averaged recall
    macro_recall = 0.
    if len(recall) > 0:
        macro_recall = np.mean(recall)
    
    return (recall, macro_recall)

# Evaluation Metric 5: F1-Score
def f1_score(confusionmatrix):
    """ Compute the F1-score per class, as well as the macro-averaged F1-score 
    given a confusion matrix.
        
    Args:
        confusionmatrix (np.ndarray): an array of shape (C, C), where C is the number of classes. 
        Rows are ground truth per class, columns are predictions.

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where:
        - f1s: a np.ndarray of shape (C,), where each element is the f1-score for class c
        - macro-f1: a float of the macro-averaged f1-score 
    """

    # Get the precision and recall 
    (precisions, macro_p) = precision(confusionmatrix)
    (recalls, macro_r) = recall(confusionmatrix)

    # Ensure precision and recall are of same length
    assert len(precisions) == len(recalls)

    # F1-Score per class
    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)