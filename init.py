"""
FILL MODULE DOCSTRING!
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from evaluation import *
from dataset import dataset
from decision_tree import decision_tree

# CONSTANTS DECLARATION
clean_dataset_file_path = "wifi_db/clean_dataset.txt"
noise_dataset_file_path = "wifi_db/noisy_dataset.txt"

def read_full_dt(file_path):
    """Read a dataset.

    Read a dataset from a given file path, split it into training and test samples and return results classes.

    Args:
        file_path (str): Location of the file on local storage.

    Returns:
        np.array, np.array: Results of the dataset spliting into X and Y and classes
    """
    x, y, classes = dataset.read_dataset(file_path)
    
    return x, y, classes

if __name__ == "__main__":
    outer_fold = 10
    inner_fold = 10


    #  Read clean dataset and split into training and testing
    x_clean,y_clean,classes_clean = None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_clean, y_clean, classes_clean = read_full_dt(clean_dataset_file_path)
    
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)


    #  Read clean dataset and split into training and testing
    x_noise,y_noise,classes_noise = None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_noise, y_noise, classes_noise = read_full_dt(noise_dataset_file_path)
    
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)


    # PART 2 - Plot 
    print ("Building Clean Decision Tree")
    clean_decision_tree = decision_tree()
    clean_decision_tree.train(x_clean,y_clean)
    print ("Step 2 (Bonus) - Plotting Decision Tree")
    clean_decision_tree.plottree()
    plt.xticks(np.arange(-100.0,100.0, 1.0))
    fig = plt.gcf()
    fig.set_size_inches(300, 50)
    fig.savefig('test2png.png', dpi=100)
    plt.show()



    # PART 3 - Evaluate
    print("Step 3 - Evaluation")
    # 3(i): Clean Dataset
    print("Evaluating clean dataset...")
    # Run the cross validation and obtain the list of confusion matrix and tree depth from each fold
    clean_confusionmatrix, clean_depth = cross_Validation(x_clean,y_clean,outer_fold) 
    print("Confusion Matrix of all folds")
    print(clean_confusionmatrix)

    # The average of all the confusion matrices
    ave_clean_cm = np.average(clean_confusionmatrix, axis=0) 
    print("Average confusion matrix")
    print(ave_clean_cm)

    # Average accuracy 
    accuracy_clean = accuracy_cm(ave_clean_cm)
    print("Accuracy on clean dataset:", accuracy_clean)
    
    # Recall 
    recall_clean = recall(ave_clean_cm)
    print("Recall on clean dataset:", recall_clean)

    # Precision
    precision_clean = precision(ave_clean_cm)
    print("Precision on clean dataset:", precision_clean)

    # F1-Score
    f1_clean = f1_score(ave_clean_cm)
    print("F1-Score on clean dataset:", f1_clean)

    # Tree-Depth
    ave_depth_clean = np.average(clean_depth)
    print("Average tree depth on clean dataset:", ave_depth_clean)
    print()


    # 3(ii): Noisy Dataset
    print("Evaluating noisy dataset...")
    # Run the cross validation and obtain the list of confusion matrix from each fold
    noisy_confusionmatrix, noisy_depth = cross_Validation(x_noise,y_noise,outer_fold) 
    print("Confusion Matrix of all folds")
    print(noisy_confusionmatrix)

    # The average of all the confusion matrices
    ave_noisy_cm = np.average(noisy_confusionmatrix, axis=0) 
    print("Average confusion matrix")
    print(ave_noisy_cm)

    # Average accuracy 
    accuracy_noise = accuracy_cm(ave_noisy_cm)
    print("Accuracy on noisy dataset:", accuracy_noise)
    
    # Recall 
    recall_noise = recall(ave_noisy_cm)
    print("Recall on noisy dataset:", recall_noise)

    # Precision
    precision_noise = precision(ave_noisy_cm)
    print("Precision on noisy dataset:", precision_noise)

    # F1-Score
    f1_noise = f1_score(ave_noisy_cm)
    print("F1-Score on noisy dataset:", f1_noise)

    # Tree-Depth
    ave_depth_noisy = np.average(noisy_depth)
    print("Average tree depth on noisy dataset:", ave_depth_noisy)
    print()



    # PART 4 - Pruning (and Evaluation)
    print("Step 4: Pruning (and Evaluation)")
    # 4(i): Clean Dataset
    print("Pruning clean dataset...")

    # Run the pruning and nested cross validation, and obtain the list of tree depth and confusion matrix from each fold
    clean_pruned_cm, clean_pruned_depth = pruning_nested_cross_Validation(x_clean,y_clean,outer_fold,inner_fold)

    print("Confusion Matrix of all folds")
    print(clean_pruned_cm)

    # The average of all the confusion matrices
    ave_clean_pruned_cm = np.average(clean_pruned_cm, axis=0) 
    print("Average confusion matrix")
    print(ave_clean_pruned_cm)

    # Average accuracy 
    accuracy_clean_pruned = accuracy_cm(ave_clean_pruned_cm)
    print("Accuracy on clean dataset:", accuracy_clean_pruned)
    
    # Recall 
    recall_clean_pruned = recall(ave_clean_pruned_cm)
    print("Recall on clean dataset:", recall_clean_pruned)

    # Precision
    precision_clean_pruned = precision(ave_clean_pruned_cm)
    print("Precision on clean dataset:", precision_clean_pruned)

    # F1-Score
    f1_clean_pruned = f1_score(ave_clean_pruned_cm)
    print("F1-Score on clean dataset:", f1_clean_pruned)

    # Tree-Depth
    ave_depth_clean_pruned = np.average(clean_pruned_depth)
    print("Average tree depth on noisy dataset:", ave_depth_clean_pruned)
    print()


    # 4(ii): Noisy Dataset
    print("Pruning noisy dataset...")

    # Run the pruning and nested cross validation, and obtain the list of confusion matrix from each fold
    noisy_pruned_cm, noisy_pruned_depth = pruning_nested_cross_Validation(x_noise,y_noise,outer_fold,inner_fold)

    print("Confusion Matrix of all folds")  
    print(noisy_pruned_cm)

    # The average of all the confusion matrices
    ave_noisy_pruned_cm = np.average(noisy_pruned_cm, axis=0) 
    print("Average confusion matrix")
    print(ave_noisy_pruned_cm)

    # Average accuracy 
    accuracy_noisy_pruned = accuracy_cm(ave_noisy_pruned_cm)
    print("Accuracy on noisy dataset:", accuracy_noisy_pruned)
    
    # Recall 
    recall_noisy_pruned = recall(ave_noisy_pruned_cm)
    print("Recall on noisy dataset:", recall_noisy_pruned)

    # Precision
    precision_noisy_pruned = precision(ave_noisy_pruned_cm)
    print("Precision on noisy dataset:", precision_noisy_pruned)

    # F1-Score
    f1_noisy_pruned = f1_score(ave_noisy_pruned_cm)
    print("F1-Score on noisy dataset:", f1_noisy_pruned)

    # Tree-Depth
    ave_depth_noisy_pruned = np.average(noisy_pruned_depth)
    print("Average tree depth on noisy dataset:", ave_depth_noisy_pruned)
    print()

