"""
The main module of the project. Here, the other 3 modules are imported
in order to read the data set, create a decision which will train on this
data set and finally, evaluate the performance of this Machine Learning model.
"""
import os
import matplotlib.pyplot as plt

from evaluation import *
from dataset import *
from decision_tree import DecisionTree

# CONSTANTS DECLARATION
CLEAN_DATASET_FILE_PATH = "wifi_db/clean_dataset.txt"
NOISE_DATASET_FILE_PATH = "wifi_db/noisy_dataset.txt"
OUTER_FOLD = 10
INNER_FOLD = 10

# Main entry point to our Decision Tree prediction program.
if __name__ == "__main__":

    x_clean, y_clean, classes_clean = None, None, None
    if os.path.isfile(CLEAN_DATASET_FILE_PATH):  # Check if dataset exists
        # Read the clean dataset and split it into training and testing subsets
        x_clean, y_clean, classes_clean = read_dataset(CLEAN_DATASET_FILE_PATH)
    else:
        print("Clean dataset does not exist.")
        print("Exiting...")
        exit(0)

    x_noise, y_noise, classes_noise = None, None, None
    if os.path.isfile(NOISE_DATASET_FILE_PATH):
        # Read the noisy dataset and split it into training and testing subsets
        x_noise, y_noise, classes_noise = read_dataset(NOISE_DATASET_FILE_PATH)
    else:
        print("Noisy dataset does not exist.")
        print("Exiting...")
        exit(0)

    # # PART 2 - Plot (BONUS)
    # print("Step 2 (Bonus) - Plotting Decision Tree")
    # print("Building Clean Decision Tree...")
    # # Create Decision Tree and train it
    # clean_decision_tree = DecisionTree()
    # clean_decision_tree.train(x_noise, y_noise)
    # # Plot the Decision Tree 
    # clean_decision_tree.plot_tree()
    # plt.xticks(np.arange(-100.0, 100.0, 1.0))
    # fig = plt.gcf()
    # fig.set_size_inches(300, 50)
    # # Save it as an image with file name: dt_bonus.png (found in folder)
    # fig.savefig('dt_bonus.png', dpi=100)
    # # plt.show() # uncomment this to show as a plot on Python

    # # PART 3 - Evaluation
    # print("Step 3 - Evaluation")
    # # 3(i): Clean Dataset
    # print("Evaluating clean dataset...")
    # # Run the cross validation and obtain the list of confusion matrices and tree depth from each fold
    # clean_confusion_matrix, clean_depth = cross_validation(x_clean, y_clean, OUTER_FOLD)
    # print("Confusion Matrix of all folds on clean dataset:")
    # print(clean_confusion_matrix)

    # # The average of all the confusion matrices
    # average_clean_cm = np.average(clean_confusion_matrix, axis=0)
    # print("Average confusion matrix: ")
    # print(average_clean_cm)

    # # Average accuracy 
    # accuracy_clean = accuracy_cm(average_clean_cm)
    # print("Accuracy on clean dataset: ", accuracy_clean)

    # # Recall 
    # recall_clean = recall(average_clean_cm)
    # print("Recall on clean dataset: ", recall_clean)

    # # Precision
    # precision_clean = precision(average_clean_cm)
    # print("Precision on clean dataset: ", precision_clean)

    # # F1-Score
    # f1_clean = f1_score(average_clean_cm)
    # print("F1-Score on clean dataset: ", f1_clean)

    # # Tree-Depth
    # ave_depth_clean = np.average(clean_depth)
    # print("Average tree depth on clean dataset: ", ave_depth_clean)

    # print()

    # # 3(ii): Noisy Dataset
    # print("Evaluating noisy dataset...")
    # # Run the cross validation and obtain the list of confusion matrices from each fold
    # noisy_confusion_matrix, noisy_depth = cross_validation(x_noise, y_noise, OUTER_FOLD)
    # print("Confusion Matrix of all folds on noisy dataset:")
    # print(noisy_confusion_matrix)

    # # The average of all the confusion matrices
    # ave_noisy_cm = np.average(noisy_confusion_matrix, axis=0)
    # print("Average confusion matrix:")
    # print(ave_noisy_cm)

    # # Average accuracy 
    # accuracy_noise = accuracy_cm(ave_noisy_cm)
    # print("Accuracy on noisy dataset: ", accuracy_noise)

    # # Recall 
    # recall_noise = recall(ave_noisy_cm)
    # print("Recall on noisy dataset: ", recall_noise)

    # # Precision
    # precision_noise = precision(ave_noisy_cm)
    # print("Precision on noisy dataset: ", precision_noise)

    # # F1-Score
    # f1_noise = f1_score(ave_noisy_cm)
    # print("F1-Score on noisy dataset: ", f1_noise)

    # # Tree-Depth
    # average_depth_noisy = np.average(noisy_depth)
    # print("Average tree depth on noisy dataset: ", average_depth_noisy)
    
    # print()

    # # PART 4 - Pruning (and Evaluation)
    # print("Step 4: Pruning (and Evaluation)")

    # #4(i): Clean Dataset
    # print("Pruning clean dataset...")
    # # Run the pruning and nested cross validation.
    # # This also prints the confusion matrix and depth from each fold.
    # clean_pruned_cm, clean_pruned_depth = pruning_nested_cross_validation(x_clean, y_clean, OUTER_FOLD, INNER_FOLD)

    # # The average of all the confusion matrices
    # ave_clean_pruned_cm = np.average(clean_pruned_cm, axis=0)
    # print("Average confusion matrix")
    # print(ave_clean_pruned_cm)

    # # Average accuracy 
    # accuracy_clean_pruned = accuracy_cm(ave_clean_pruned_cm)
    # print("Accuracy on clean dataset:", accuracy_clean_pruned)

    # # Recall 
    # recall_clean_pruned = recall(ave_clean_pruned_cm)
    # print("Recall on clean dataset:", recall_clean_pruned)

    # # Precision
    # precision_clean_pruned = precision(ave_clean_pruned_cm)
    # print("Precision on clean dataset:", precision_clean_pruned)

    # # F1-Score
    # f1_clean_pruned = f1_score(ave_clean_pruned_cm)
    # print("F1-Score on clean dataset:", f1_clean_pruned)

    # # Tree-Depth
    # ave_depth_clean_pruned = np.average(clean_pruned_depth)
    # print("Average tree depth on clean dataset:", ave_depth_clean_pruned)
    # print()

    # 4(ii): Noisy Dataset
    print("Pruning noisy dataset...")

    # Run the pruning and nested cross validation.
    # This also prints the confusion matrix and depth from each fold.
    noisy_pruned_cm, noisy_pruned_depth = pruning_nested_cross_validation(x_noise, y_noise, OUTER_FOLD, INNER_FOLD)

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
    average_depth_noisy_pruned = np.average(noisy_pruned_depth)
    print("Average tree depth on noisy dataset:", average_depth_noisy_pruned)
    print()
