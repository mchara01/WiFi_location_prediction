import os
import numpy as np
from dataset import dataset
from decision_tree import decision_tree
import matplotlib.pyplot as plt
from evaluation import *

# CONSTANTS DECLARATION
clean_dataset_file_path = "WIFI_db/clean_dataset.txt"
noise_dataset_file_path = "WIFI_db/noisy_dataset.txt"

def read_full_dt(file_path):
    """This function reads the dataset from a given file path, splits it into training and test samples and returns the result along with their classes.

    Args:
        file_path ([String]): Location of the file on local storage
        test_proportion ([float]): Percentage of testing samples in the dataset

    Returns:
        np.array, np.array: Results of the dataset spliting into X and Y and classes
    """
    x, y, classes = dataset.read_dataset(file_path)
    

    return x, y, classes


# def cross_Validation(x,y):
#     acc_list = []
#     indices_list = nested_k_fold_cross_validation(10, len(x))
#     for k in range(len(indices_list)):
#         print("K:" + str(k))
#         inner_list = []
#         for j in range(len(indices_list[k])):
#             print("J:" + str(j))
#             train_indices = indices_list[k][j][0]
#             test_indices = indices_list[k][j][1]
#             val_indices = indices_list[k][j][2]
#             x_train = x[train_indices]
#             y_train = y[train_indices]

#             x_test = x[test_indices]
#             y_test = y[test_indices]
#             test_clean = np.concatenate((x_test, y_test.reshape(len(y_test),1)), axis=1)

#             clean_decision_tree = decision_tree()
#             clean_data_tree , clean_data_depth = clean_decision_tree.train(x_train,y_train)
#             #y_pred = clean_decision_tree.predict(x_test)
#             acc = evaluate(test_clean, clean_decision_tree)
#             print("Acc: " + str(acc))
#             inner_list.append(acc)
#         acc_list.append(inner_list)
#     return acc_list

# Used for Part 3 - Evaluation with simple cross validation
def cross_Validation(x,y,k_folds):
    indices_list = k_fold_indices(k_folds,len(x))

    # Initialise the list that will store the confusion matrix from each fold
    results = [] 
    for k in indices_list:
        test_idx = k[0]
        x_test = x[test_idx]
        y_test = y[test_idx]

        train_idx = k[1]
        x_train = x[train_idx]
        y_train = y[train_idx]

        k_decision_tree = decision_tree()
        data_tree , data_depth = k_decision_tree.train(x_train,y_train)

        y_predicted = k_decision_tree.predict(x_test)
        final_cm = confusion_matrix(y_test,y_predicted)

        results.append(final_cm)
    
    return results

# Used for Part 4 - Pruning (and Evaluation) with nested cross validation
def nested_cross_Validation(x,y,outer_fold, inner_fold):
    indices_list = nested_k_fold_indices(outer_fold,inner_fold, len(x))
    result_dt = []
    for k in indices_list:
        test_indx = k[0]
        x_test = x[test_indx]
        y_test = y[test_indx]
        
        best_dt = None
        best_acc = None
        for j in k[1]:
            train_indices = j[0]
            val_indices = j[1]
            x_train = x[train_indices]
            y_train = y[train_indices]
            x_val = x[val_indices]
            y_val = y[val_indices]

            current_decision_tree = decision_tree()
            data_tree , data_depth = current_decision_tree.train(x_train,y_train)

            acc = evaluate(x_val,y_val, current_decision_tree)

            if best_acc is None or acc > best_acc:
                print("changed best acc to {}".format(acc) )
                best_acc = acc
                best_dt = current_decision_tree

        y_predicted = best_dt.predict(x_test)

        final_cm = confusion_matrix(y_test,y_predicted)
        print(final_cm)
        result_dt.append([best_dt,final_cm])

    return result_dt

if __name__ == "__main__":
    outer_fold = 10
    inner_fold = 10


    #  reading clean dataset and split into training and testing
    x_clean,y_clean,classes_clean = None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_clean, y_clean, classes_clean = read_full_dt(clean_dataset_file_path)
    
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)


    #  reading clean dataset and split into training and testing
    x_noise,y_noise,classes_noise = None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_noise, y_noise, classes_noise = read_full_dt(noise_dataset_file_path)
    
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)


    print("clean dataset")
    clean_dt_evaluation = cross_Validation(x_clean,y_clean,outer_fold)
    print(clean_dt_evaluation)
    # print("noisy dataset")
    # noise_dt_evaluation = new_cross_Validation(x_noise,y_noise,outer_fold,inner_fold)


    # print("Sum confusion matrix")
    # print("clean dataset")
    # print("noisy dataset")

    

