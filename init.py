import os
import numpy as np
from dataset import dataset
from decision_tree import decision_tree

# CONSTANTS DECLARATION
clean_dataset_file_path = "WIFI_db/clean_dataset.txt"
noise_dataset_file_path = "WIFI_db/noisy_dataset.txt"

def read_dt(file_path, test_proportion):
    """This function reads the dataset from a given file path, splits it into training and test samples and returns the result along with their classes.

    Args:
        file_path ([String]): Location of the file on local storage
        test_proportion ([float]): Percentage of testing samples in the dataset

    Returns:
        np.array, np.array, np.array, np.array, np.array: Results of the dataset spliting into X and Y train and  X and Y test samples
    """
    # Read noisy and datasets
    x, y, classes = dataset.read_dataset(file_path)
    # Split datasets into training, testing 
    x_train, x_test, y_train, y_test = dataset.split_dataset(x,y,test_proportion) 

    return x_train, x_test, y_train, y_test, classes


if __name__ == "__main__":
    testing_proportion = 0.25

    #  reading clean dataset and split into training and testing
    x_train_clean, x_test_clean, y_train_clean, y_test_clean, classes_clean = None, None, None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_train_clean, x_test_clean, y_train_clean, y_test_clean, classes_clean = read_dt(clean_dataset_file_path, testing_proportion)
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)
        
    #  reading clean dataset and split into training and testing
    x_train_noise, x_test_noise, y_train_noise, y_test_noise, classes_noise = None, None, None, None, None
    if os.path.isfile(noise_dataset_file_path):
        x_train_noise, x_test_noise, y_train_noise, y_test_noise, classes_noise = read_dt(noise_dataset_file_path, testing_proportion)
    else:
        print("File 2 does not exist.")
        print("Exiting...")
        exit(0)




    print(x_train_clean)
    clean_data_tree , clean_data_depth = decision_tree.decision_tree_learning(x_train_clean,y_train_clean,0)
    # noise_data_tree , noise_data_depth = decision_tree.decision_tree_learning(x_train_noise,y_train_noise,0)
