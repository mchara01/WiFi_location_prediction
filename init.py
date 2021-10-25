import os
import numpy as np
from dataset import dataset
from decision_tree import decision_tree
import matplotlib.pyplot as plt
# CONSTANTS DECLARATION
clean_dataset_file_path = "WIFI_db/clean_dataset.txt"
noise_dataset_file_path = "WIFI_db/noisy_dataset.txt"

def read_dt(file_path, test_proportion, validation_proportion):
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
    x_test, x_val, y_test, y_val = dataset.split_dataset(x_test,y_test,validation_proportion) 

    return x_train, x_test, x_val, y_train, y_test, y_val, classes


if __name__ == "__main__":
    # calculation as follows: split first by testing proportion, and then by validation proportion
    # for eg: if testing_proportion = 0.3, validation_proportion = 0.5, then train:test:val = 0.7:0.15:0.15
    testing_proportion = 0.3
    validation_proportion = 0.5 


    #  reading clean dataset and split into training and testing
    x_train_clean, x_test_clean, x_val_clean, y_train_clean, y_test_clean, y_val_clean, classes_clean = None, None, None, None, None, None, None
    if os.path.isfile(clean_dataset_file_path):
        x_train_clean, x_test_clean, x_val_clean, y_train_clean, y_test_clean, y_val_clean, classes_clean = read_dt(clean_dataset_file_path, testing_proportion, validation_proportion)
    
    else:
        print("File 1 does not exist.")
        print("Exiting...")
        exit(0)
        
    #  reading clean dataset and split into training and testing
    x_train_noise, x_test_noise, x_val_noise, y_train_noise, y_test_noise, y_val_noise, classes_noise = None, None, None, None, None, None, None
    if os.path.isfile(noise_dataset_file_path):
        x_train_noise, x_test_noise, x_val_noise, y_train_noise, y_test_noise, y_val_noise, classes_noise = read_dt(noise_dataset_file_path, testing_proportion, validation_proportion)
    else:
        print("File 2 does not exist.")
        print("Exiting...")
        exit(0)



    clean_decision_tree = decision_tree()
    noise_decision_tree = decision_tree()
    clean_data_tree , clean_data_depth = clean_decision_tree.train(x_train_clean,y_train_clean)
    noise_data_tree , noise_data_depth = noise_decision_tree.train(x_train_noise,y_train_noise)

    # acc of test set
    print("acc of test set")
    y_test_predicted_clean = clean_decision_tree.predict(x_test_clean)
    y_test_predicted_noise = clean_decision_tree.predict(x_test_noise)

    print(decision_tree.evaluate_acc(y_test_predicted_clean,y_test_clean))
    print(decision_tree.evaluate_acc(y_test_predicted_noise,y_test_noise))

    # acc of validation
    print("acc of val set")
    y_val_predicted_clean = clean_decision_tree.predict(x_test_clean)
    y_val_predicted_noise = clean_decision_tree.predict(x_test_noise)

    print(decision_tree.evaluate_acc(y_val_predicted_clean,y_test_clean))
    print(decision_tree.evaluate_acc(y_val_predicted_noise,y_test_noise))

    clean_decision_tree.plottree()
    plt.xticks(np.arange(-100.0,100.0, 1.0))
    fig = plt.gcf()
    fig.set_size_inches(200, 50)
    fig.savefig('test2png.png', dpi=100)
    plt.show()