import os
import numpy as np
from numpy.random import default_rng

class dataset:

    def read_dataset(filepath):
        """ Read in the dataset from the specified filepath

        Args:
            filepath (str): The filepath to the dataset file

        Returns:
            tuple: returns a tuple of (x, y, classes), each being a numpy array. 
                    - x is a numpy array with shape (N, K), 
                        where N is the number of instances
                        K is the number of features/attributes
                    - y is a numpy array with shape (N, ), and should be integers from 0 to C-1
                        where C is the number of classes 
                    - classes : a numpy array with shape (C, ), which contains the 
                        unique class labels corresponding to the integers in y
        """

        data = np.loadtxt(filepath)

        x = data[:,:-1]
        y = data[:,-1]

        classes, y = np.unique(y, return_inverse=True)
        return (x, y, classes)


    def split_dataset(x, y, test_proportion, random_generator=default_rng()):
        """ Split dataset into training and test sets, according to the given 
            test set proportion.
        
        Args:
            x (np.ndarray): Instances, numpy array with shape (N,K)
            y (np.ndarray): Class labels, numpy array with shape (N,)
            test_proprotion (float): the desired proportion of test examples 
                                    (0.0-1.0)
            random_generator (np.random.Generator): A random generator

        Returns:
            tuple: returns a tuple of (x_train, x_test, y_train, y_test) 
                - x_train (np.ndarray): Training instances shape (N_train, K)
                - x_test (np.ndarray): Test instances shape (N_test, K)
                - y_train (np.ndarray): Training labels, shape (N_train, )
                - y_test (np.ndarray): Test labels, shape (N_train, )
        """

        shuffled_indices = random_generator.permutation(len(x))
        n_test = round(len(x) * test_proportion)
        n_train = len(x) - n_test
        x_train = x[shuffled_indices[:n_train]]
        y_train = y[shuffled_indices[:n_train]]
        x_test = x[shuffled_indices[n_train:]]
        y_test = y[shuffled_indices[n_train:]]
        return (x_train, x_test, y_train, y_test)