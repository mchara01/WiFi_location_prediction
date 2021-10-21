import numpy as np
from numpy.lib.function_base import average
from tree_node import tree_node

class decision_tree:

    def find_entropy(y_train):
        """Calculates the entropy of a given label set

        Args:
            y_train (list): List of integer label values in a given dataset

        Returns:
            Float: 
        """
        values_list = np.unique(y_train)
        sum_total = 0
        for value in values_list:
            total = np.sum(y_train[value == y_train])
            probability = total / y_train.shape[0]
            log_value = np.log2(probability)
            sum_total -= (probability * log_value)
        return sum_total

    def find_information_gain(entropy, y_subset_left, y_subset_right):
        """[summary]

        Args:
            entropy ([type]): [description]
            y_subset_left ([type]): [description]
            y_subset_right ([type]): [description]

        Returns:
            [type]: [description]
        """
        # calculate left subsets and right subsets entropy 
        left_Entropy = decision_tree.find_entropy(y_subset_left)
        right_entropy = decision_tree.find_entropy(y_subset_right)
        total_y = y_subset_left.shape[0] + y_subset_right.shape[0]
        left =  (float(y_subset_left.shape[0]) / total_y) * left_Entropy 
        right =  (float(y_subset_right.shape[0]) / total_y) * right_entropy 
        return entropy - (left + right)

    def find_split(x_train,y_train):
        entropy = decision_tree.find_entropy(y_train)
        current_best_feature_gain = None
        current_best_feature = None
        current_best_feature_split = None
        best_feature_x_train_left = None
        best_feature_y_train_left = None
        best_feature_x_train_right = None
        best_feature_y_train_right = None
        # print(x_train)
        for current_feature in range(x_train.shape[1]-1):
            current_best_gain  = None
            current_best_value = None
            best_x_train_left = None
            best_y_train_left = None
            best_x_train_right = None
            best_y_train_right = None
            values = np.array(x_train[:, current_feature], copy=True)
            print(values)
            values_sorted_idx = np.argsort(values)
            for current_elem_idx in range(x_train.shape[0] - 1):
                elem_1_idx = values_sorted_idx[current_elem_idx]
                elem_2_idx = values_sorted_idx[current_elem_idx+1]

                split_value = float(values[elem_2_idx]+values[elem_1_idx]) / 2
                left_split_idx = np.argwhere(values > split_value)
                right_split_idx = np.argwhere(values <= split_value)
                x_train_left = x_train[left_split_idx]
                y_train_left = y_train[left_split_idx]
                x_train_right = x_train[right_split_idx]
                y_train_right = y_train[right_split_idx]
                gain = decision_tree.find_information_gain(entropy,y_train_left,y_train_right)
                if current_best_gain is None or gain > current_best_gain:
                    current_best_gain = gain
                    current_best_value = split_value
                    best_x_train_left = x_train_left
                    best_y_train_left = y_train_left
                    best_x_train_right = x_train_right
                    best_y_train_right = y_train_right
            
            if current_best_feature_gain is  None or current_best_gain > current_best_feature_gain:
                current_best_feature_gain = current_best_gain
                current_best_feature = current_feature
                current_best_feature_split = current_best_value
                best_feature_x_train_left = best_x_train_left
                best_feature_y_train_left = best_y_train_left
                best_feature_x_train_right = best_x_train_right
                best_feature_y_train_right = best_y_train_right

        return current_best_feature,\
                current_best_feature_split,\
                best_feature_x_train_left,\
                best_feature_y_train_left,\
                best_feature_x_train_right,\
                best_feature_y_train_right

    def decision_tree_learning( x_train, y_train, depth):
        # if all samples from same labels then stop
        if x_train is None:
            return tree_node(None,None,False,None),depth

        if x_train.shape[0] < 2:
            return tree_node(None,None,True,x_train[0]),depth
        first_label= x_train[0]
        if np.sum(len(x_train[x_train == first_label])) == len(x_train):
            return tree_node(None,None,False,first_label),depth
        # else 
        else:
            # find split of the node
            feature, split_value,\
                x_train_left,y_train_left,\
                x_train_right,y_train_right = decision_tree.find_split(x_train,y_train)
            print(x_train_left.size)
            print(x_train_left)
            # create node with the split 
            node  = tree_node(feature,split_value,False, None)
            # find the left branch recursively
            node.left, left_depth = decision_tree.decision_tree_learning(x_train_left,y_train_left,depth +1)
            # find the right branch recursively
            node.right, right_depth = decision_tree.decision_tree_learning(x_train_right,y_train_right,depth +1)
            # return node and max depth of the two branches 
            return node, max(left_depth,right_depth)
