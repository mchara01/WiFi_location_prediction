"""
The Decision Tree module contains any method required to
develop this Machine Learning model in a eager, supervised
learning fashion.
"""
import numpy as np
import matplotlib.pyplot as plt

from tree_node import TreeNode


class DecisionTree:

    def __init__(self):
        """Initialisation of DecisionTree class object parameters."""
        self.root_node = None
        self.depth = None

    def train(self, x_train, y_train):
        """ Training of the Decision Tree.

        Training of the Decision Tree using the given training features (x_train) and
        labels (y_train) of the dataset.

        Args:
            x_train (np.array): Features of the dataset
            y_train (np.array): Labels of the dataset
        """
        # Train recursively the DecisionTree object
        self.root_node, self.depth = DecisionTree.decision_tree_learning(self, x_train, y_train, 0)

    def predict(self, x_test):
        """Function takes a set of attributes and predicts their corresponding labels.

        Args:
            x_test (np.array): An array of attributes for each dataset
        
        Returns:
            np.array: An array of labels predicted by the model for the inputted attributes
        """
        # Initialise y_predict with zeros
        y_predict = np.zeros((len(x_test),), dtype=np.int)

        for i, instance in enumerate(x_test):
            # Start from the root node of the decision tree
            curr_node = self.root_node

            # Recursively traverse the tree while it is not a leaf node
            while not curr_node.leaf:
                # Extract attribute and value from current node
                curr_attribute = curr_node.attribute
                curr_value = curr_node.value
                # Find which is next node (from the left or right child)
                if instance[curr_attribute] > curr_value:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right

            # Once a leaf node is reached, output the label of the leaf node
            y_predict[i] = curr_node.label

        return y_predict

    def find_entropy(self, y_train):
        """Calculates the entropy of a given label set.

        Args:
            y_train (np.array): List of integer label values in a given dataset

        Returns:
            float: Entropy of a label set.
        """
        # Find all unique classes in dataset (i.e. room numbers)
        values_list = np.unique(y_train)
        entropy = 0
        # Iterate though each room value (0-3)
        for value in values_list:
            # Number of samples with value as their label
            total = y_train[value == y_train].shape[0]
            # Calculate probability of the value from whole dataset
            probability = total / y_train.shape[0]
            # Calculate log2 of probability
            log_value = np.log2(probability)
            # Multiply the probability with the log value and subtract it from the total entropy
            entropy -= (probability * log_value)
        return entropy

    def find_information_gain(self, entropy, y_subset_left, y_subset_right):
        """Calculates the Information Gain.

         Information Gain calculation using the entropy of the whole dataset and
         two subsets.

        Args:
            entropy (float): Entropy of whole dataset
            y_subset_left (np.array): Left label subset
            y_subset_right (np.array): Right label subset

        Returns:
            float: Information gain
        """
        # Calculate left subsets and right subsets entropy
        left_entropy = DecisionTree.find_entropy(self, y_subset_left)
        right_entropy = DecisionTree.find_entropy(self, y_subset_right)
        # Find total number of samples in two subsets
        total_y = y_subset_left.shape[0] + y_subset_right.shape[0]
        # Calculate Remainder
        left = (float(y_subset_left.shape[0]) / total_y) * left_entropy
        right = (float(y_subset_right.shape[0]) / total_y) * right_entropy
        # Calculation of the information gain
        info_gain = entropy - (left + right)
        return info_gain

    def find_split(self, x_train, y_train):
        """Finds the optimal split for a given dataset.

        Iterates through all the features and for each find the optimal splitting value
        which will yield the optimal information gain. Then, find amongst all the features
        which one provides the highest information gain and what is the splitting value for that.

        Args:
            x_train (np.array): Features of training set
            y_train (np.array): Labels of training set

        Returns:
            float, float, float, float, float, float: Returns the best feature along with its best split value
             and the left and right subsets, produced after the division based on that feature.
        """
        # Find entropy of dataset from label set
        entropy = DecisionTree.find_entropy(self, y_train)
        # Variable initialisation
        current_best_feature_gain = None
        current_best_feature = None
        current_best_feature_split = None
        best_feature_x_train_left = None
        best_feature_y_train_left = None
        best_feature_x_train_right = None
        best_feature_y_train_right = None
        # Iterate through all the features
        for current_feature in range(x_train.shape[1]):
            current_best_gain = None
            current_best_value = None
            best_x_train_left = None
            best_y_train_left = None
            best_x_train_right = None
            best_y_train_right = None
            # Copy/Extract the feature (column) values
            values = np.array(x_train[:, current_feature], copy=True)
            # Indices that sort the values list
            values_sorted_idx = np.argsort(values)
            # Iterate through all the samples for a specific feature
            for current_elem_idx in range(x_train.shape[0] - 1):
                # Take a set of elements each time from the sorted index list
                elem_1_idx = values_sorted_idx[current_elem_idx]
                elem_2_idx = values_sorted_idx[current_elem_idx + 1]
                # Average of the two is the split value
                split_value = float(values[elem_2_idx] + values[elem_1_idx]) / 2
                # Split values to left and right based on the calculated split value
                left_split_idx = np.argwhere(values > split_value).flatten()
                right_split_idx = np.argwhere(values <= split_value).flatten()
                assert (left_split_idx.shape[0] + right_split_idx.shape[0] == values.shape[0])
                x_train_left = x_train[left_split_idx]
                y_train_left = y_train[left_split_idx]
                x_train_right = x_train[right_split_idx]
                y_train_right = y_train[right_split_idx]
                # Calculate Information gained from the splitting value for this feature
                info_gain = DecisionTree.find_information_gain(self, entropy, y_train_left, y_train_right)
                # Check if its the fist IG calculation OR the current IG is higher than the previous best
                if current_best_gain is None or info_gain > current_best_gain:
                    current_best_gain = info_gain
                    current_best_value = split_value
                    best_x_train_left = x_train_left
                    best_y_train_left = y_train_left
                    best_x_train_right = x_train_right
                    best_y_train_right = y_train_right
            # Same check as the previous, but on feature level
            if current_best_feature_gain is None or current_best_gain > current_best_feature_gain:
                current_best_feature_gain = current_best_gain
                current_best_feature = current_feature
                current_best_feature_split = current_best_value
                best_feature_x_train_left = best_x_train_left
                best_feature_y_train_left = best_y_train_left
                best_feature_x_train_right = best_x_train_right
                best_feature_y_train_right = best_y_train_right

        return current_best_feature, current_best_feature_split, \
            best_feature_x_train_left, \
            best_feature_y_train_left, \
            best_feature_x_train_right, \
            best_feature_y_train_right

    def decision_tree_learning(self, x_train, y_train, depth):
        """Recursive creation of Decision Trees.

        Utilisation of the training dataset for creating the DecisionTree in
        a recursive manner.

        Args:
            x_train ([type]): Features in training dataset
            y_train (np.array): Labels in training dataset
            depth (int): Depth of Decision tree

        Returns:
            TreeNode, int: Returns a Node and  the maximum depth between the two branches
        """
        first_label = y_train[0]
        if y_train[y_train == first_label].shape[0] == y_train.shape[0]:
            return TreeNode(None, None, None, None, True, first_label, y_train.shape[0]), depth
        else:
            # Find split of the node
            feature, split_value, x_train_left, y_train_left, x_train_right, y_train_right = DecisionTree.find_split(
                self, x_train, y_train)
            # Create node with the split value as root
            node = TreeNode(feature, split_value, None, None, False, None, None)
            # Find the left branch recursively while increasing depth
            node.left, left_depth = DecisionTree.decision_tree_learning(self, x_train_left, y_train_left, depth + 1)
            # Find the right branch recursively while increasing depth
            node.right, right_depth = DecisionTree.decision_tree_learning(self, x_train_right, y_train_right, depth + 1)
            # Return node and max depth between the two branches
            return node, max(left_depth, right_depth)

    def final_depth(self):
        '''final depth of the tree
        compute the depth of the tree from the root node
        
        returns:
         depth (int): the depth of the tree from the root node
        '''
        return self.root_node.final_depth()

    def plot_tree(self, node=None, x=0, y=0, width=100.0):
        """Tree visualisation - BONUS

        Visualisation of the Decision Tree using recursion.

        Args:
            node (object):
            x (float): X coardinates of parent node
            y (float): Depth of Decision tree
            width (float): Node width, used for displaying it
        """
        depth_dist = 20

        # If start of plotting, make node = root node
        if node is None:
            node = self.root_node

        # If it's not a leaf node
        if not node.leaf:
            # Node's split condition to be printed on the tree
            label_node = str(node.attribute) + ' > ' + str(node.value)

            # Text formatting
            plt.text(x, y, label_node, horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white'), fontsize="xx-large", fontweight=500)

            # x, y: parent coordinates; xl, yl: left node coordinates; xr, yr: right node coordinates
            xl = x - (width / 2)
            yl = y - depth_dist
            xr = x + 1 + (width / 2)
            yr = y - depth_dist

            # Plot left side recursively
            plt.plot([x, xl], [y, yl])
            self.plot_tree(node.left, xl, yl, width / 2)

            # Plot right side recursively
            plt.plot([x, xr], [y, yr])
            self.plot_tree(node.right, xr, yr, width / 2)

        # If leaf node reached, end recursion
        if node.leaf:
            label_node = 'Leaf: ' + str(node.label)
            plt.text(x, y, label_node, fontsize=8, horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white'))
