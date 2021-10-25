import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.lib.function_base import average
from tree_node import tree_node

class decision_tree:
    def __init__(self):
        """init of decision tree 
        """
        self.root_node = None
        self.depth = None

    def train(self,x_train,y_train):
        """[summary]

        Args:
            x_train ([type]): [description]
            y_train ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.root_node, self.depth = decision_tree.decision_tree_learning(x_train,y_train,0)
        return self.root_node, self.depth

    def predict(self,x_test):
        """[summary]

        Args:
            x_test ([type]): [description]
        """
        y_predict = np.zeros((len(x_test),), dtype=np.int) # initialise y_test that will store 
        
        for i,instance in enumerate(x_test):

            # start at the root node
            curr_node = self.root_node

            # recursively enter the tree if it is not the leaf node
            while not curr_node.leaf:
                curr_attribute = curr_node.attribute
                curr_value = curr_node.value

                if instance[curr_attribute] > curr_value:
                    curr_node = curr_node.left
                else: 
                    curr_node = curr_node.right
            
            # if leaf node reached, output the label of the leaf node
            y_predict[i] = curr_node.label

        return y_predict

    def evaluate_acc (y_predict, y_test):
        
        return sum(y_predict==y_test)/len(y_test)

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
            total = y_train[value == y_train].shape[0]
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
        for current_feature in range(x_train.shape[1]):
            current_best_gain  = None
            current_best_value = None
            best_x_train_left = None
            best_y_train_left = None
            best_x_train_right = None
            best_y_train_right = None
            values = np.array(x_train[:, current_feature], copy=True)
            values_sorted_idx = np.argsort(values)
            for current_elem_idx in range(x_train.shape[0] - 1):
                elem_1_idx = values_sorted_idx[current_elem_idx]
                elem_2_idx = values_sorted_idx[current_elem_idx+1]

                split_value = float(values[elem_2_idx]+values[elem_1_idx]) / 2
                left_split_idx = np.argwhere(values > split_value).flatten()
                right_split_idx = np.argwhere(values <= split_value).flatten()
                assert(left_split_idx.shape[0] + right_split_idx.shape[0] == values.shape[0])
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

    def decision_tree_learning(x_train, y_train, depth):
        # if all samples from same labels then stop
        # if y_train.shape[0] < 2:
        #     return tree_node(None,None,True,y_train[0]),depth
        first_label= y_train[0]
        if y_train[y_train == first_label].shape[0] == y_train.shape[0]:
            return tree_node(None,None,True,first_label),depth
        # else 
        else:
            # find split of the node
            feature, split_value,\
                x_train_left,y_train_left,\
                x_train_right,y_train_right = decision_tree.find_split(x_train,y_train)
    
            # create node with the split 
            node  = tree_node(feature,split_value,False, None)
            # find the left branch recursively
            node.left, left_depth = decision_tree.decision_tree_learning(x_train_left,y_train_left,depth +1)
            # find the right branch recursively
            node.right, right_depth = decision_tree.decision_tree_learning(x_train_right,y_train_right,depth +1)
            # return node and max depth of the two branches 
            return node, max(left_depth,right_depth)

    def plottree(self, node=None, x=0, y=0, width=100.0):
        depth_dist = 20
        tree_coords = []

        # if starting the plot, node = root node
        if node is None:
            node = self.root_node

        # if it's not a leaf node
        if not node.leaf: 
            # the node split condition to be printed on the tree
            label_node = str(node.attribute)+'>'+str(node.value)
            print('plotted',x,y)            
            plt.text(x,y,label_node,fontsize=8,horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white'))

            # x, y: parent coordinates; xl, yl: left node coordinates; xr, yr: right node coordinates
            plt.text(x,y,label_node,horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white'),fontsize="xx-large",fontweight=500)
            
            xl = x  - (width/2)
            yl = y - depth_dist
            xr = x + 1 +(width/2)
            yr = y - depth_dist


            # plot left side recursively
            plt.plot([x, xl], [y, yl])
            self.plottree(node.left, xl, yl,width/2)

            # plot right child node recursively
            plt.plot([x, xr], [y, yr])
            self.plottree (node.right, xr, yr,width/2)
        
        # if leaf node reached, end of recursion.
        if node.leaf:
            label_node = 'leaf:'+str(node.label)
            plt.text(x,y, label_node, fontsize=8, horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white'))
            print('plotted leaf node',x,y)
        



        

        
        


