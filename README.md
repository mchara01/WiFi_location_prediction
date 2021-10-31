# WiFi_location_prediction
building a decision tree model to predict in which of the 4 rooms the user is standing.
the prediction is based on two datasets found in WIFI_db folder *clean_dataset.txt* and *noisy_dataset.txt*
## run the code
the init.py file is the starting point that will print out all of the results based on the coursework sections, the init.py loads both the clean and the noisy datasets from wifi_db file. to change the datasets, add the needed datasetss to the wifi_db file and change the constant varibale in the init.py script. the script will create a clean tree based on the full clean datatset to plot out the tree, then the first section will evaluate the algorithm via cross validation by creating trees based on the training folds and test it via the testing fold, the separtion is made to make sure the model is unbaised to the test data and gives the true evaluation from the destribution of the unseen data. then the secound part of the script will perform nested cross-validation to tune and evaluate a pruned version of the tree.
## the report
the report will show all of the evaluated statistics done on the unpruned and pruned tree via cross validationa and nested validation respectively. furthermore, the report will elaborate more on the full clean dataset tree and closeup of the upper part of it.
## code segmentation:
the code contains:
1. decision_tree class:
    1. training and predicting algorithm along with the needed methods to train such as finding information gain, entropy and geting the best split based on the information gain
    2. ploting the tree. 
    
2. dataset class:
   1. reading the datasets and dividing the data into features, labels and classes (maping between the real labels and converted labels for convience in usage).
   2. get k fold splits by dividing the dataset into k-fold
   3. get k fold indices for training and testing splits
   4. get nested k fold indices for training validation and testing splits
   
3. evaluation class:
   1. apply cross validation on a given dataset and return the confusion matrix of the results
   2. find best pruning tree via nested cross validation and return the confusion matrix of the results
   3. pruning simulation method to simulate the pruning given a validation dataset
   4. finding the confsuion matrix between labels-predicted and real labels
   5. finding Accuracy using confusion matrix 
   6. finding Precision using confusion matrix 
   7. finding Recall using confusion matrix 
   8. finding F1-Score using confusion matrix 
   
4. tree_node class:
   1. store either a decision or leaf node based on the iniliziation using the leaf boolean
   2. if decision, it store attribute (feature number), value, leaf node, right node   
   3. if leaf, it stores label, label count

5. init.py file:   
   1. contians the main code to run.
   2. the file run each part of the coursework based on the description below using the classes implemented:
      1. step 1: the code first load both datasets sets from the txt files 
      2. step 2: plot the tree of the whole clean dataset
      3. step 3: evaluate based on cross validation of each dataset by creating the tree training on the training dataset fold and predict the test fold from the 10 folds and return there confusion matrices and then compute the recall, precision, and f1-score of the averged confusion matrix for each dataset
      4. step 4: apply pruning simulation in a nested cross validation of 10 outer folds, 10 inner folds and return the confusion matrices of the outer folds, which is evaluated on the unbaised and unused test folds for each iteration.
## Courswork parts
### part 1 loading the data:
all data is loaded into numpay arrays and segmented into x (features) and y (labels) and classes (maping between the real labels and converted labels for convience in usage)

### part 2 Creating Decision Tree based on the full clean dataset for ploting:
a full have been created based on the full clean dataset to plot a Decision Tree.

### part 3 perform cross validation of the Decision Tree to evaluate the algorithm 
perform cross validation by dividing the datasets into folds and then choose 1 fold for testing and the rest for training to evalaute the tree, then we loop through choosing other folds and keeping the rest as training to build a fresh tree, this would resuls

### part 4 perform nested cross validation to tune the pruning of a decsion tree and evaluate the algorithm
perfroming nestsed cross validation to tune a tree built on the training folds and validated for pruning using the validation fold, and after the validationa nd choosing the best from the inner fold a evalaution based on the testing fold is done to estimate the error based on unseen (unbaised) data

## Authors

* **Salim Al-Wahaibi** - *saa221@ic.ac.uk*
* **Wei Jie Chua** - *wc1021@ic.ac.uk*
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Marcos-Antonios Charalambous** - *mc921@ic.ac.uk*

## License
[MIT License](https://choosealicense.com/licenses/mit/)
