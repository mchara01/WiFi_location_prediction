# WiFi_location_prediction
building a decision tree model to predict in which of the 4 rooms the user is standing.
the prediction is based on two datasets found in WIFI_db folder *clean_dataset.txt* and *noisy_dataset.txt*

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

### part 2 Creating Decision Trees:

### part 3

## Authors

* **Salim Al-Wahaibi** - *saa221@ic.ac.uk*
* **Wei Jie Chua** - *xxx@ic.ac.uk*
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Marcos-Antonios Charalambous** - *mc921@ic.ac.uk*

## License
[MIT License](https://choosealicense.com/licenses/mit/)
