# WiFi location prediction using DT
In this project, we developed a Decision Tree model to predict in which of the 4 rooms the user is standing.
The prediction is based on two datasets found in the WiFi_db directory; *clean_dataset.txt* and *noisy_dataset.txt*.
We then evaluated how well our model performs on each of the two datasets according to various
evaluation metrics. More details regarding the evaluation can be found in the report.


## Code Execution Instructions

Run the following command to execute the code:

1. Clone the project via HTTPS with: `git clone https://github.com/mchara01/WiFi_location_prediction.git` <br>
1. cd into the cloned directory with: `cd WiFi_location_prediction`  <br>
1. Run the project with the following commands: `python3 init.py` <br>

This project requires Python >= 3.5 <br>

## Project Specifics
The init.py is the project's main file. It is the module were all of the results are printed, according to the coursework sections. 
It starts by loading both the clean and the noisy datasets from the wifi_db directory. 
To change the datasets, just add the new datasets to the wifi_db directory and change the constant variable at the start of init.py.

Once the script is executed, it will develop a Decision Tree, based on the full clean dataset, which will then plot using recursion. <br>
The first section of evaluation, will evaluate the algorithm via cross-validation by creating Decision Trees based on the training folds and the testing them with the test folds. This separation is made to make sure the model is unbiased to the test data and gives the true evaluation from the distribution of the unseen data, avoiding overfitting. <br>
The second part of the script will perform nested cross-validation to tune and evaluate pruned versions of the trees.

## Report
The report thoroughly describes all of the evaluation statistics done on the un-pruned and pruned tree via cross-validation and nested cross-validation respectively. 
Furthermore, it includes a visualisation of the full decision tree, which was trained with the clean dataset. 

## Project layout
Structure of repository:
`````
WiFi_location_prediction
├── README.md
├── WIFI_db
│ ├── clean_dataset.txt
│ └── noisy_dataset.txt
├── co553_DTcoursework_V21_22.pdf
├── dataset.py
├── decision_tree.py
├── dt_bonus.png
├── evaluation.py
├── final_result.txt
├── init.py
└── tree_node.py
`````
Our project's codebase consists of:

1. decision_tree.py:
    1. Training and predicting algorithm along with the required methods to train, such as finding information gain, entropy and finding the best split based on the information gain.
    2. Plotting the tree. 
    
2. dataset.py:
   1. Reading the datasets and dividing the data into features, labels and classes (mapping between the real labels and converted labels for convenience in usage).
   2. Get k-fold splits by dividing the dataset into k-fold.
   3. Get k-fold indices for training and testing splits.
   4. Get nested k-fold indices for training validation and testing splits.
   
3. evaluation.py:
   1. Apply cross-validation on a given dataset and return the confusion matrix of the results.
   2. Find the best pruning tree via nested cross-validation and return the confusion matrix of the results.
   3. Pruning simulation method to simulate the pruning given a validation dataset.
   4. Finding the confusion matrix between labels-predicted and real labels.
   5. Finding Accuracy using a confusion matrix.
   6. Finding Precision using a confusion matrix.
   7. Finding Recall using a confusion matrix.
   8. Finding F1-Score using a confusion matrix.
   
4. tree_node.py:
   1. Store either a decision or leaf node based on the initialisation using the leaf boolean.
   2. If a decision node, it stores attribute (feature number), value, left node, right node. 
   3. If a leaf node, it stores label and label count.

5. init.py:   
   1. Contains the main code to run.
   2. This file runs each part of the coursework based on the description below using the above modules:
      1. Step 1: The code first loads both datasets sets from the text files. 
      2. Step 2: Plot the tree using the whole clean dataset. 
      3. Step 3: Evaluate based on cross-validation of each dataset by creating the tree training on the training dataset fold and predicting the test fold from the 10 folds and return their confusion matrices and then compute the *recall*, *precision*, and *f1-score* of the averaged confusion matrix for each dataset.
      4. Step 4: Apply pruning simulation in nested cross-validation of 10 outer folds, 10 inner folds and return the confusion matrices of the outer folds, which are evaluated on the unused test folds for each iteration.

## Coursework parts

### Part 1 - Loading the data:
All data is loaded into NumPy arrays and segmented into x (features) and y (labels) and classes (mapping between the real labels and converted labels for convenience in usage).

### Part 2 - Creating Decision Tree based on the full clean dataset for plotting:
Full Decision Tree creation based on the full clean dataset to plot a Decision Tree.

### Part 3 - Perform cross-validation of the Decision Tree to evaluate the algorithm:
Perform cross-validation by dividing the datasets into folds and then choose 1 fold for testing and the rest for training to evaluate the tree, then we loop through choosing other folds and keeping the rest as training to build a fresh tree, this would results.

### Part 4 - Perform nested cross-validation to tune the pruning of a decision tree and evaluate the algorithm:
Performing nested cross-validation to tune a tree built on the training folds and validated for pruning using the validation fold. After validating and choosing the best from the inner fold, an evaluation based on the testing fold is done to estimate the error based on unseen (unbiased) data. 

## Authors

* **Salim Al-Wahaibi** - *saa221@ic.ac.uk*
* **Wei Jie Chua** - *wc1021@ic.ac.uk*
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Marcos-Antonios Charalambous** - *mc921@ic.ac.uk*

## License
[MIT License](https://choosealicense.com/licenses/mit/)