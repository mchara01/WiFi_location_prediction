import numpy as np
from decision_tree import decision_tree
import pprint

pp = pprint.PrettyPrinter(indent=4)
x = np.array([[1,1],[2,2],[3,3]])
y = np.array([1,2,2])

# print(decision_tree.find_split(x,y))

pp.pprint(decision_tree.decision_tree_learning(x,y,0)[0].__dict__)