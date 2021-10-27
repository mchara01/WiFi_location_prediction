class TreeNode:

    def __init__(self, attribute, value, left, right, leaf, label, label_counts):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.label = label
        self.label_counts = label_counts

    def set_leaf(self, label, label_counts):
        self.attribute = None
        self.value = None
        self.left = None
        self.right = None
        self.leaf = True
        self.label = label
        self.label_counts = label_counts

    def change_attribute(self, node):
        self.attribute = node.attribute
        self.value = node.value
        self.left = node.left
        self.right = node.right
        self.leaf = node.leaf
        self.label = node.label
        self.label_counts = node.label_counts

    def clone(self):
        return TreeNode(self.parent_node, self.attribute, self.value, self.left, self.right, self.leaf, self.label,
                         self.label_counts)

    def __repr__(self):
        return str({"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                    "label_counts": self.label_counts, "left": self.left, "right": self.right})

    def __str__(self):
        return {"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                "label_counts": self.label_counts, "left": self.left, "right": self.right}

    def __dir__(self):
        return {"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                "label_counts": self.label_counts, "left": self.left, "right": self.right}
