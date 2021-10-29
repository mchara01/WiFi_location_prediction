"""
The TreeNode object implementation.
"""


class TreeNode:

    def __init__(self, attribute, value, left, right, leaf, label, label_counts):
        """Initialisation of TreeNode object parameters"""
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf
        self.label = label
        self.label_counts = label_counts

    # def set_leaf(self, label, label_counts):
    #     """Set a TreeNode object as a leaf.
    #
    #     This function set the calling TreeNode object as a leaf. This is done
    #     by nullifying (making None) all of its parameters besides the label and
    #     the label_counts
    #
    #      Args:
    #         label ([type]): [description]
    #         label_counts ([type]): [description]
    #
    #     Returns:
    #         [type]: [description]
    #     """
    #     self.attribute = None
    #     self.value = None
    #     self.left = None
    #     self.right = None
    #     self.leaf = True
    #     self.label = label
    #     self.label_counts = label_counts

    def change_attribute(self, node):
        """Copy attributes from a given node.

        Args:
            node (TreeNode): The node from which the calling node will copy the parameters from
        """
        self.attribute = node.attribute
        self.value = node.value
        self.left = node.left
        self.right = node.right
        self.leaf = node.leaf
        self.label = node.label
        self.label_counts = node.label_counts

    def clone(self):
        return TreeNode(self.parent_node, self.attribute, self.value, self.left, self.right, self.leaf, self.label, self.label_counts)

    def __repr__(self):
        return str({"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                    "label_counts": self.label_counts, "left": self.left, "right": self.right})

    def __str__(self):
        return {"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                "label_counts": self.label_counts, "left": self.left, "right": self.right}

    def __dir__(self):
        return {"attribute": self.attribute, "value": self.value, "leaf": self.leaf, "label": self.label,
                "label_counts": self.label_counts, "left": self.left, "right": self.right}
