class tree_node:

    def __init__(self, attribute, value, leaf, label):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None
        self.leaf = leaf
        self.label = label
    
    def __repr__(self):
        return str({"attribute":self.attribute,"value":self.value,"leaf":self.leaf,"label":self.label,"left":self.left,"right":self.right})

    def __str__(self):
        return {"attribute":self.attribute,"value":self.value,"leaf":self.leaf,"label":self.label,"left":self.left,"right":self.right}

    def __dir__(self) :
        return {"attribute":self.attribute,"value":self.value,"leaf":self.leaf,"label":self.label,"left":self.left,"right":self.right}
