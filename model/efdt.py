from .tree import Node, Tree


class EfdtNode(Node):
    def __init__(self):
        super().__init__()


class EfdtTree(Tree):
    def __init__(self):
        super().__init__()
        self.root = EfdtNode()

    # given training samples (X, y), grows a tree
    def fit(self, X, y):
        pass

    # update the tree by adding more training examples
    def update(self, X, y):
        pass

    # give prediction of X_test
    def predict(self, X_test):
        pass
