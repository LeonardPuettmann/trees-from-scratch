from sklearn.tree import DecisionTreeRegressor 
import numpy as np 

class GradientBoostingRegressor():
    """
    Gradient boosts a decision tree regressor. 
    """
    def __init__(self, n_trees, learning_rate, max_depth):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, x, y):
        self.trees = []
        self.F0 = y.mean()
        Fm = self.F0
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(x, y - Fm)
            Fm += self.learning_rate * tree.predict(x)
            self.trees.append(tree)

    def predict(self, x):
        return self.F0 + self.learning_rate * np.sum([t.predict(x) for t in self.trees], axis=0)