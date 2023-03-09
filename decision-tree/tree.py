def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    # Counts the number of each type of example in the dataset
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    # Calculate the gini impurity
    # See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    # The uncertainty of the starting node, minus the weighted impurity of the two child nodes
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows, header):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    # !!! Very important !!! The last feature of the list/ rows is the label
    # The label should NOT be included in the feature list
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val, header)

            # Try splitting on the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the dataset
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

def build_tree(rows, header):
    """
    Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    # Return a leaf is no further information gain
    gain, question = find_best_split(rows, header)
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value to partition on
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch
    true_branch = build_tree(true_rows, header)

    # Recursively build the false branch
    false_branch = build_tree(false_rows, header)

    return Decision_Node(question, true_branch, false_branch)    

def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", str(node.predictions))
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    # Base case, we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true or false branch
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs
    
class Question():
    """
    A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """
    def __init__(self, column, value, header):
        self.column = column
        self.value = value 
        self.header = header
    
    def match(self, example):
        val = example[self.column]
        if str(val).isnumeric():
            return val >= self.value
        else:
            return val == self.value
        
    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = '=='
        if str(self.value).isnumeric():
            condition = '>='
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value)) # Header is a variable from outside the function

class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    """
    A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self,
                question,
                true_branch,
                false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch