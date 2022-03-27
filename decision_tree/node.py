class Node:
    def __init__(self, x, y, gini_imp, classes_and_counts, depth=0):
        self.x = x
        self.y = y
        self.gini_imp = gini_imp
        self.classes_and_counts = classes_and_counts
        self.left_node = None
        self.right_node = None
        self.split_rule = None
        self.depth = depth

    def __str__(self):
        prefix = "----" * self.depth
        spaces = "    " * self.depth
        return f'{prefix}| Gini impurity: {round(self.gini_imp, 2)}\n{spaces}| Classes: {self.classes_and_counts}\n{spaces}| Predicted class: {self.get_class()}'

    def get_class(self):
        biggest_observations = max(list(self.classes_and_counts.values()))
        for key in list(self.classes_and_counts.keys()):
            if self.classes_and_counts[key] == biggest_observations:
                return key
