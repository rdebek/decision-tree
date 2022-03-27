import numpy as np
from scipy.stats import mode

from decision_tree.node import Node
from decision_tree.utility import get_classes_and_counts, get_gini_impurity, get_gini_gain, get_split_values


class DecisionTreeClassifier:
    def __init__(self, max_depth=25, min_gini_gain=0.0, mode='default'):
        self.x = None
        self.y = None
        self.root = None
        self.max_depth: int = max_depth
        self.allowed_modes = ['default', 'continuous-mean', 'continuous-median', 'discrete']
        self.mode = mode if mode in self.allowed_modes else 'default'
        self.min_gini_gain: float = min_gini_gain
        self.right_count, self.left_count = 0, 0

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.root = self.create_root()
        self.split_node(self.root)

    def split_node(self, node: Node):
        gini_best, split_attr_index, split_val = 0, 0, 0
        for i, attribute in enumerate(node.x.T):
            splits = get_split_values(attribute)
            for split in splits:
                left_indexes, right_indexes = self.handle_missing_values(attribute, split)
                observation_list_left = list(get_classes_and_counts([node.y[index] for index in left_indexes]).values())
                observation_list_right = list(
                    get_classes_and_counts([node.y[index] for index in right_indexes]).values())
                gini_gain = get_gini_gain(node.gini_imp, observation_list_left, observation_list_right)
                if gini_gain > gini_best:
                    gini_best = gini_gain
                    split_val, split_attr_index = split, i
                    final_left_indexes, final_right_indexes = left_indexes, right_indexes

        if gini_best > self.min_gini_gain:
            left_x, left_y = np.asarray([node.x[i] for i in final_left_indexes]), [node.y[i] for i in
                                                                                   final_left_indexes]
            right_x, right_y = np.asarray([node.x[i] for i in final_right_indexes]), [node.y[i] for i in
                                                                                      final_right_indexes]

            node.split_rule = {'split_attribute_index': split_attr_index, 'split_value': split_val,
                               'gini_gain': gini_best}
            new_node_right = Node(right_x, right_y, get_gini_impurity(list(get_classes_and_counts(right_y).values())),
                                  get_classes_and_counts(right_y), depth=node.depth + 1)
            node.right_node = new_node_right
            self.split_node(new_node_right)
            new_node_left = Node(left_x, left_y, get_gini_impurity(list(get_classes_and_counts(left_y).values())),
                                 get_classes_and_counts(left_y), depth=node.depth + 1)
            node.left_node = new_node_left
            self.split_node(new_node_left)

    def create_root(self) -> Node:
        classes_and_counts = get_classes_and_counts(self.y)
        gini_impurity = get_gini_impurity(list(classes_and_counts.values()))
        return Node(self.x, self.y, gini_impurity, classes_and_counts)

    def predict(self, x) -> list:
        classes_array = []
        x = np.asarray(x)
        if x.ndim == 1:
            x = [x]
        for record in x:
            leaf = self.get_leaf(record, self.root)
            classes_array.append(leaf)
        return classes_array

    def get_leaf(self, record, starting_node: Node):
        current_node = starting_node
        depth = self.max_depth
        while current_node.left_node and current_node.right_node and depth > 0:
            split_attribute_index = current_node.split_rule['split_attribute_index']
            if np.isnan(record[split_attribute_index]):
                return self.check_imputation_mode(record, current_node, split_attribute_index)
            elif record[split_attribute_index] <= current_node.split_rule['split_value']:
                current_node = current_node.left_node
            else:
                current_node = current_node.right_node
            depth -= 1
        return current_node.get_class()

    def check_imputation_mode(self, record, current_node: Node, split_attribute_index: int) -> int:
        if self.mode == 'default':
            return self.default_handler(record, current_node)
        if self.mode == 'continuous-mean':
            return self.continuous_mean_handler(record, current_node, split_attribute_index)
        if self.mode == 'continuous-median':
            return self.continuous_median_handler(record, current_node, split_attribute_index)
        if self.mode == 'discrete':
            return self.discrete_handler(record, current_node, split_attribute_index)

    def discrete_handler(self, record, current_node: Node, attribute_index: int, return_mode=False):
        mode_val = mode(self.x.T[attribute_index])[0][0]
        if return_mode:
            return mode_val
        record[attribute_index] = mode_val
        return self.get_leaf(record, current_node)

    def continuous_mean_handler(self, record, current_node: Node, attribute_index: int, return_mean=False):
        mean = np.nanmean(self.x.T[attribute_index])
        if return_mean:
            return mean
        record[attribute_index] = mean
        return self.get_leaf(record, current_node)

    def continuous_median_handler(self, record, current_node: Node, attribute_index: int, return_median=False):
        median = np.nanmedian(self.x.T[attribute_index])
        if return_median:
            return median
        record[attribute_index] = median
        return self.get_leaf(record, current_node)

    def default_handler(self, record, starting_node: Node):
        left_leaf_class = self.get_leaf(record, starting_node.left_node)
        right_leaf_class = self.get_leaf(record, starting_node.right_node)

        if left_leaf_class == right_leaf_class:
            return left_leaf_class
        elif starting_node.classes_and_counts[left_leaf_class] > starting_node.classes_and_counts[right_leaf_class]:
            return left_leaf_class
        else:
            return right_leaf_class

    def handle_missing_values(self, attribute, split):
        left_indexes, right_indexes = [], []
        chosen_mode = self.mode
        if chosen_mode == 'default':
            chosen_mode = 'continuous-mean'
        if chosen_mode == 'continuous-mean':
            for index, value in enumerate(attribute):
                if np.isnan(value):
                    value = np.nanmean(attribute)
                    attribute[index] = value
                if value <= split:
                    left_indexes.append(index)
                else:
                    right_indexes.append(index)
        elif chosen_mode == 'continuous-median':
            for index, value in enumerate(attribute):
                if np.isnan(value):
                    value = np.nanmedian(attribute)
                    attribute[index] = value
                if value <= split:
                    left_indexes.append(index)
                else:
                    right_indexes.append(index)
        elif chosen_mode == 'discrete':
            for index, value in enumerate(attribute):
                if np.isnan(value):
                    value = mode(attribute)[0][0]
                    attribute[index] = value
                if value <= split:
                    left_indexes.append(index)
                else:
                    right_indexes.append(index)
        return left_indexes, right_indexes

    def print_tree(self, node: Node):
        print(node)
        if node.left_node:
            self.print_tree(node.left_node)
        if node.right_node:
            self.print_tree(node.right_node)

