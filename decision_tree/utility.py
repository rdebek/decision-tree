from statistics import mean
from random import randint


def get_classes_and_counts(y: list) -> dict:
    classes_with_counts = {}
    for _class in y:
        if _class not in classes_with_counts:
            classes_with_counts[_class] = 1
        else:
            classes_with_counts[_class] += 1
    return classes_with_counts


def get_gini_impurity(observations: list) -> float:
    gini_helper = 0
    for observation in observations:
        gini_helper += pow((observation / sum(observations)), 2)
    return 1 - gini_helper


def get_gini_gain(gini_parent: float, observation_list_left: list, observation_list_right: list) -> float:
    gini_right = get_gini_impurity(observation_list_right)
    gini_left = get_gini_impurity(observation_list_left)

    observations_right = sum(observation_list_right)
    observations_left = sum(observation_list_left)

    observations_sum = observations_right + observations_left

    return gini_parent - ((gini_left * observations_left / observations_sum) + (
            gini_right * observations_right / observations_sum))


def get_split_values(attribute: list) -> list:
    splits = []
    attribute = sorted(attribute)
    for i in range(len(attribute) - 1):
        splits.append(mean(attribute[i:i + 2]))
    return splits


def flatten_array(array: list) -> list:
    helper_arr = []
    for x in array:
        for y in x:
            helper_arr.append(y)
    return helper_arr


def get_missing_indexes(n_indexes, record_size):
    return_arr = []
    for i in range(n_indexes):
        random_index = randint(0, record_size-1)
        while random_index in return_arr:
            random_index = randint(0, record_size-1)
        return_arr.append(random_index)
    return return_arr



