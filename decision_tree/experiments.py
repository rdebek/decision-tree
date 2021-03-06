from statistics import mean
from time import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier as SciKitTreeClassifier

from decision_tree.datasets import wine_dataset, breast_dataset, car_dataset
from decision_tree.decision_tree import DecisionTreeClassifier
from decision_tree.node import Node
from decision_tree.utility import flatten_array, get_missing_indexes


def compare_to_reference_test(dataset, n_tests):
    reference_results = []
    implemented_results = []
    reference_tree = SciKitTreeClassifier()
    implemented_tree = DecisionTreeClassifier()

    for i in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.7)

        start = time()
        reference_tree.fit(X_train, y_train)
        predictions = reference_tree.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        end = time()
        elapsed_time = end - start
        reference_results.append({'time': elapsed_time, 'accuracy': acc, 'predictions': predictions})

        start = time()
        implemented_tree.fit(X_train, y_train)
        predictions = implemented_tree.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        end = time()
        elapsed_time = end - start
        implemented_results.append({'time': elapsed_time, 'accuracy': acc, 'predictions': predictions})

    return reference_results, implemented_results


def parse_results(reference_results, implemented_results):
    average_reference_time = mean([test['time'] for test in reference_results])
    average_implemented_time = mean([test['time'] for test in implemented_results])

    average_reference_accuracy = mean([test['accuracy'] for test in reference_results])
    average_implemented_accuracy = mean([test['accuracy'] for test in implemented_results])

    reference_predictions = []
    reference_predictions.extend(test['predictions'] for test in reference_results)
    reference_predictions = flatten_array(reference_predictions)

    implemented_predictions = []
    implemented_predictions.extend(test['predictions'] for test in implemented_results)
    implemented_predictions = flatten_array(implemented_predictions)

    print(f'Por??wnanie zaimplementowanego drzewa decyzyjnego z wersj?? referencyjn??.\n'
          f'??redni czas budowania drzewa[s]:\n'
          f'- Drzewo referencyjne: {average_reference_time}\n'
          f'- Drzewo zaimplementowane: {average_implemented_time}\n'
          f'??rednia trafno???? przewidywa??[%]:\n'
          f'- Drzewo referencyjne: {average_reference_accuracy * 100}\n'
          f'- Drzewo zaimplementowane: {average_implemented_accuracy * 100}\n'
          f'Pokrycie przewidywa??: {accuracy_score(reference_predictions, implemented_predictions) * 100}\n')


def test_imputation_handler(attribute_index, dataset, imputation_mode):
    dummy_node = Node('test', 'test', 'test', {0: 1})
    dummy_tree = DecisionTreeClassifier()
    records = dataset.copy()
    true_values = []
    estimated_values = []
    for record in records:
        true_val = record[attribute_index]
        true_values.append(true_val)
        record[attribute_index] = np.nan
        dummy_tree.x = records
        if imputation_mode == 'mean':
            estimated_val = dummy_tree.continuous_mean_handler(record, dummy_node, attribute_index, True)
        elif imputation_mode == 'median':
            estimated_val = dummy_tree.continuous_median_handler(record, dummy_node, attribute_index, True)
        elif imputation_mode == 'mode':
            estimated_val = dummy_tree.discrete_handler(record, dummy_node, attribute_index, True)
        records[attribute_index] = true_val
        estimated_values.append(estimated_val)
    return true_values, estimated_values


def gather_mean_squared_errors(dataset, imputation_mode):
    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(dataset.data)
    mean_squared_errors = []
    amount_of_attributes = len(scaled_dataset.T)
    for i in range(amount_of_attributes):
        true_values, estimated_values = test_imputation_handler(i, scaled_dataset, imputation_mode)
        mean_squared_errors.append(mean_squared_error(true_values, estimated_values))
    plot_mean_squared_errors(amount_of_attributes, mean_squared_errors)
    return mean(mean_squared_errors)


def plot_mean_squared_errors(amount_of_attributes: int, mean_squared_errors: list):
    plt.plot([i for i in range(amount_of_attributes)], mean_squared_errors, 'bo', label='B????d ??redniokwadratowy')
    plt.plot([i for i in range(amount_of_attributes)], [mean(mean_squared_errors) for _ in range(amount_of_attributes)],
             linewidth=3.0, color='red', ls='--', zorder=0, label='??redni b????d ??redniokwadratowy')
    plt.xlabel('Numer atrybutu')
    plt.ylabel('B????d ??redniokwadratowy')
    plt.legend()
    plt.title('Wykres b????d??w ??redniokwadratowych\n (Wine Data Set)')
    plt.show()


def get_mode_accuracies(dataset):
    dummy_node = Node('test', 'test', 'test', {0: 1})
    dummy_tree = DecisionTreeClassifier()
    dummy_tree.x = dataset.data
    dataset_size = len(dataset.data)
    amount_of_attributes = len(dataset.data.T)
    helper = []
    accuracies = []
    for j in range(amount_of_attributes):
        for i in range(dataset_size):
            helper.append(dummy_tree.discrete_handler(dataset.data[i][j], dummy_node, j, True))
        accuracies.append(accuracy_score(dataset.data.T[j], helper))
        helper.clear()
    plot_mode_accuracies(accuracies)
    return accuracies


def plot_mode_accuracies(accuracies):
    accuracies = [acc * 100 for acc in accuracies]
    for j, attribute in enumerate(accuracies):
        plt.plot([j + 1 for _ in range(int(attribute))], [i for i in range(int(attribute))], lw=18, color='royalblue')
    plt.ylim(0, 50)
    plt.xlabel('Numer atrybutu')
    plt.ylabel('Procent trafionych uzupe??nie??')
    plt.title('Wykres % trafionych uzupe??nie?? w zale??no??ci od atrybutu\n (Car Evaluation Dataset)')
    plt.yticks([i for i in range(0, 51, 5)])
    plt.show()


def get_mean_median_accuracy(dataset, n_tests, n_missing_vals: int, imputation_mode: str):
    dtc = DecisionTreeClassifier(mode=imputation_mode)
    results_dict = {}
    for i in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.7)
        dtc.fit(X_train, y_train)
        no_missing_vals_acc = accuracy_score(y_test, dtc.predict(X_test))
        records = X_test.copy()
        for record in records:
            missing_indexes = get_missing_indexes(n_missing_vals, len(record))
            for index in missing_indexes:
                record[index] = np.nan
        missing_vals_acc = accuracy_score(y_test, dtc.predict(records))
        results_dict[f'test {i}'] = [no_missing_vals_acc, missing_vals_acc]
    average_no_missing_vals_acc = mean([results_dict[f'test {i}'][0] for i in range(n_tests)])
    average_missing_vals_acc = mean([results_dict[f'test {i}'][1] for i in range(n_tests)])
    return average_no_missing_vals_acc, average_missing_vals_acc


def test_and_plot_mean_median_acc(dataset, n_tests, imputation_mode):
    average_missing_vals_accuracies = []
    average_no_missing_vals_accuracies = []

    for i in range(1, len(dataset.data.T) + 1):
        avg_no_missing, avg_missing = get_mean_median_accuracy(dataset, n_tests, i, imputation_mode)
        average_no_missing_vals_accuracies.append(avg_no_missing * 100)
        average_missing_vals_accuracies.append(avg_missing * 100)

    plt.plot([i for i in range(1, len(dataset.data.T) + 1)], average_no_missing_vals_accuracies, color='red',
             label='Bez brakuj??cych warto??ci', marker='o')
    plt.plot([i for i in range(1, len(dataset.data.T) + 1)], average_missing_vals_accuracies, color='green',
             label='Z brakuj??cymi warto??ciami', marker='o')
    plt.legend()
    plt.title('Por??wnanie trafno??ci przewidywa?? -  uzupe??nianie przy pomocy ??redniej\n (Wine Data Set)')
    plt.yticks([i for i in range(40, 101, 5)])
    plt.xlabel('Liczba brakuj??cych atrybut??w')
    plt.ylabel('??rednia trafno???? przewidywa?? [%]')
    plt.show()

    return average_no_missing_vals_accuracies, average_missing_vals_accuracies


def test_own_method(dataset, n_tests):
    dtc = DecisionTreeClassifier()
    with_missing_vals = defaultdict(list)
    for i in range(n_tests):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.7)
        dtc.fit(X_train, y_train)
        for j in range(len(X_test.T)):
            records = X_test.copy()
            records[:, j] = np.nan
            with_missing_vals[j].append(accuracy_score(y_test, dtc.predict(records)) * 100)
    plot_test_own_method(with_missing_vals, len(X_test.T))
    return with_missing_vals


def plot_test_own_method(with_missing_vals, attributes):
    [plt.plot(i, mean(with_missing_vals[i]), marker='o', color='darkviolet') for i in range(attributes)]
    plt.xticks([i for i in range(attributes)])
    plt.yticks([i for i in range(50, 101, 5)])
    plt.title(
        'Wykres trafno??ci przewidywa?? w zale??no??ci od brakuj??cego atrybutu\n (Breast Cancer Wisconsin (Diagnostic) Data Set)')
    plt.xlabel('Number brakuj??cego atrybutu')
    plt.ylabel('Trafno???? przewidywa?? [%]')
    plt.show()


def test_training_with_missing_values(dataset, n_tests, n_missing_values, mode='default'):
    reference = []
    result = []
    dtc = DecisionTreeClassifier(mode=mode)
    for i in range(n_tests):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.6)
        dtc.fit(X_train, y_train)
        reference.append(accuracy_score(y_test, dtc.predict(X_test)))
        for record in X_train:
            missing_indexes = get_missing_indexes(n_missing_values, len(record))
            for index in missing_indexes:
                record[index] = np.nan
        dtc.fit(X_train, y_train)
        result.append(accuracy_score(y_test, dtc.predict(X_test)))
    return mean(reference), mean(result)


def plot_difference(x, y1, y2):
    plt.plot(x, y1, label='Mediana')
    plt.plot(x, y2, label='??rednia arytmetyczna')
    plt.xlabel('Procent brakuj??cych warto??ci w zbiorze')
    plt.ylabel('R????nica pomi??dzy otrzymanimi wynikami')
    plt.title('Wykres zale??no??ci r????nicy pomi??dzy wynikami\n od procentu brakuj??cych warto??ci w zbiorze')
    plt.legend()
    plt.show()


def plot_discrete(x, y):
    plt.plot(x, y, label='Dominanta')
    plt.xlabel('Procent brakuj??cych warto??ci w zbiorze')
    plt.ylabel('R????nica pomi??dzy otrzymanimi wynikami')
    plt.title('Wykres zale??no??ci r????nicy pomi??dzy wynikami\n od procentu brakuj??cych warto??ci w zbiorze')
    plt.legend()
    plt.show()


def run_tests():
    # Por??wnanie implementacji do referencyjnej (scikit-learn)
    reference_results, implemented_results = compare_to_reference_test(wine_dataset, 20)
    parse_results(reference_results, implemented_results)

    # Obliczenie b????d??w ??redniokwadratowych (??rednia arytmetyczna, mediana)
    gather_mean_squared_errors(wine_dataset, 'mean')

    # Test uzupe??niania za pomoc?? dominanty
    get_mode_accuracies(car_dataset)

    # Por??wnanie trafno??ci z i bez brakuj??cych warto??ci atrybut??w
    test_and_plot_mean_median_acc(wine_dataset, 10, 'continuous-mean')

    # Badanie algorytmu w??asnego
    test_own_method(breast_dataset, 10)


if __name__ == "__main__":
    run_tests()
