from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np


class CarDataset():
    def __init__(self):
        self.target = None
        self.data = None
        self.open_file()
        self.lb = LabelEncoder()
        self.lb2 = LabelEncoder()
        self.transform_attributes()

    def open_file(self):
        helper_array = []
        with open('../datasets/car.data', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                helper_array.append(row)
        self.parse_array(helper_array)

    def parse_array(self, array):
        data_array = []
        target_array = []
        for record in array:
            data_array.append(record[:-1])
            target_array.append(record[-1])
        self.data = np.asarray(data_array)
        self.target = target_array

    def transform_attributes(self):
        for i in range(len(self.data.T)):
            self.data.T[i] = self.lb2.fit_transform(self.data.T[i])
        self.target = self.lb.fit_transform(self.target)
        self.data = self.data.astype(float)


wine_dataset = load_wine()
breast_dataset = load_breast_cancer()
car_dataset = CarDataset()
