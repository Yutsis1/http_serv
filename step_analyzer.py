import csv
from typing import Dict
import os
import matplotlib.pyplot as plt
import numpy as np
import pykalman


class StepAnalyzer:

    def __init__(self, csv_data_path: str):
        self.dict_data = self.read_data(csv_data_path=csv_data_path)

    def draw_plot(self,
                  data,
                  name: str = 'default',
                  save_data: bool = False):
        y_data = np.array(data)
        y_data.astype(np.float64)
        x_data = np.arange(0.0, len(y_data))
        plt.plot(x_data, y_data)
        plt.title(name)
        if save_data:
            self.save_plot_to_format(name=name)
        plt.show()

    def print_data(self, key_for_print):
        """
        method for print data from self.dict_data for specific key
        :param key_for_print: the key which data should be printed
        :return:
        """
        print(self.dict_data[key_for_print])

    @property
    def list_headers(self):
        """
        property for use list of headers
        :return: list of keys of self.dict_data
        """
        list_headers = []
        for key in self.dict_data.keys():
            list_headers.append(key)
        return list_headers

    @staticmethod
    def save_plot_to_format(name: str, fmt: str = 'png'):
        """
        method for save
        :param name:
        :param fmt:
        :return:
        """
        pwd = os.getcwd()
        path = './pictures/'
        os.chdir(path)
        plt.savefig("{}.{}".format(name, fmt))
        os.chdir(pwd)

    @staticmethod
    def read_data(csv_data_path: str,
                  print_data: bool = False) -> Dict[str, list]:
        """
        Method for read data and sort it to dict.
        It method used instead of DictReader due to in file the data isn't sorted
        :param csv_data_path: path to csv file
        :param print_data: debug option flag
        :return:
        """
        with open(csv_data_path, newline='') as f:
            csv_reader = csv.reader(f)
            import operator
            sorted_data = sorted(csv_reader, key=operator.itemgetter(3), reverse=False)
            dict_data = dict.fromkeys(sorted_data[-1])

            for key, i in zip(dict_data.keys(), range(len(sorted_data[-1]))):
                dict_data[key] = []
                for val in sorted_data[:-1]:
                    dict_data[key].append(val[i])
            if print_data:
                print(dict_data)
            return dict_data


if __name__ == '__main__':

    accel = StepAnalyzer('BMI120 Accelerometer.csv')
    for key in accel.dict_data.keys():
        if not key == 'time':
            accel.draw_plot(data=accel.dict_data[key],
                            save_data=True,
                            name=key)

