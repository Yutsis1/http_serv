import csv
from typing import Dict
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pykalman


class StepAnalyzer:

    def __init__(self, csv_data_path: str):
        self.df_data = self.read_data(csv_data_path=csv_data_path)

    def draw_plot(self, y_data, x_data=None,
                  name: str = 'default',
                  x_label: str = 'X',
                  y_label: str = 'Y',
                  enable_grid=False,
                  save_data: bool = False,
                  *args):
        """
        Method for draw plot and safe it as png picture if it needs
        :param y_data: data for the Y-axis
        :param x_data: data for X-axis (if none set a sequence from 0 to number of items in y_data)
        :param name: name for a picture
        :param x_label: name for X-axis
        :param y_label: name for Y-axis
        :param enable_grid: enable grid if it needs
        :param save_data: flag for save picture to ./pictures
        :param args: addition args for plt
        :return:
        """
        y_data = np.array(y_data)
        y_data.astype(np.float64)
        x_data = x_data or np.arange(0.0, len(y_data))
        # set XY labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if enable_grid:
            plt.grid()
        plt.plot(x_data, y_data, args)
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
        print(self.df_data[key_for_print])

    @property
    def list_headers(self):
        """
        property for use list of headers
        :return: list of keys of self.dict_data
        """
        list_headers = []
        for key in self.df_data.keys():
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
                  print_data: bool = False) -> pd.DataFrame:
        """
        Method for read data and sort it to pandas.DataFrame.
        :param csv_data_path: path to csv file
        :param print_data: debug option flag
        :return:
        """

        def check_debug_option(print_option):
            if print_data:
                print(print_option)

        with open(csv_data_path, newline='') as f:

            df = pd.read_csv(csv_data_path)
            check_debug_option(df)

            # reorder
            df = df.sort_values(by=['time'])
            check_debug_option(df)

            return df


if __name__ == '__main__':

    accel = StepAnalyzer('BMI120 Accelerometer.csv')
    # accel.read_data('BMI120 Accelerometer.csv', print_data=True)
    for key in accel.df_data.keys():
        if not key == 'time':
            accel.draw_plot(y_data=accel.df_data[key],
                            save_data=True,
                            name=key)
