import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.common import Saver
from scipy.integrate import trapezoid
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter


class StepAnalyzer:

    def __init__(self, csv_data_path: str):
        self.df_data_raw = self.read_data(csv_data_path=csv_data_path)
        self.data_name = csv_data_path.split('.')[0]
        self.df_data_raw['dt'] = self.calculate_dt()
        list_t = []
        for i, elem in enumerate(self.df_data_raw['dt']):
            list_t.append(float(elem + list_t[i - 1] if i > 0 else 0))
        self.df_data_raw['t'] = list_t

        #     set kalman filters
        self.kfX = self.set_ext_kalman(P=np.var(self.df_data_raw['rawX']))
        self.kfY = self.set_ext_kalman(P=np.var(self.df_data_raw['rawY']))
        self.kfZ = self.set_ext_kalman(P=np.var(self.df_data_raw['rawZ']))

        #     savers classes
        self.saverX = Saver(self.kfX)
        self.saverY = Saver(self.kfY)
        self.saverZ = Saver(self.kfZ)

        #       pints
        self.pointsX = self.kfX.batch_filter(zs=self.df_data_raw['rawX'], saver=self.saverX)
        self.pointsY = self.kfX.batch_filter(zs=self.df_data_raw['rawX'], saver=self.saverY)
        self.pointsZ = self.kfX.batch_filter(zs=self.df_data_raw['rawX'], saver=self.saverZ)

    #
    def kalman_iter(self, kf):
        pass

    @staticmethod
    def set_ext_kalman(R=1, P=1, dt=0.02):
        # set as 3*1 matrix due to we measure ddx
        kf = KalmanFilter(dim_x=3, dim_z=1)
        kf.x = np.zeros(3)
        kf.P[0, 0] = P
        kf.P[1, 1] = 1
        kf.P[2, 2] = 1
        kf.R *= R ** 2
        # kf.Q = Q_discrete_white_noise(3, dt, Q)
        kf.F = np.array([[1., dt, .5 * dt * dt],
                         [0., 1., dt],
                         [0., 0., 1.]])
        kf.H = np.array([[1., 0., 0.]])

        return kf

    @staticmethod
    def HJacobian_at(x):
        """ compute Jacobian of H matrix at x """

        horiz_dist = x[0]
        altitude = x[2]
        from math import sqrt
        denom = sqrt(horiz_dist ** 2 + altitude ** 2)
        return np.array([[horiz_dist / denom, 0., altitude / denom]])

    @staticmethod
    def hx(x):
        """ compute measurement for slant range that
        would correspond to state x.
        """
        return (x[0] ** 2 + x[2] ** 2) ** 0.5

    def calculate_vel_sum(self, series):
        for i, elem in enumerate(series):
            yield series[i - 1] + elem * self.df_data_raw['dt'][i]

    def calculate_vel_trapz(self, series):
        """
        Method which integrate data and multiply on dt for
        :param series:
        :return:
        """
        integrated_data = self.integration_data_trapz(series)
        for i, data in enumerate(integrated_data):
            yield data * self.df_data_raw['dt'][i]

    def calculate_dt(self, series_time=None):
        """
        Method for calculation dt for between measurements
        :param series_time:
        :return:
        """
        series_time = series_time or self.df_data_raw['time']
        # first dt is equal as 0 due to it start of measurement
        list_dt = [0]
        for i in range(1, len(series_time)):
            # chose only seconds
            #  dt = prev_time - current_time
            dt = float(series_time[i][-6:]) - float(series_time[i - 1][-6:])
            # assertion for prevent time travels
            assert dt >= 0, "series is not sorted element[{}]={} more than element[{}]={}".format(
                i - 1, float(series_time[i - 1][-6:]),
                i, float(series_time[i][-6:]))
            list_dt.append(dt)
        return list_dt

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
        plt.close()
        y_data = np.array(y_data)
        y_data.astype(np.float64)
        x_data = x_data if x_data.any() else np.arange(0.0, len(y_data))
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
        print(self.df_data_raw[key_for_print])

    @property
    def list_headers(self):
        """
        property for use list of headers
        :return: list of keys of self.dict_data
        """
        list_headers = []
        for key in self.df_data_raw.keys():
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
    def integration_data_trapz(data):
        """
        Generator based on trapezoid integration method
        For using this method (b-a) = 1 every time
        If require define (b-a) simplest way to multiply current value on (b-a)
        :param data: sequence for integration
        :return:
        """
        list_i = []
        for i in data:
            list_i.append(i)
            yield trapezoid(list_i)

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

            return df.reset_index(drop=True)


if __name__ == '__main__':

    def make_raw_plots(step_analyzer: StepAnalyzer):
        for key in step_analyzer.df_data_raw.keys():
            if not key == 'time':
                step_analyzer.draw_plot(y_data=step_analyzer.df_data_raw[key],
                                        save_data=True,
                                        name=key + ' ' + step_analyzer.data_name,
                                        y_label='Y',

                                        x_label='X')


    def make_raw_velocit(step_analyzer: StepAnalyzer,
                         make_plot=True, save_data=True):
        dict_data_for_plot = {}
        for key in step_analyzer.df_data_raw.keys():
            if not key == 'time':
                integrator = step_analyzer.integration_data_trapz(step_analyzer.df_data_raw[key].to_numpy(dtype=float))
                data_for_plot = [x for x in integrator]
                if make_plot:
                    step_analyzer.draw_plot(y_data=data_for_plot,
                                            save_data=save_data,
                                            name='integrated velocity' + key[-1] + ' ' + step_analyzer.data_name,
                                            y_label='Y',
                                            x_label='X')
                dict_data_for_plot[key] = data_for_plot
        return dict_data_for_plot


    #
    accel = StepAnalyzer('BMI120 Accelerometer.csv')
    gyro = StepAnalyzer('BMI120 Gyroscope.csv')

    # print([x for x in accel.calculate_vel(accel.df_data_raw['rawX'])])
    # accel.draw_plot(y_data=[x for x in accel.calculate_vel_trapz(accel.df_data_raw['rawZ'])],
    #                 x_data=accel.df_data_raw['t'])
    # accel.draw_plot(y_data=[x for x in accel.calculate_vel_sum(accel.df_data_raw['rawY'])],
    #                 x_data=accel.df_data_raw['t'])
    # gyro.draw_plot(y_data=[x for x in gyro.calculate_vel(gyro.df_data_raw['rawY'])],
    #                 x_data=gyro.df_data_raw['t'])
    # velocity_accel = make_raw_velocit(accel, make_plot=True)
    # integrator_way = accel.integration_data_trapz(velocity_accel)
    # # accel.draw_plot(y_data=accel,
    # #                 save_data=True,
    # #                 name='integrated velocity' + key[-1] + ' ' + step_analyzer.data_name,
    # #                 y_label='Y',
    # #                 x_label='X')
    # make_raw_plots(accel)
    #
    # velocity_gyro = make_raw_velocit(gyro, make_plot=True)
    # make_raw_plots(gyro)
    print("end")
