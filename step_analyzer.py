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
        self.pos_keys = list(self.df_data_raw.keys())[:-1]
        self.data_name = csv_data_path.split('.')[0]
        self.df_data_raw['dt'] = self.calculate_dt()
        list_t = []
        for i, elem in enumerate(self.df_data_raw['dt']):
            list_t.append(float(elem + list_t[i - 1] if i > 0 else 0))
        self.df_data_raw['t'] = list_t

        #     set kalman filters
        self.kfX = self.set_kalman_filter(P=np.var(self.df_data_raw['rawX']))
        self.kfY = self.set_kalman_filter(P=np.var(self.df_data_raw['rawY']))
        self.kfZ = self.set_kalman_filter(P=np.var(self.df_data_raw['rawZ']))

        #     savers classes
        self.saverX = Saver(self.kfX)
        self.saverY = Saver(self.kfY)
        self.saverZ = Saver(self.kfZ)

        #       pints
        self.kfX.batch_filter(zs=self.df_data_raw['rawX'], saver=self.saverX)
        self.kfY.batch_filter(zs=self.df_data_raw['rawY'], saver=self.saverY)
        self.kfZ.batch_filter(zs=self.df_data_raw['rawZ'], saver=self.saverZ)

        for data, axis in zip([self.saverX, self.saverY, self.saverZ],
                              ['X', 'Y', 'Z']):
            df = pd.DataFrame(np.array(data.x))
            df.columns = [x + axis for x in ['kf_', 'kf_v_', 'kf_p_']]
            self.pos_keys.extend(df.columns)
            self.df_data_raw = self.df_data_raw.join(df)

    @staticmethod
    def set_kalman_filter(R=1, P=1, dt=0.02):
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
    def set_ext_kalman_filter(R=1, P=1, dt=0.02):
        ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        ekf.x = np.zeros(3)
        ekf.P[0, 0] = P
        ekf.P[1, 1] = 1
        ekf.P[2, 2] = 1
        ekf.R *= R ** 2
        ekf.F = np.array([[1., dt, .5 * dt * dt],
                          [0., 1., dt],
                          [0., 0., 1.]])
        ekf.H = np.array([[1., 0., 0.]])

    @staticmethod
    def HJacobian_at(x, dt):
        """ compute Jacobian of H matrix at x """
        return [x[0], x[0] * dt, x[0] * dt + 0.5 * x[0] * (dt ** 2)]

    @staticmethod
    def hx(x):
        """
        """
        return x[0]

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
                  save_data: bool = True,
                  show: bool = False,
                  path: str = './pictures/',
                  fmt: str = 'png',
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
        x_data = x_data if not (x_data is None) else np.arange(0.0, len(y_data))
        # set XY labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if enable_grid:
            plt.grid()
        plt.plot(x_data, y_data, args)
        plt.title(name)
        if save_data:
            pwd = os.getcwd()
            os.chdir(path)
            plt.savefig("{}.{}".format(name, fmt))
            os.chdir(pwd)
        if show:
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

    def make_raw_plots(step_analyzer: StepAnalyzer, **kwargs):
        for key in step_analyzer.pos_keys:
            step_analyzer.draw_plot(y_data=step_analyzer.df_data_raw[key],
                                    save_data=True,
                                    name=key + ' ' + step_analyzer.data_name,
                                    y_label='Y',
                                    x_label='X', **kwargs)


    def make_raw_velocit(step_analyzer: StepAnalyzer,
                         make_plot=True, save_data=True, **kwargs):
        dict_data_for_plot = {}
        for key in step_analyzer.pos_keys:
            integrator = step_analyzer.integration_data_trapz(step_analyzer.df_data_raw[key].to_numpy(dtype=float))
            data_for_plot = [x * y for x, y in zip(integrator, step_analyzer.df_data_raw['dt'])]
            if make_plot:
                step_analyzer.draw_plot(y_data=data_for_plot,
                                        save_data=save_data,
                                        name='integrated velocity' + key[-1] + ' ' + step_analyzer.data_name,
                                        y_label='Y',
                                        x_label='X', **kwargs)
            dict_data_for_plot[key] = data_for_plot
        return dict_data_for_plot


    def make_raw_position(step_analyzer: StepAnalyzer,
                          make_plot=True, save_data=True, **kwargs):
        dict_data_for_plot = {}
        for key in step_analyzer.pos_keys:
            integrator = step_analyzer.integration_data_trapz(step_analyzer.df_data_raw[key].to_numpy(dtype=float))
            integrator = step_analyzer.integration_data_trapz(integrator)
            data_for_plot = [x for x in integrator]

            if make_plot:
                step_analyzer.draw_plot(y_data=data_for_plot,
                                        save_data=save_data,
                                        name='integrated position' + key[-1] + ' ' + step_analyzer.data_name,
                                        y_label='Y',
                                        x_label='X',
                                        **kwargs
                                        )
            dict_data_for_plot[key] = data_for_plot
        return dict_data_for_plot


    accel = StepAnalyzer('BMI120 Accelerometer.csv')
    gyro = StepAnalyzer('BMI120 Gyroscope.csv')
    #
    for data in [accel, gyro]:
        y_data = [0.0]
        for i, x in enumerate(data.df_data_raw['dt']):
            y_data.append(y_data[i] + float(x))
        y_data.pop(0)
        print(y_data)
        make_raw_plots(step_analyzer=data, x_data=y_data)
        make_raw_velocit(step_analyzer=data, x_data=y_data)
        make_raw_position(step_analyzer=data, x_data=y_data)
    print(type(accel.df_data_raw))
    print("end")
