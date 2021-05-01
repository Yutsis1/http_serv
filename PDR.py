import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from scipy.stats import norm


class PDR(object):
    def __init__(self, csv_path: str):
        self.df_raw = pd.read_csv(csv_path) # type: DataFrame
        self.df_raw = self.df_raw.sort_values(by=['time'])
        self.df_raw = self.df_raw.reset_index(drop=True)
        self.df = pd.DataFrame(columns=self.df_raw.columns) # type: DataFrame
        rotation_raw_w = self.df_raw.iloc[0]['rotation_rawW']
        for index, row in self.df_raw.iterrows():

            row_rotation_raw_w = row['rotation_rawW']
            if not (row_rotation_raw_w == rotation_raw_w):
                rotation_raw_w = row_rotation_raw_w
                self.df = self.df.append(self.df_raw.iloc[index])

        # print(self.df)
        # data include all columns without time
        # data_keys = list(self.df.keys())[:-1]
        # self.df_data = self.df[data_keys]
        # self.df_data = self.df_data.drop_duplicates()
        # print(self.df_data.join(self.df))
        self.df = self.df.reset_index(drop=True)
        self.df['dt'] = self.calculate_dt()
        list_t = []
        for i, elem in enumerate(self.df['dt']):
            list_t.append(float(elem + list_t[i - 1] if i > 0 else 0.0))
        self.df['t'] = list_t
        self.g = 9.8

    def coordinate_conversion(self):
        gravity, linear = [], []
        for key in self.df.keys():
            if 'gravity' in key:
                gravity.append(self.df[key])
            elif 'accel' in key:
                linear.append(self.df[key])
        gravity = np.array(gravity).T
        linear = np.array(linear).T

        # g_x = gravity[:, 0]
        g_y = gravity[:, 1]
        g_z = gravity[:, 2]

        # linear_x = linear[:, 0]
        linear_y = linear[:, 1]
        linear_z = linear[:, 2]

        theta = np.arctan(np.abs(g_z / g_y))

        # Get vertical acceleration (remove g）
        a_vertical = linear_y * np.cos(theta) + linear_z * np.sin(theta)

        return a_vertical

    def calculate_dt(self, series_time=None):
        """
        Method for calculation dt for between measurements
        :param series_time:
        :return:
        """
        series_time = series_time or self.df['time'].values
        # first dt is equal as 0 due to it start of measurement
        list_dt = [0.0]
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

    def quaternion2euler(self):
        """
        method for caculation pitch, yaw and roll

        :return:
        """
        rotation = []
        for key in self.df.keys():
            if 'rotation' in key:
                rotation.append(self.df[key])

        rotation = np.array(rotation).T
        rx = rotation[:, 0]
        ry = rotation[:, 1]
        rz = rotation[:, 2]
        rw = rotation[:, 3]

        pitch = np.arcsin(2 * (rw * ry - rz * rx))
        roll = np.arctan2(2 * (rw * rx + ry * rz), 1 - 2 * (rx * rx + ry * ry))
        yaw = np.arctan2(2 * (rw * rz + rx * ry), 1 - 2 * (rz * rz + ry * ry))
        # pitch, roll, yaw = rx, ry, rz
        return pitch, roll, yaw

    @staticmethod
    def draw_plot(y_data, x_data=None,
                  name: str = 'default',
                  x_label: str = 'X',
                  y_label: str = 'Y',
                  enable_grid=True,
                  save_data: bool = True,
                  show: bool = False,
                  path: str = './pictures_1/',
                  fmt: str = 'png',
                  *args,
                  **kwargs):
        """
        Method for draw plot and safe it as png picture if it needs
        :param show:
        :param fmt:
        :param path:
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
        plt.plot(x_data, y_data, *args, **kwargs)
        plt.title(name)
        if save_data:
            pwd = os.getcwd()
            os.chdir(path)
            plt.savefig("{}.{}".format(name, fmt))
            os.chdir(pwd)
        if show:
            plt.show()

    def step_counter(self,
                     # frequency=100,
                     ):
        # offset = frequency / 100
        # g = 9.794
        a_vertical = self.coordinate_conversion()
        # slide = frequency * offset  # Sliding window(sample data at 100Hz）
        # frequency = 100 * offset
        # Pedestrian acceleration threshold
        # min_acceleration = 0.2 # * g  # 0.2g
        # max_acceleration = 1 # * g  # 2g
        # Peak interval (s)
        # min_interval = 0.4
        # min_interval = 0.001 if walk_type == 'normal' else 3  # 'abnormal
        # max_interval = 1
        # Calculate the number of steps
        steps = []
        peak = {'index': 0, 'acceleration': 0}

        # Detect peak with 40 * offset as sliding window
        # Condition 1: The peak is between 0.2 g~2g
        for i, v in enumerate(a_vertical):
            if i>0:
                if v > a_vertical[i-1]:
                    peak['index'] = i-1
                    peak['acceleration'] = a_vertical[i-1]
                    steps.append(peak)
        #     if v > max_acceleration:
        #         peak['acceleration'] = v
        #         peak['index'] = i
        #         steps.append(peak)
        #         peak = {'index': 0, 'acceleration': 0}

            # if peak['acceleration'] <= v <= max_acceleration and v >= min_acceleration:
            #     peak['acceleration'] = v
            #     peak['index'] = i
            # # if i % slide == 0 and peak['index'] != 0:
            # elif v > 2.2:
            #     steps.append(peak)
            #     peak = {'index': 0, 'acceleration': 0}

        # Condition 2:The interval before the two peaks is at least greater than 0.4 s * offset
        # When del is used, the principle of first record and then delete is generally adopted
        # if len(steps) > 0:
        #     last_step = steps[0]
        #     dirty_points = []
        #     for key, step_dict in enumerate(steps):
        #         # print(step_dict['index'])
        #         if key == 0:
        #             continue
        #         if step_dict['index'] - last_step['index'] < min_interval * frequency:
        #             # print('last:', lastStep['index'], 'this:', step_dict['index'])
        #             if step_dict['acceleration'] <= last_step['acceleration']:
        #                 dirty_points.append(key)
        #             else:
        #                 last_step = step_dict
        #                 dirty_points.append(key - 1)
        #         else:
        #             last_step = step_dict
        #
        #     counter = 0  # The number of record deletions, as a deviation value
        #     for key in dirty_points:
        #         del steps[key - counter]
        #         counter = counter + 1

        return steps

        # Step size estimation
        # The current method is not scientific,temporary use

    @staticmethod
    def step_stride(max_acceleration):
        return np.power(max_acceleration, 1 / 4) * 0.5

        # Heading Angle
        # Use yaw directly according to posture

    def step_heading(self):
        _, _, yaw = self.quaternion2euler()
        # init_theta = yaw[0] # Initial angle
        # for i, v in enumerate(yaw):
        # yaw[i] = -(v-init_theta)
        # Since yaw is counterclockwise positive,
        # converting to clockwise is more in line with the conventional way of thinking
        # yaw[i] = -v
        return yaw

    def pdr_position(self, frequency=100,

                     offset=0,
                     init_position=(0, 0)):
        yaw = self.step_heading()
        steps = self.step_counter()
        position_x = []
        position_y = []
        x = init_position[0]
        y = init_position[1]
        position_x.append(x)
        position_y.append(y)
        strides = []
        angle = [offset]
        for v in steps:
            index = v['index']
            length = self.step_stride(v['acceleration'])
            strides.append(length)
            theta = yaw[index] + offset
            angle.append(theta)
            x = x + length * np.sin(theta)
            y = y + length * np.cos(theta)
            position_x.append(x)
            position_y.append(y)
        # The step size is counted in a state,
        # the last position has no next step,
        # so the step size is denoted as 00
        return position_x, position_y, strides + [0], angle


if __name__ == '__main__':
    pdr = PDR("Data_1.csv")
    position = pdr.pdr_position()
    print(pdr.pdr_position())
    print(pdr.pdr_position()[3])
    # accel_rawX,accel_rawY,accel_rawZ,gyro_rawX,gyro_rawY,gyro_rawZ,gravity_rawX,gravity_rawY,gravity_rawZ,rotation_rawX,rotation_rawY,rotation_rawZ,rotation_rawW,time
    # pdr.draw_plot(y_data=pdr.df['accel_rawY'], x_data=pdr.df['t'], show=True)
    # pdr.draw_plot(y_data=pdr.df['accel_rawZ'], x_data=pdr.df['t'], show=True, marker='^')
    # pdr.draw_plot(y_data=position[1], x_data=position[0], show=True, marker='^')
    pdr.draw_plot(y_data=pdr.df['accel_rawX'], x_data=pdr.df['t'], show=True, marker='^')
    print('end')
