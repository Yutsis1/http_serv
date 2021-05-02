from itertools import chain
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints, ExtendedKalmanFilter as EKF, \
    JulierSigmaPoints

from PDR import PDR


class Trace(PDR):
    def __init__(self, csv_path,
                 v0=np.array([0.0, 0.0, 0.0]),
                 a0=np.array([0.0, 0.0, 0.0]),
                 s0=np.array([0.0, 0.0, 0.0])):
        super(Trace, self).__init__(csv_path=csv_path)
        pitch, roll, yaw = self.quaternion2euler()
        self.Rm_raw = R.from_rotvec((pitch[0], roll[0], yaw[0])).as_matrix()

        self.v = v0
        self.a = a0
        self.s = s0

        self.set_ukf()

        # self.df_gyro, self.df_acc = self.filtered_data_separate()
        self.set_whole_ukf()
        self.set_pdr_ukf()
        # self.set_whole_ekf()

    def update_rotation_matrix_raw(self, vector, dt):
        vector = np.array(vector).T
        sigma = abs(vector * dt)
        sin_sigma = np.sin(sigma)
        cos_sigma = np.cos(sigma)
        sin_sigma_derevetive_sigma = (sin_sigma / sigma)
        second = (1 - cos_sigma) / np.square(sigma)
        skew_semetrik = np.exp(self.skew_matrix_from_vector(vector) * dt)
        self.Rm_raw *= np.eye(3) + \
                       sin_sigma_derevetive_sigma @ skew_semetrik + \
                       second @ np.square(skew_semetrik)
        return self.Rm_raw

    def update_rotation_matrix_scipy(self, vector):
        """

        :param vector:
        :return:
        """
        self.Rm_raw = R.from_rotvec(vector).as_matrix()
        return self.Rm_raw

    def update_acceleration(self, acc):
        """

        :param acc: 3 element of acceleration without gravity [ax, ay, az]
        :return:
        """
        self.a = self.Rm_raw @ np.asarray(acc).T
        return self.a

    def update_velocity(self, dt):
        self.v = self.v + self.a * dt
        return self.v

    def update_s(self, dt):
        self.s = self.s + self.update_velocity(dt)
        return self.s

    def set_ukf(self):
        def fx_9(x, dt):
            """

            :param x: array like [ddx, dx, x, ddy, dy, y, ddz, dz , z]
            :param dt:
            :return:
            """
            X = np.array([[1, dt, 0.5 * dt ** 2],
                          [0, 1, dt],
                          [0, 0, 1]], dtype=float)
            Y = np.array([[1, dt, 0.5 * dt ** 2],
                          [0, 1, dt],
                          [0, 0, 1]], dtype=float)
            Z = np.array([[1, dt, 0.5 * dt ** 2],
                          [0, 1, dt],
                          [0, 0, 1]], dtype=float)
            F = block_diag(X, Y, Z)

            return F @ x

        def fx_6(x, dt):
            """

            :param x: array like [ddrx, drx, rx, ddry, dry, ry, ddrz, drz , rz]
            :param dt:
            :return:
            """
            # X = np.array([[1, dt, 0],
            #               [0, 1, 0],
            #               [0, 0, 1]], dtype=float)
            # Y = np.array([[1, dt, 0],
            #               [0, 1, 0],
            #               [0, 0, 1]], dtype=float)
            # Z = np.array([[1, dt, 0],
            #               [0, 1, 0],
            #               [0, 0, 1]], dtype=float)
            X = np.array([[1, dt],
                          [0, 1]], dtype=float)
            Y = np.array([[1, dt],
                          [0, 1]], dtype=float)
            Z = np.array([[1, dt],
                          [0, 1]], dtype=float)
            F = block_diag(X, Y, Z)

            return F @ x

        def hx_9(x):
            return np.array([x[2], x[5], x[8]])

        def hx_6(x):
            return np.array([x[1], x[2], x[5]])

        self.ukf_acc = UKF(dim_x=9,
                           dim_z=3, fx=fx_9, hx=hx_9, dt=self.df['dt'].mean(),
                           points=MerweScaledSigmaPoints(n=9,
                                                         alpha=0.1,
                                                         beta=2.,
                                                         kappa=-1))
        self.ukf_acc.R = np.diag([self.df['accel_rawX'].var(),
                                  self.df['accel_rawY'].var(),
                                  self.df['accel_rawZ'].var()])

        self.ukf_gyro = UKF(dim_x=6,
                            dim_z=3,
                            fx=fx_6,
                            hx=hx_6,
                            dt=self.df['dt'].mean(),
                            points=MerweScaledSigmaPoints(n=6,
                                                          alpha=0.1,
                                                          beta=2.,
                                                          kappa=-1))
        self.ukf_gyro.R = np.diag([self.df['gyro_rawX'].var(),
                                   self.df['gyro_rawY'].var(),
                                   self.df['gyro_rawZ'].var()])

    def set_whole_ukf(self):
        def fx(x, dt):
            Rp = R.from_rotvec(np.array([x[10], x[12], x[14]]))  # type: R

            # x_a = Rm.as_matrix() @ np.array([x[0], x[3], x[6]], dtype=float)
            # x_v = Rm.as_matrix() @ np.array([x[1], x[4], x[7]], dtype=float)
            # x_s = Rm.as_matrix() @ np.array([x[2], x[5], x[8]], dtype=float)
            # x_w = Rm.as_matrix() @ np.array([x[9], x[11], x[13]], dtype=float)
            # x_r = Rm.as_matrix() @ np.array([x[10], x[12], x[14]], dtype=float)
            # x[0], x[3], x[6] = x_a[0], x_a[1], x_a[2]
            # x[1], x[4], x[7] = x_v[0], x_v[1], x_v[2]
            # x[2], x[5], x[8] = x_s[0], x_s[1], x_s[2]
            # x[9], x[11], x[13] = x_w[0], x_w[1], x_w[2]
            # x[10], x[12], x[14] = x_r[0], x_r[1], x_r[2]
            def skew_matrix_from_vector(vector):
                return np.array([[0, -vector[2], vector[1]],
                                 [vector[2], 0, -vector[0]],
                                 [-vector[1], vector[0], 0]])

            I = np.eye(3, 3)
            vector_w = np.array([x[9], x[11], x[13]])
            skew_matrix_from_vectro_w = skew_matrix_from_vector(vector_w)
            Rn = Rp * ((2 * I + skew_matrix_from_vectro_w * dt) / (2 * I - skew_matrix_from_vectro_w * dt))

            Sa = skew_matrix_from_vector(np.array([x[0], x[3], x[6]], dtype=float))

            X_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)
            Y_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)

            Z_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)

            X_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Y_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Z_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            # F_w = block_diag(X_w, Y_w, Z_w)
            list_matrix = [X_a, Y_a, Z_a,
                           X_w, Y_w, Z_w]
            F = block_diag(*list_matrix)
            return F @ x

        def hx(x):
            # return np.array([x[2], x[5], x[8], x[10], x[12], x[14]])
            # return np.array([x[0], x[3], x[6], x[9], x[11], x[13]])
            return x[[0, 3, 6, 9, 11, 13]]

        self.ukf_whole = UKF(dim_x=15,
                             dim_z=6,
                             fx=fx,
                             hx=hx,
                             dt=self.df['dt'].mean(),
                             points=JulierSigmaPoints(15, kappa=0.1))
        # points=MerweScaledSigmaPoints(n=15,
        #                               alpha=0.1,
        #                               beta=2.,
        #                               kappa=-1))
        self.ukf_whole.R = np.diag([self.df['accel_rawX'].var(),
                                    self.df['accel_rawY'].var(),
                                    self.df['accel_rawZ'].var(),
                                    self.df['gyro_rawX'].var(),
                                    self.df['gyro_rawY'].var(),
                                    self.df['gyro_rawZ'].var()])

    def set_pdr_ukf(self):
        def fx(x, dt):
            X_a = np.array([[1, dt, 0.5 * (dt ** 2)],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)

            Y_a = np.array([[1, dt, 0.5 * (dt ** 2)],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)

            Z_a = np.array([[1, dt, 0.5 * (dt ** 2)],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)

            X_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Y_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Z_w = np.array([[1, dt],
                            [0, 1]], dtype=float)

            list_matrix = [X_a, Y_a, Z_a,
                           X_w, Y_w, Z_w]
            F = block_diag(*list_matrix)
            return F @ x

        def hx(x):
            # return np.array([x[2], x[5], x[8], x[10], x[12], x[14]])
            # return np.array([x[0], x[3], x[6], x[9], x[11], x[13]])
            return x[[0, 3, 6, 9, 11, 13]]

        self.ukf_pdr = UKF(dim_x=15,
                           dim_z=6,
                           fx=fx,
                           hx=hx,
                           dt=self.df['dt'].mean(),
                           points=JulierSigmaPoints(15, kappa=0.1))
        # points=MerweScaledSigmaPoints(n=15,
        #                               alpha=0.1,
        #                               beta=2.,
        #                               kappa=-1))
        self.ukf_pdr.R = np.diag([self.df['accel_rawX'].var(),
                                  self.df['accel_rawY'].var(),
                                  self.df['accel_rawZ'].var(),
                                  self.df['gyro_rawX'].var(),
                                  self.df['gyro_rawY'].var(),
                                  self.df['gyro_rawZ'].var()])

        self.ukf_pdr.P *= 100

    def set_whole_ekf(self):
        def fx(dt):
            X_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)
            Y_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)
            Z_a = np.array([[1, dt, 0.5 * dt ** 2],
                            [0, 1, dt],
                            [0, 0, 1]], dtype=float)
            F_a = block_diag(X_a, Y_a, Z_a)

            X_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Y_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            Z_w = np.array([[1, dt],
                            [0, 1]], dtype=float)
            F_w = block_diag(X_w, Y_w, Z_w)

            F = block_diag(X_a, Y_a, Z_a,
                           X_w, Y_w, Z_w)
            return F

        def hx(x):
            # np.array([x[2] * np.cos, x[5], x[8], x[10], x[12], x[14]])
            Rm = R.from_rotvec(np.array([x[10], x[12], x[14]]))  # type: R

            return np.sqrt(x[2] ** 2 + x[5] ** 2 + x[8] ** 2)

        def HJacobian_at(x):
            Rm_c = R.from_rotvec(np.array([x[10], x[12], x[14]]).T)  # type: R
            Rm = Rm_c.as_matrix()

            return block_diag(Rm, np.ones((3, 3)))

        self.ekf_whole = EKF(dim_x=15,
                             dim_z=6)
        self.ekf_whole.F = fx(dt=self.df['dt'].mean())

        self.hx = hx
        self.HJacobian_at = HJacobian_at

    @staticmethod
    def skew_matrix_from_vector(vector):
        """
        set skew symmetric matrix
        :param vector: array like with 3 elements
        :return:
        """
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])

    def unfiltered_trace(self):
        gyro = []
        accel = []
        rotation = []
        for key in self.df.keys():
            if 'gyro' in key:
                gyro.append(self.df[key])
            if 'accel' in key:
                accel.append(self.df[key])
            if 'rotation' in key:
                rotation.append(self.df[key])
        rotation = rotation[:3]

        pitch, roll, yaw = self.quaternion2euler()
        list_s = []
        list_v = []
        list_rotation_matrix = []
        list_rotation_matrix_scipy = []
        list_acc = []
        for wx, wy, wz, ax, ay, az, dt, p, r, ya in zip(gyro[0], gyro[1], gyro[2],
                                                        accel[0], accel[1], accel[2],
                                                        list(self.df['dt'])[1:],
                                                        rotation[0], rotation[1], rotation[2]):
            vector = (wx, wy, wz)
            acc = (ax, ay, az)
            list_rotation_matrix.append(self.update_rotation_matrix_raw(vector=vector, dt=dt))
            # list_rotation_matrix_scipy.append(trace.update_rotation_matrix_scipy((p, r, ya)))
            list_acc.append(self.update_acceleration(acc))
            list_v.append(self.update_velocity(dt))
            list_s.append(self.update_s(dt))

        return list_rotation_matrix, list_acc, list_v, list_s

    def filtered_data_separate(self,
                               update_df=False) -> Tuple[DataFrame, DataFrame]:
        list_filtered_w = []
        list_filtered_acc = []
        for i, row in self.df.iterrows():
            if i == 0:
                continue

            r = row['dt']
            self.ukf_acc.predict(dt=row['dt'])
            self.ukf_gyro.predict(dt=row['dt'])

            # update acceleration
            z_acc = np.array([row['accel_rawX'],
                              row['accel_rawY'],
                              row['accel_rawZ']])
            self.ukf_acc.update(z=z_acc)
            list_filtered_acc.append(self.ukf_acc.x)

            # update gyro
            z_gyro = np.array([row['gyro_rawX'],
                               row['gyro_rawY'],
                               row['gyro_rawZ']])
            self.ukf_gyro.update(z=z_gyro)
            list_filtered_w.append(self.ukf_gyro.x)
        array_filtered_w = np.array(list_filtered_w)
        array_filtered_acc = np.array(list_filtered_acc)
        list_acc_columns = ['f_a_X', 'f_a_Y', 'f_a_Z',
                            'f_v_X', 'f_v_Y', 'f_v_Z',
                            'f_p_X', 'f_p_Y', 'f_p_Z', ]
        df_acc = DataFrame(array_filtered_acc, columns=list_acc_columns)
        list_gyro_columns = ['f_w_X', 'f_w_Y', 'f_w_Z',
                             'f_r_X', 'f_r_Y', 'f_r_Z']
        df_gyro = DataFrame(array_filtered_w, columns=list_gyro_columns)

        if update_df:
            self.df = self.df.join(df_acc)
            self.df = self.df.join(df_gyro)

        return df_gyro, df_acc

    def filtered_pdr_data(self,
                          update_df=False) -> DataFrame:

        steps_up, steps_down = self.step_counter()
        df_steps = pd.concat([steps_up, steps_down])
        df_steps = df_steps.sort_index()
        # list_dt = self.calculate_dt(df_steps['time'])
        # for ldt, sdt in zip()
        list_filtered = []

        # for i, row in self.df.iterrows():
        i = 0
        row = self.df.iloc[i]
        t = self.df.iloc[i]['t']
        dt = t - row['t']
        z_array = np.array([row['accel_rawX'],
                            row['accel_rawY'],
                            row['accel_rawZ'],
                            row['gyro_rawX'],
                            row['gyro_rawY'],
                            row['gyro_rawZ']])
        self.ukf_pdr.x[0], self.ukf_pdr.x[3], self.ukf_pdr.x[6], self.ukf_pdr.x[9], \
        self.ukf_pdr.x[11], self.ukf_pdr.x[13] = z_array
        self.ukf_pdr.predict(dt=dt)
        for i, row in df_steps.iterrows():
            if i == 0:
                continue
            # self.ukf_whole.predict(dt=row['dt'])

            # update
            z_array = np.array([row['accel_rawX'],
                                row['accel_rawY'],
                                row['accel_rawZ'],
                                row['gyro_rawX'],
                                row['gyro_rawY'],
                                row['gyro_rawZ']
                                ])
            vector = (self.ukf_pdr.x[10],
                      self.ukf_pdr.x[12],
                      self.ukf_pdr.x[14])
            # R_matrix = R.from_rotvec(vector).as_matrix()
            # z_array[0:3] = R_matrix @ z_array[0:3]
            # z_array[3:] = R_matrix @ z_array[3:]

            self.ukf_pdr.update(z=z_array)

            list_filtered.append(self.ukf_pdr.x)
            dt = row['t'] - t
            t = row['t']
            self.ukf_pdr.predict(dt=dt)
        array_filtered = np.array(list_filtered)
        list_columns = ['f_a_X', 'f_a_Y', 'f_a_Z',
                        'f_v_X', 'f_v_Y', 'f_v_Z',
                        'f_p_X', 'f_p_Y', 'f_p_Z',
                        'f_w_X', 'f_w_Y', 'f_w_Z',
                        'f_r_X', 'f_r_Y', 'f_r_Z'
                        ]

        df_pdr = DataFrame(array_filtered, columns=list_columns)

        if update_df:
            self.df = self.df.join(df_pdr)

        return df_pdr

    def filtered_data_whole(self,
                            update_df=False):
        list_filtered = []

        for i, row in self.df.iterrows():
            self.ukf_whole.predict(dt=row['dt'])
            # self.ukf_whole.predict(dt=dt)

            # update
            z_array = np.array([row['accel_rawX'],
                                row['accel_rawY'],
                                row['accel_rawZ'],
                                row['gyro_rawX'],
                                row['gyro_rawY'],
                                row['gyro_rawZ']
                                ])
            # vector = (self.ukf_whole.x[10],
            #           self.ukf_whole.x[12],
            #           self.ukf_whole.x[14])
            # R_matrix = R.from_rotvec(vector).as_matrix()
            # z_array[0:3] = R_matrix @ z_array[0:3]
            # z_array[3:] = R_matrix @ z_array[3:]

            self.ukf_whole.update(z=z_array)

            list_filtered.append(self.ukf_whole.x)
        array_filtered = np.array(list_filtered)
        list_columns = ['f_a_X', 'f_a_Y', 'f_a_Z',
                        'f_v_X', 'f_v_Y', 'f_v_Z',
                        'f_p_X', 'f_p_Y', 'f_p_Z',
                        'f_w_X', 'f_w_Y', 'f_w_Z',
                        'f_r_X', 'f_r_Y', 'f_r_Z'
                        ]

        df_whole = DataFrame(array_filtered, columns=list_columns)

        if update_df:
            self.df = self.df.join(df_whole)

        return df_whole


if __name__ == '__main__':
    trace = Trace(csv_path='Data_1.csv')
    list_rotation_matrix, list_acc, list_v, list_s = trace.unfiltered_trace()

    # y_data = trace.df['f_p_X']
    # x_data = trace.df['f_p_Y']
    # trace.draw_plot(y_data=x_data, x_data=y_data, show=True)

    # for i, row in trace.df.iterrows():
    #     z_array = np.array([row['accel_rawX'],
    #                         row['accel_rawY'],
    #                         row['accel_rawZ'],
    #                         row['gyro_rawX'],
    #                         row['gyro_rawY'],
    #                         row['gyro_rawZ']
    #                         ])
    #
    #     trace.ekf_whole.predict_update(z=z_array,
    #                                    Hx=trace.hx,
    #                                    HJacobian=trace.HJacobian_at)
    #     print(trace.ekf_whole.x)
    #
    # df = trace.filtered_data_whole()
    # df_gyro, df_acc = trace.filtered_data_separate()
    df = trace.filtered_data_whole()
    df_pdr = trace.filtered_pdr_data()
    # trace.draw_plot(y_data=df_acc['f_p_Y'],
    #                 x_data=df_acc['f_p_X'],
    #                 show=True,
    #                 marker='^')
    trace.draw_plot(
        # y_data=df_pdr['f_a_X'],
        y_data=df['f_p_Y'],
        # x_data=df_pdr['f_p_X'],
        # x_data=trace.df['t'],
        show=True,
        marker='^'
    )
    print(trace.df['accel_rawX'].var())
