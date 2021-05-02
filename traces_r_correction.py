import numpy as np
import pandas as pd
from numpy.linalg import inv
from pandas import DataFrame
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints, ExtendedKalmanFilter as EKF, \
    JulierSigmaPoints
from scipy.spatial.transform import Rotation as R

from trace_without_r_correction import Trace


class Trace_r(Trace):
    def __init__(self, csv_path,
                 v0=np.array([0.0, 0.0, 0.0]),
                 a0=np.array([0.0, 0.0, 0.0]),
                 s0=np.array([0.0, 0.0, 0.0])):
        super(Trace_r, self).__init__(csv_path, v0=np.array([0.0, 0.0, 0.0]),
                                      a0=np.array([0.0, 0.0, 0.0]),
                                      s0=np.array([0.0, 0.0, 0.0]))
        # np.seterr(divide='ignore', invalid='ignore')

    # Override
    def set_whole_ukf(self):
        def fx(x, dt):
            R_ = R.from_rotvec(np.array([x[0], x[1], x[2]]))  # type: R
            Rp = R_.as_matrix()

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
            vector_w = np.array([x[3], x[4], x[5]])
            skew_matrix_from_vector_w = skew_matrix_from_vector(vector_w)
            Rn = Rp * ((2 * I + skew_matrix_from_vector_w * dt) * inv(2 * I - skew_matrix_from_vector_w * dt))

            Sa = skew_matrix_from_vector(np.array([x[12], x[13], x[14]], dtype=float))

            F = np.eye(15)  # type: np.ndarray

            # Add correction values
            F[0:3, 3:6] += Rn * dt
            F[9:12, 12:15] += Rn * dt
            # add skew symmetric matrix to help for calculations
            F[9:12, 0:3] -= Sa * dt

            # update velocity
            F[6:9, 9:12] = np.eye(3) * dt
            return F @ x

        def hx(x):
            # return x[[3, 4, 5, 12, 13, 14]]
            return x[[0, 1, 2, 12, 13, 14]]

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

    # Override
    def filtered_data_whole(self,
                            update_df=False):
        list_filtered = []

        for i, row in self.df.iterrows():
            self.ukf_whole.predict(dt=row['dt'])
            # self.ukf_whole.predict(dt=dt)

            # update
            z_array = np.array([
                row['accel_rawX'],
                row['accel_rawY'],
                row['accel_rawZ'],
                row['gyro_rawX'],
                row['gyro_rawY'],
                row['gyro_rawZ'],

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
        list_columns = [
            'f_w_X', 'f_w_Y', 'f_w_Z',
            'f_r_X', 'f_r_Y', 'f_r_Z',
            'f_p_X', 'f_p_Y', 'f_p_Z',
            'f_v_X', 'f_v_Y', 'f_v_Z',
            'f_a_X', 'f_a_Y', 'f_a_Z',
        ]

        df_whole = DataFrame(array_filtered, columns=list_columns)

        if update_df:
            self.df = self.df.join(df_whole)

        return df_whole

        # Override

    def set_pdr_ukf(self):
        def fx(x, dt):
            R_ = R.from_rotvec(np.array([x[0], x[1], x[2]]))  # type: R
            Rp = R_.as_matrix()

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
            vector_w = np.array([x[3], x[4], x[5]])
            skew_matrix_from_vector_w = skew_matrix_from_vector(vector_w)
            Rn = Rp * ((2 * I + skew_matrix_from_vector_w * dt) * inv(2 * I - skew_matrix_from_vector_w * dt))

            Sa = skew_matrix_from_vector(np.array([x[12], x[13], x[14]], dtype=float))

            F = np.eye(15)  # type: np.ndarray

            # Add correction values
            F[0:3, 3:6] += Rn * dt
            F[9:12, 12:15] += Rn * dt
            # add skew symmetric matrix to help for calculations
            F[9:12, 0:3] -= Sa * dt

            # update velocity
            F[6:9, 9:12] = np.eye(3) * dt
            return F @ x

        def hx(x):
            # return x[[3, 4, 5, 12, 13, 14]]
            return x[[0, 1, 2, 12, 13, 14]]

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
        self.ukf_whole.R = np.diag([
            self.df['gyro_rawX'].var(),
            self.df['gyro_rawY'].var(),
            self.df['gyro_rawZ'].var(),
            self.df['accel_rawX'].var(),
            self.df['accel_rawY'].var(),
            self.df['accel_rawZ'].var(),
        ])

        # Override

    def filtered_pdr_data(self,
                          update_df=False):
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
        z_array = np.array([
            row['gyro_rawX'],
            row['gyro_rawY'],
            row['gyro_rawZ'],
            row['accel_rawX'],
            row['accel_rawY'],
            row['accel_rawZ'],
        ])
        self.ukf_pdr.x[0], self.ukf_pdr.x[3], self.ukf_pdr.x[6], self.ukf_pdr.x[9], \
        self.ukf_pdr.x[11], self.ukf_pdr.x[13] = z_array
        self.ukf_pdr.predict(dt=dt)
        for i, row in df_steps.iterrows():
            self.ukf_whole.predict(dt=row['dt'])
            # self.ukf_whole.predict(dt=dt)

            # update
            z_array = np.array([
                row['gyro_rawX'],
                row['gyro_rawY'],
                row['gyro_rawZ'],
                row['accel_rawX'],
                row['accel_rawY'],
                row['accel_rawZ'],
            ])

            self.ukf_whole.update(z=z_array)

            list_filtered.append(self.ukf_whole.x)
        array_filtered = np.array(list_filtered)
        list_columns = [
            'f_w_X', 'f_w_Y', 'f_w_Z',
            'f_r_X', 'f_r_Y', 'f_r_Z',
            'f_p_X', 'f_p_Y', 'f_p_Z',
            'f_v_X', 'f_v_Y', 'f_v_Z',
            'f_a_X', 'f_a_Y', 'f_a_Z',
        ]

        df_whole = DataFrame(array_filtered, columns=list_columns)

        if update_df:
            self.df = self.df.join(df_whole)

        return df_whole


if __name__ == '__main__':
    trace = Trace_r(csv_path='Data_1.csv')
    df = trace.filtered_data_whole()
    trace.draw_plot(x_data=df['f_p_X'],
                    y_data=df['f_p_Y'],
                    show=True,
                    marker='x'
                    )
    print(df)
