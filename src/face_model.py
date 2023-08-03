from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import RBFInterpolator
from sklearn import preprocessing
import numpy as np





class GPModel():
    def __init__(self, sensor_pos, sensor_temps):
        kernel = RBF()
        self.__gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_x_train = self.__scaler.transform(sensor_pos)

        self.__gp.fit(scaled_x_train, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        return self.__gp.predict(scaled_pos_xy)[0]
        



class IDWModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        self.__scaled_x_train = self.__scaler.transform(sensor_pos)

        self.__sensor_temps = sensor_temps

        self.__pos_to_temp = {}
        for i, pos in enumerate(self.__scaled_x_train):
            self.__pos_to_temp[tuple(pos)] = self.__sensor_temps[i]


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        if tuple(scaled_pos_xy[0]) in self.__pos_to_temp:
            return self.__pos_to_temp[tuple(scaled_pos_xy[0])]
        else:
            weights = np.zeros(len(self.__scaled_x_train))
            for i, temp in enumerate(self.__sensor_temps):
                weights[i] = 1/(5 * np.linalg.norm(self.__scaled_x_train[i] - scaled_pos_xy))
            temp_xy = np.sum(weights * self.__sensor_temps)/np.sum(weights)
            return temp_xy




class RBFModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_x_train = self.__scaler.transform(sensor_pos)

        self.__interpolater = RBFInterpolator(scaled_x_train, sensor_temps)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        return self.__interpolater(scaled_pos_xy)[0]







if __name__ == "__main__":
    sensor_positions = np.array([-0.0001364,-0.0064293,-0.0001364,-0.0092576,-0.0001364,0.0084192]).reshape(-1, 2)
    sensor_temperatures = np.array([160.5637951542764, 167.201582353572, 371.40222295188596]).reshape(-1)

    gp_model = GPModel(sensor_positions, sensor_temperatures)
    print(gp_model.get_temp(np.array([0.01, 0.01])))