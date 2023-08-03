from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import RBFInterpolator
from sklearn import preprocessing
import numpy as np





class GPModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__gp = GaussianProcessRegressor(
            kernel=RBF(), 
            n_restarts_optimizer=10, 
            normalize_y=True
        )
        self.__gp.fit(scaled_pos, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        return self.__gp.predict(scaled_pos_xy)[0]
        



class IDWModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        self.__scaled_pos = self.__scaler.transform(sensor_pos)

        self.__sensor_temps = sensor_temps
        self.__pos_to_temp = {}
        for i, pos in enumerate(self.__scaled_pos):
            self.__pos_to_temp[tuple(pos)] = self.__sensor_temps[i]


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        if tuple(scaled_pos_xy[0]) in self.__pos_to_temp:
            return self.__pos_to_temp[tuple(scaled_pos_xy[0])]
        else:
            weights = np.zeros(len(self.__scaled_pos))
            for i, temp in enumerate(self.__sensor_temps):
                weights[i] = 1/(5 * np.linalg.norm(self.__scaled_pos[i] - scaled_pos_xy))
            temp_xy = np.sum(weights * self.__sensor_temps)/np.sum(weights)
            return temp_xy




class RBFModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_pos = self.__scaler.transform(sensor_pos)

        self.__interpolater = RBFInterpolator(scaled_pos, sensor_temps)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        return self.__interpolater(scaled_pos_xy)[0]




class UniformGPModle():
    def __init__(self, sensor_pos, sensor_temps):
        y_values = sensor_pos[:, 1].reshape(-1, 1)
        self.__scaler = preprocessing.StandardScaler().fit(y_values)
        scaled_y_values = self.__scaler.transform(y_values)

        kernel = RBF()
        self.__gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        self.__gp.fit(scaled_y_values, sensor_temps)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy[1].reshape(1, 1))
        return self.__gp.predict(scaled_pos_xy)[0]
   





class UniformRBFModel():
    def __init__(self, sensor_pos, sensor_temps):
        y_values = sensor_pos[:, 1].reshape(-1, 1)
        self.__scaler = preprocessing.StandardScaler().fit(y_values)
        scaled_y_values = self.__scaler.transform(y_values)

        self.__interpolater = RBFInterpolator(scaled_y_values, sensor_temps)


    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy[1].reshape(1, 1))
        return self.__interpolater(scaled_pos_xy)[0]
















if __name__ == "__main__":
    sensor_positions = np.array([-0.0001364,-0.0064293,-0.0001364,-0.0092576,-0.0001364,0.0084192]).reshape(-1, 2)
    sensor_temperatures = np.array([160.5637951542764, 167.201582353572, 371.40222295188596]).reshape(-1)

    gp_model = GPModel(sensor_positions, sensor_temperatures)
    print(gp_model.get_temp(np.array([0.01, 0.01])))