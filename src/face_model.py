from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn import preprocessing
import numpy as np





class GPModel():
    def __init__(self, sensor_pos, sensor_temps):
        kernel = RationalQuadratic()
        self.__gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        self.__scaler = preprocessing.StandardScaler().fit(sensor_pos)
        scaled_x_train = self.__scaler.transform(sensor_pos)

        self.__gp.fit(scaled_x_train, sensor_temps)

    
    def get_temp(self, pos_xy):
        scaled_pos_xy = self.__scaler.transform(pos_xy.reshape(1, 2))
        return self.__gp.predict(scaled_pos_xy)[0]
        



class IDWModel():
    def __init__(self, sensor_pos, sensor_temps):
        self.__x_train = sensor_pos
        self.__sensor_temps = sensor_temps

        self.__pos_to_temp = {}
        for i, pos in enumerate(self.__x_train):
            self.__pos_to_temp[tuple(pos)] = self.__sensor_temps[i]


    def get_temp(self, pos_xy):
        if tuple(pos_xy) in self.__pos_to_temp:
            return self.__pos_to_temp[set(pos_xy)]
        else:
            weights = np.zeros(len(self.__x_train))
            for i, temp in enumerate(self.__sensor_temps):
                weights[i] = 1/np.linalg.norm(self.__x_train[i] - pos_xy)
            temp_xy = np.sum(weights * self.__sensor_temps)/np.sum(weights)
            return temp_xy









if __name__ == "__main__":
    sensor_positions = np.array([-0.0001364,-0.0064293,-0.0001364,-0.0092576,-0.0001364,0.0084192])
    sensor_temperatures = np.array([160.5637951542764, 167.201582353572, 371.40222295188596])

    gp_model = GPModel(sensor_positions, sensor_temperatures)
    print(gp_model.get_temp(np.array([0.01, 0.01])))