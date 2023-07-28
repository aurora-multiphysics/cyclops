import numpy as np





class PlaneModel():
    def __init__(self, sensor_pos, sensor_temps):
        # A plane is defined by Ax + By + CT = D where A, B, C are the components of the normals and D is a constant
        # Hence to find the T at a specific position in the plane all we need is T = (D - Ax - By)/C
        sensor_temps = sensor_temps.reshape(-1, 1)

        pos1 = np.concatenate((sensor_pos[0:2], sensor_temps[0]))
        pos2 = np.concatenate((sensor_pos[2:4], sensor_temps[1]))
        pos3 = np.concatenate((sensor_pos[4:6], sensor_temps[2]))

        normal = np.cross(pos2 - pos1, pos3 - pos1)
        self.__A = normal[0]
        self.__B = normal[1]
        self.__C = normal[2]
        self.__D = np.dot(normal, pos1)
    

    def get_temp(self, pos_xy):
        # We return the temperature at a specific x and y position
        temperature_tensor = (self.__D - self.__A * pos_xy[0] - self.__B * pos_xy[1])/self.__C
        return temperature_tensor




if __name__ == "__main__":
    sensor_positions = np.array([-0.0001364,-0.0064293,-0.0001364,-0.0092576,-0.0001364,0.0084192])
    sensor_temperatures = np.array([160.5637951542764, 167.201582353572, 371.40222295188596])
    plane = PlaneModel(sensor_positions, sensor_temperatures)
    print(plane.get_temp(np.array([0, 0.1])))