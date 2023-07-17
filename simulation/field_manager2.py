from simulation.gaussian_model import NoiselessGPModel
from simulation.exodus_reader import ExodusReader
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np



class FaceManager():
    def __init__(self, face_x_data, face_y_data, face_T_data):
        self.__face_x = np.array(face_x_data)
        



    def check_pos(self, pos):
        x, y = pos[0], pos[1]
        if x < -0.0112 or x > 0.0112 or y < -0.011 or y > 0.02 or (x**2 + y**2) < 0.06**2:
            return False
        return True


    def get_T(self, pos):
        pass








class LineManager():
    def __init__(self, line_y_data, line_T_data):
        self.__line_y = np.array(line_y_data)
        self.__pos_to_T = {}
        for i, pos in enumerate(line_y_data):
            self.__pos_to_T[pos] = line_T_data[i]
        self.__line_y.sort()


    def get_T(self, pos):
        for i, y_coord in enumerate(self.__line_y):
            if y_coord > pos:
                min_y = self.__line_y[i-1]
                max_y = self.__line_y[i+1]
                break
        y_diff = max_y - min_y
        T_diff = self.__pos_to_T[max_y] - self.__pos_to_T[min_y]
        return self.__pos_to_T[min_y] + (pos - min_y)/y_diff * T_diff


    def check_pos(self, y):
        if y < -0.11 or y > 0.02:
            return False
        return True
        


if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    line_y, line_T = reader.line_only(face_x, face_y, face_T)
    managerL = LineManager(line_y, line_T)


    print(managerL.get_T(np.array([[0.0]])))

