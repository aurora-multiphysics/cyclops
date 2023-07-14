from gaussian_model import NoiselessGPModel
from exodus_reader import ExodusReader
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np



class FaceManager():
    def __init__(self, face_x_data, face_y_data, face_T_data, max_items):
        if max_items < len(face_x_data):
            face_x = np.zeros((max_items, 1))
            face_y = np.zeros((max_items, 1))
            face_T = np.zeros((max_items, 1))
            for i in range(max_items):
                rand_i = np.random.randint(0, len(face_x_data))
                face_x[i] = face_x_data[rand_i]
                face_y[i] = face_y_data[rand_i]
                face_T[i] = face_T_data[rand_i]
        else:
            face_x = np.array(face_x_data).reshape(-1, 1)
            face_y = np.array(face_y_data).reshape(-1, 1)
            face_T = np.array(face_y_data).reshape(-1, 1)

        face_pos = np.column_stack((face_x, face_y))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(face_x_data, face_y_data, face_T_data, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        self.__model = NoiselessGPModel(face_pos, face_T)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        temp_data = self.__model.predict(np.column_stack((np.array(face_x_data).reshape(-1, 1), np.array(face_y_data).reshape(-1, 1))), std_out = False).reshape(-1)
        print(temp_data)
        surf = ax.plot_trisurf(face_x_data, face_y_data, temp_data, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close()



    def check_pos(self, pos):
        x, y = pos[0], pos[1]
        if x < -0.0112 or x > 0.0112 or y < -0.011 or y > 0.02 or (x**2 + y**2) < 0.06**2:
            return False
        return True


    def get_T(self, pos):
        if self.check_pos(pos) == False:
            raise Exception("Invalid position")
        else:
            if len(pos) == 1:
                return self.__model.predict(pos)[0], 0
            else:
                test_T, std_T = self.__model.predict(pos)
                return test_T, std_T








class LineManager():
    def __init__(self, line_y_data, line_T_data, max_items):
        if max_items < len(line_y_data):
            line_y = np.zeros((max_items, 1))
            line_T = np.zeros((max_items, 1))
            for i in range(max_items):
                rand_i = np.random.randint(0, len(line_y_data))
                line_y[i] = line_y_data[rand_i]
                line_T[i] = line_T_data[rand_i]
        else:
            line_y = np.array(line_y_data).reshape(-1, 1)
            line_T = np.array(line_T_data).reshape(-1, 1)

        self.__model = NoiselessGPModel(line_y, line_T)
        self.__model.plot(x_test = np.linspace(-0.012, 0.02, 100).reshape(-1, 1))


    def get_T(self, pos):
        if len(pos) == 1:
            return self.__model.predict(pos)[0], 0
        else:
            test_T, std_T = self.__model.predict(pos)
            return test_T, std_T


    def check_pos(self, y):
        if y < -0.11 or y > 0.02:
            return False
        return True
        


if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    line_y, line_T = reader.line_only(face_x, face_y, face_T)
    managerL = LineManager(line_y, line_T, 100)


    print(managerL.get_T(np.array([[0.0]])))

    print("\n")
    managerF = FaceManager(face_x, face_y, face_T, 1000)

