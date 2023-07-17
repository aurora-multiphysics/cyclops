from simulation.gaussian_model import NoiselessGPModel
from simulation.exodus_reader import ExodusReader
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np



class FaceManager():
    def __init__(self, face_x_data, face_y_data, face_T_data, max_items):
        if max_items < len(face_x_data):
            face_x, face_y, face_T = self.shorten(face_x_data, face_y_data, face_T_data, max_items)
        else:
            face_x = np.array(face_x_data).reshape(-1, 1)
            face_y = np.array(face_y_data).reshape(-1, 1)
            face_T = np.array(face_y_data).reshape(-1, 1)

        face_pos = np.column_stack((face_x, face_y))
        self.__model = NoiselessGPModel(face_pos, face_T)
        self.plot_model(face_x_data, face_y_data, face_T_data)


    
    def plot_model(self, face_x_data, face_y_data, face_T_data):
        # Plot the perfect temperature surface
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_trisurf(face_x_data, face_y_data, face_T_data, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Plot the GP's interpretation of the temperature surface
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        temp_data = self.__model.predict(np.column_stack((np.array(face_x_data).reshape(-1, 1), np.array(face_y_data).reshape(-1, 1))), std_out = False).reshape(-1)
        surf = ax.plot_trisurf(face_x_data, face_y_data, temp_data, cmap=cm.jet, linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        plt.close()


    def shorten(self, face_x_data, face_y_data, face_T_data, max_items):
        face_x = np.zeros((max_items, 1))
        face_y = np.zeros((max_items, 1))
        face_T = np.zeros((max_items, 1))
        for i in range(max_items):
            rand_i = np.random.randint(0, len(face_x_data))
            face_x[i] = face_x_data[rand_i]
            face_y[i] = face_y_data[rand_i]
            face_T[i] = face_T_data[rand_i]
        return face_x, face_y, face_T


    def check_pos(self, pos):
        # Check if the position is a valid position
        x, y = pos[0], pos[1]
        if x < -0.012 or x > 0.012 or y < -0.011 or y > 0.02 or (x**2 + y**2) < 0.06**2:
            return False
        return True


    def get_T(self, pos):
        # Return the temperature (and standard deviation if possible) at a specific position
        test_T = self.__model.predict(pos, std_out = False)
        return test_T








class LineManager():
    def __init__(self, line_y_data, line_T_data, max_items):
        if max_items < len(line_y_data):
            line_y, line_T = self.shorten(line_y_data, line_T_data, max_items)
        else:
            line_y = np.array(line_y_data).reshape(-1, 1)
            line_T = np.array(line_T_data).reshape(-1, 1)

        self.__model = NoiselessGPModel(line_y, line_T)
        self.__model.plot(x_test = np.linspace(-0.012, 0.02, 100).reshape(-1, 1))


    def shorten(self, line_y_data, line_T_data, max_items):
        # Compress the line_y and line_T arrays by extracting elements randomly
        line_y = np.zeros((max_items, 1))
        line_T = np.zeros((max_items, 1))

        for i in range(max_items):
            rand_i = np.random.randint(0, len(line_y_data))
            line_y[i] = line_y_data[rand_i]
            line_T[i] = line_T_data[rand_i]
        return line_y, line_T


    def get_T(self, pos):
        # Return the temperature at a specific position using the model
        if len(pos) == 1:
            return self.__model.predict(pos)[0], 0
        else:
            test_T, std_T = self.__model.predict(pos)
            return test_T, std_T


    def check_pos(self, y):
        # Check if the position is a valid position
        if y < -0.11 or y > 0.02:
            return False
        return True
        


if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    line_y, line_T = reader.line_only(face_x, face_y, face_T)
    managerL = LineManager(line_y, line_T, 100)


    print(managerL.get_T(np.array([[0.0]])))

    managerF = FaceManager(face_x, face_y, face_T, 500)
    print(managerF.get_T(np.array([
    [ 0.00370761,  0.01909151],
    [ 0.01087046, -0.00770011],
    [-0.00662228, -0.00953438],
    [ 0.00573314,  0.00117392],
    [-0.00063428, -0.01048563]])))