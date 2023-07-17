import numpy as np



class PlaneModel():
    def __init__(self, pos1, pos2, pos3):
        # A plane is defined by Ax + By + CT = D where A, B, C are the components of the normals and D is a constant
        # Hence to find the T at a specific position in the plane all we need is T = (D - Ax - By)/C
        normal = np.cross(pos2 - pos1, pos3 - pos1)
        self.__A = normal[0]
        self.__B = normal[1]
        self.__C = normal[2]
        self.__D = np.dot(normal, pos1)
    

    def get_T(self, pos_xy):
        return (self.__D - self.__A * pos_xy[:, 0] - self.__B * pos_xy[:, 1])/self.__C




if __name__ == "__main__":
    plane = PlaneModel(np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([2, 4, 5]))
    print(plane.get_T(np.array([[1, 2], [2, 3], [2, 4]])))