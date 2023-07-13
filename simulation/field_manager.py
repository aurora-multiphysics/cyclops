from gaussian_model import ExactGPModel
from exodus_reader import ExodusReader
from matplotlib import pyplot as plt
import torch




class FaceManager():
    def __init__(self, face_x, face_y, face_T):
        self.__face_T = torch.tensor(face_T)
        self.__num_pos = self.__face_T.size()

        self.__face_x = torch.tensor(face_x)
        self.__face_y = torch.tensor(face_y)
        
    

    def get_T(self, pos):
        # Checks it is a valid position on the face
        x_pos, y_pos = pos[0], pos[1]
        if x_pos < -0.0115 or x_pos > 0.0115:
            raise Exception("X pos out of range")
        if y_pos < -0.0115 or y_pos > 0.0195:
            raise Exception("Y pos out of range")
        if x_pos**2 + y_pos**2 < 0.006**2:
            raise Exception("In the circle")
        
        # Get closest coordinate
        x_close = torch.square(self.__face_x - x_pos * torch.ones(self.__num_pos))
        y_close = torch.square(self.__face_y - y_pos * torch.ones(self.__num_pos))
        pos_index = torch.argmin(torch.add(x_close, y_close))
        return face_T[pos_index]

        
    
    def compare_T(self, arr_T):
        return torch.sum(torch.abs(torch.sub(arr_T, self.__face_T)))



class LineManager():
    def __init__(self, line_y, line_T):
        self.__model = ExactGPModel(torch.tensor(line_y), torch.tensor(line_T))
        self.__model.learn(repetitions=100)
        self.__model.plot(test_x=torch.linspace(-0.011, 0.019, 100))


    def get_T(self, pos):
        return self.__model.predict(test_x=torch.tensor(pos)).mean.numpy()

        


if __name__ == "__main__":
    reader = ExodusReader("monoblock_out.e")
    face_x, face_y, face_T = reader.front_only()
    line_y, line_T = reader.line_only(face_x, face_y, face_T)
    manager = LineManager(line_y, line_T)
