from src.csv_reader import SymmetricReader, UniformReader
from src.face_model import GPModel, RBFModel
from src.graph_manager import GraphManager
import numpy as np



if __name__ == '__main__':
    symmetric_reader = SymmetricReader('temperature_field.csv')
