from src.face_model import GPModel, IDWModel, RBFModel, UniformRBFModel, UniformGPModel
from results_manager import ResultsManager
from csv_reader import CSVReader
import numpy as np






def show_symmetric_setup(num_sensors, csv_reader, model_type):
    pass


def show_uniform_setup(num_sensors):
    pass



if __name__=='__main__':
    csv_reader = CSVReader(
        'temperature_field.csv',
        GPModel,
        UniformRBFModel
    )
    show_symmetric_setup(10, csv_reader)