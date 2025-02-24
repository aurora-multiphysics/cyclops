import numpy as np
from pyomo.environ import *

class ProblemSetup(ConcreteModel):
    """Problem class; uses Pyomo ConcreteModel class to build a model
    describing the problem to be optimised."""

    def __init__(
        self,
        spatial_dims: int,
        num_obj: int,
        surf_pts: np.ndarray,
        surf_faces: list,
        loss_function: callable,
        **kwargs
    ):
        """ Set up the problem
        
            Args:
            
            returns:
        """
        super().__init__(**kwargs)
        
        # Storing model attributes
        self.__spatial_dims = spatial_dims
        self.__num_obj = num_obj
        self.__surf_pts = surf_pts
        self.__surf_faces = surf_faces
        self.__loss_function = loss_function