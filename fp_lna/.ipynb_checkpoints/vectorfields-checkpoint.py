"""
pre-defined convenience Vector-Fields
"""
import numpy as np
from fplanck_fmod.utility import value_to_vector
from scipy.interpolate import RegularGridInterpolator


# def gaussian_potential(center, width, amplitude):
#     """A Gaussian potential

#     Arguments:
#         center    center of Gaussian (scalar or vector)
#         width     width of Gaussian  (scalar or vector)
#         amplitude amplitude of Gaussian, (negative for repulsive)
#     """

#     center = np.atleast_1d(center)
#     ndim = len(center)
#     width = value_to_vector(width, ndim)

#     def potential(*args):
#         U = np.ones_like(args[0])

#         for i, arg in enumerate(args):
#             U *= np.exp(-np.square((arg - center[i])/width[i]))

#         return -amplitude*U

#     return potential


def vectorfield_from_data(grid, data):
    """create a vector field from data on a grid
    
    Arguments:
        grid     list of grid arrays along each dimension
        data     potential data
    """
    grid = np.asarray(grid)
    if grid.ndim == data.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)
    def vectorfield(*args):
        return f(args)

    return vectorfield
