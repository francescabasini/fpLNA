import numpy as np
#from fplanck_fmod.utility import value_to_vector
from scipy.interpolate import RegularGridInterpolator
import warnings
from scipy.interpolate import interp1d
from numpy import linalg as LA

# CASE 1: 1d projection
def find_proj_tangent(xy_coord_p, s1, s0, m_tang_STAB):
    """
    Function that projects a point onto tangent line along a direction,
    takes as input:
    xy_coord_p  - point coordinates
    s1          - slope of tangent line
    s0          - intercept of tangent line
    m_tang_STAB - slope of direction 

    Note: s1 and s0 are given in f(x) format
    """
    # unstable man tangent
    # m2 = 1/s1
    # q2 = -s0/s1
    
    m2 = s1
    q2 = s0
    # Is it Plus or minus? m tang?
    q1 = xy_coord_p[1]- m_tang_STAB*xy_coord_p[0]
    #q1 = s0_stab
    m1 = m_tang_STAB
    
    #  intersection
    intersection_x = (q2-q1)/(m1-m2)
    intersection_y = m1*intersection_x+q1
    #intersection_y = m2*intersection_x+q2
    #intersection_y = (m2*q1-m1*q2)/(m1-m2)
    
    return [intersection_x, intersection_y]
    
def project_observed_on_Wu(observed_data, s1, s0, m_tang_STAB, direction_X_scaled, xs, spline_tau_gives_s):
    """
    Function that takes observed data in 2d and projects them onto 
    unstable manifold in arclength parameterisation
    """
    nn_data = observed_data.shape[0]
    points_on_tangent = np.array([find_proj_tangent(observed_data[data_i, :], s1, s0, m_tang_STAB) for data_i in range(nn_data)])
    # projected_sims_Together = proj_all_time(Stoch_sims_all_together, 1/the_landscape.m_unstable, -the_landscape.q_unstable/the_landscape.m_unstable,
                               #the_landscape.m_stable, the_landscape.q_unstable)
    # from tangent to Wu - based on the y
    tau_sims = (points_on_tangent[:,1]-xs[1])/direction_X_scaled[1]
    #spline from tau to s-arclength
    observed_on_Wu = spline_tau_gives_s(tau_sims)
    return observed_on_Wu

## CASE 2: 2D reconstruction
def find_peaks_variances(s_fp, sol_fp):
    saddle_pos = np.abs(s_fp).argmin()
    sol_fp_cut = sol_fp[saddle_pos:]
    # rough peaks
    peaks_positions = [sol_fp.argmax(), sol_fp_cut.argmax()+saddle_pos]
    # spline function
    fp_spline_fun = interp1d(s_fp, sol_fp, kind='cubic')

    handful_point = 10
    exact_peaks = [] 
    exact_peak_values = []
    
    sigmas_s = []
    for ii in range(2):
        # finding exact peaks
        peak_pos = peaks_positions[ii]
        # left-right sides
        x_side = s_fp.data[peak_pos-handful_point:peak_pos+handful_point+1]
        y_side = sol_fp.data[peak_pos-handful_point:peak_pos+handful_point+1]
        a, b, c = np.polyfit(x_side, y_side, 2)
        exact_peaks.append(-b/(2*a))
        exact_peak_values.append(fp_spline_fun(exact_peaks[ii]))
        warnings.filterwarnings("ignore")
        
        # Finding Variances
        # center
        x_side_center = x_side - exact_peaks[ii]

        a_center, b_center, c_center = np.polyfit(x_side_center, y_side, 2)
        sigmas_s.append(np.sqrt(-1/a_center))
    return exact_peaks, sigmas_s

# weights of the two mixtures
def find_weights(s_fp, sol_fp):
    ds = s_fp.data[1]-s_fp.data[0]
    # normalising constant
    Z_fp = np.sum(sol_fp.data*ds)

    saddle_pos = np.abs(s_fp).argmin()
    weight_left = np.sum(sol_fp.data[:saddle_pos]*ds)/Z_fp
    weight_right = np.sum(sol_fp.data[saddle_pos:]*ds)/Z_fp
    return np.array([weight_left, weight_right])
    
# def proj_all_time(xy_matrix_sims, s1, s0, m_tang_STAB):
#     proj_all_time_mat = np.ones(xy_matrix_sims.shape)
#     for tt in range(xy_matrix_sims.shape[2]):
#         for gg in range(xy_matrix_sims.shape[1]):
#             proj_all_time_mat[:,gg, tt] = find_proj_tangent(xy_matrix_sims[:,gg, tt], s1, s0, m_tang_STAB)
#     return(proj_all_time_mat)


def gausMix_twoD_pdf(means, variances, weights):
    """A N dimensional Mixture of n Gaussians probability distribution function

    Arguments:
        means          means of Gaussians (n vector)
        variances      covariance structure of Gaussians (n vector)
        weights        weights of the n-gaussians (n vector)
    """

    ndim = len(weights)

    def pdf(*args): # matrix of data, x, y
        values = np.zeros(args[0])
        
        for N, args in range(args):
            # for each component
            k = data_xy.shape[1]
            for i in range(ndim):
                kernels = np.ones_like(args[0])
                # each 2d-gaussian
                kernels *= np.exp(-(data_xy[nn]-means[i])@LA.inv(variances[i])@(data_xy[nn]-means[i])/2)/np.sqrt((2*np.pi)**k * LA.det(variances[i]))
                #kernels *= 1/np.sqrt(2*np.pi*variances[i]) * np.exp(-np.square((arg - means[i]))/(variances[i]*2))
                values += kernels*weights[i]
            return values
    return pdf

######## FUNCTIONS EXISISTING

def gausMix_oneD_pdf(means, variances, weights):
    """A 1 dimensional Mixture of n Gaussians probability distribution function

    Arguments:
        means          means of Gaussians (n vector)
        variances      covariance structure of Gaussians (n vector)
        weights        weights of the n-gaussians (n vector)
    """

    ndim = len(means)

    def pdf(*args): #only gives x
        values = np.zeros_like(args[0])
        
        for N, arg in enumerate(args):
            # would work in N dim
            for i in range(ndim):
                kernels = np.ones_like(args[0])
                kernels *= 1/np.sqrt(2*np.pi*variances[i]) * np.exp(-np.square((arg - means[i]))/(variances[i]*2))
                values += kernels*weights[i]
            return values
    return pdf

def pdf0_from_data(range_pdf, data_pdf):
    """create the initial distribution from data on a grid
    
    Arguments:
        grid     list of grid arrays along each dimension
        data     pdf0 data
    """
    grid = np.asarray(range_pdf)
    if grid.ndim == data_pdf.ndim == 1:
        grid = (grid,)

    f = RegularGridInterpolator(grid, data_pdf, bounds_error=False, fill_value=None)
    def pdf(*args):
        values = f(args)
        #values = interp_f(given_sim_grid)
        return values/np.sum(values)

    # still normalize?
    return pdf


#------------ WORK IN PROGRESS -----------

# def gausMix_ND_pdf(means, covariances, weights):
#     """A N dimensional Mixture of n Gaussians probability distribution function

#     Arguments:
#         means          means of Gaussians (n vector)
#         covariances    covariance structure of Gaussians (nx NxN array)
#         weights        weights of the n-gaussians (n vector)
#     """

#     ndim = len(means)

#     def pdf(*args): #only gives x
#         values = np.zeros_like(args[0])
        
#         for n, arg in enumerate(args):
#             print(n)
#             # would work in N dim
#             kernels = np.ones_like(args[0])
#             kernels *= np.exp(-((arg[n]-means)*LA.inv(covariances[n])*(arg[n]-means))/2) / np.sqrt((2*np.pi)**ndim*LA.det(covariances[n]))
                
#             values += kernels*weights[n]
#         return values
#     return pdf

