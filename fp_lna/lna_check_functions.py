"""
pre-defined convenience force functions
"""
import numpy as np
from numpy import linalg as LA
from scipy.stats.distributions import chi2

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline

def get_t_lambda(eigen_seq, tol = 1e-05):
    """
    Function to find the first time at which the eigenvalue increase
    can be consider linear in logscale.
    """
    grad_2 = np.gradient(np.gradient(np.log10(eigen_seq)))
    stable_derivative = np.where(grad_2<-tol)
    t_lambda = np.diff(stable_derivative).argmax()
    return t_lambda

def get_closest_t_to_saddle(x_t, xs):
    """
    Function to find the time at which LNA means is closest to saddle in L2 sense.
    """
    #how_many_cond = x_t.shape[0]
    #if how_many_cond == 1:
    how_far_saddle_all_time_bb = LA.norm(x_t - xs, axis = 1)
    min_pos = how_far_saddle_all_time_bb.argmin()
    #min_val = how_far_saddle_all_time_bb[min_pos]
    closest_to_saddle_data = min_pos
    # else:
    #     closest_to_saddle_data = np.zeros((how_many_cond), dtype = np.int8)
    #     for bb in range(how_many_cond):
    #         how_far_saddle_all_time_bb = LA.norm(x_t[bb,:, :] - xs, axis = 1)
    #         min_pos = how_far_saddle_all_time_bb.argmin()
    #         #min_val = how_far_saddle_all_time_bb[min_pos]
    #         closest_to_saddle_data[bb] = min_pos
    return closest_to_saddle_data

def return_ellipse_points(means, covariance, prob, thick = 50):
    tt=np.linspace(0, 2*np.pi, thick)
    x = np.cos(tt)
    y = np.sin(tt)
    ap = np.transpose(np.column_stack([x, y]))

    eig_val, eig_dir = LA.eig(covariance) 
    D = len(means)
    sdwidth = np.sqrt(chi2.ppf(prob, df=D))
    eig_val = sdwidth * np.sqrt(eig_val) # convert variance to sdwidth*sd

    bp = np.transpose(eig_dir @ (np.diag(eig_val) @ ap)) + means
    return bp

def intersection_check(ellipse_coords, m, q):
    is_interection = [True, True]
    
    for cc in range(ellipse_coords.shape[1]):
        if cc == 1:
            # obtaining the x points
            line_using = ellipse_coords[:,cc]*(1/m)+(-q/m)
        else:
            #obtaining the y points
            line_using = ellipse_coords[:,cc]*m+q
            
        intersection_line = ellipse_coords[:,(1-cc)] - line_using
        sum_True = np.sum(intersection_line>=0)
        # some are pos, some neg - interesction
        if (sum_True ==0 or sum_True==len(intersection_line)):
             is_interection[cc] = False
    if np.any(is_interection):
        return True
    else:
        return False  

# same function
def proj_onto_lineA_along_lineB(xy_coord_p, mA, qA, mB):
    # A is unstable
    #m2 = 1/s1
    #q2 = -s0/s1
    qB = xy_coord_p[1]- mB*xy_coord_p[0] 
    #  intersection
    intersection_x = (qA-qB)/(mB-mA)
    intersection_y = mB*intersection_x+qB

    return np.array([intersection_x, intersection_y])

#Projecting ellipse onto tangent
def proj_range_ellipse(the_ellipse_p, mA, qA, mB):
    proje = np.apply_along_axis(proj_onto_lineA_along_lineB, 1, the_ellipse_p, 
                    mA = mA, qA = qA, mB = mB)

    # check the slope of 1/s1 to decide which ranges (x, y) to take:
    condition_slope = (np.arctan(mA)<=np.pi/2) & (np.arctan(mA)>=0)
    if condition_slope:
        # consider y extremes
        ellipse_prog_ranges = [min(proje[:,1]), max(proje[:,1])]
    else:
        #consider the projection at the bottom: so x extremes
        ellipse_prog_ranges = [min(proje[:,0]), max(proje[:,0])]
    return ellipse_prog_ranges

def ellipse_on_S(the_ellipse_p, mA, qA, mB, 
direction_X_scaled, Sad, arclength_s, spline_tau_gives_s):
    ranges_on_y = proj_range_ellipse(the_ellipse_p, mA, qA, mB)
    # the y's
    v_vector = direction_X_scaled
    on_tau = (ranges_on_y-Sad[1])/v_vector[1]
    #spline from tau to s-arclength
    proj_onto_man = spline_tau_gives_s(on_tau)
    return proj_onto_man


def condition_on_s(the_ellipse_p, m_unstable, q_unstable, m_stable,
 direction_X_scaled, Sad, arclength_s, spline_tau_gives_s, Sa_e, Sb_e):
    # ideally you would just need the Sa_e values inside
    proj_onto_man = ellipse_on_S(the_ellipse_p, m_unstable, q_unstable, m_stable, 
                            direction_X_scaled, Sad, arclength_s, spline_tau_gives_s)
    # not of extremes but the attractors, they should coincide
    man_extreme = arclength_s[[0, -1]]
    off_1 = proj_onto_man>man_extreme
    
    off_1_both = not np.sum(off_1) ==1
    off_2 = proj_onto_man<man_extreme
    off_2_both = not np.sum(off_2)== 1
    off = off_1_both or off_2_both
    if off:
        return proj_onto_man, proj_onto_man[1]-proj_onto_man[0], False
    else:
        # you should access the value for the two extremes
        Var_lna_in_range = ((proj_onto_man[1]<=Sb_e) & (proj_onto_man[0]>=Sa_e))
        if Var_lna_in_range:
            return proj_onto_man, proj_onto_man[1]-proj_onto_man[0], True
        else:
            return proj_onto_man, proj_onto_man[1]-proj_onto_man[0], False
    # projecting manifold values, width and whether inrange
 


##### EXISTIG FUNCTIONS
# def force_from_data(grid, data):
#     """create a force function from data on a grid
    
#     Arguments:
#         grid     list of grid arrays along each dimension
#         data     force data (shape [ndim, ...])
#     """
#     grid = np.asarray(grid)
#     if grid.ndim == data.ndim == 1:
#         grid = (grid,)

#     f = RegularGridInterpolator(grid, np.moveaxis(data, 0, -1), bounds_error=False, fill_value=None)
#     def force(*args):
#         return np.moveaxis(f(args), -1, 0)

#     return force
