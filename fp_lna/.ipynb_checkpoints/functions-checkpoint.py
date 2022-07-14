"""
pre-defined convenience probability distribution functions
"""
import numpy as np
# import numba ?
from scipy import interpolate
from scipy import optimize
from numpy import linalg as LA
import matplotlib.pyplot as plt

import scipy

# for the EuMa sim
from scipy.stats import multivariate_normal
# for the LNA
from scipy.integrate import solve_ivp


#from fplanck_fmod.utility import value_to_vector
from scipy.interpolate import RegularGridInterpolator

# Function to check again
def eliminate_duplicates(Fixed_points_found):
	n = Fixed_points_found.shape[0]
	# already in order
	selected_fixed = np.array([True]*n)
	for ii in range(n-1):
		x_close = np.abs(Fixed_points_found[ii,0] - Fixed_points_found[ii+1,0])<= 1e-5
		y_close = np.abs(Fixed_points_found[ii,1] - Fixed_points_found[ii+1,1])<= 1e-5
		if x_close or y_close:
			selected_fixed[ii] = False
	return Fixed_points_found[selected_fixed,:]

#only not-numbized function
def ffix_solve(uvw, G_point):
    how_many_ic = 100;
    # random normal, wide variance
    ic_x = np.random.normal(-4, 4, how_many_ic)
    ic_y = np.random.normal(-4, 4, how_many_ic)
    ic_set = np.column_stack([ic_x, ic_y]);
    
    all_fp = []
    #a series of initial conditions
    ic = ic_set[0,:]
    new_fp = optimize.fsolve(G_point, ic, args = (uvw));
    if abs(new_fp[0])<1e-6:
        new_fp[0] = 0
    if abs(new_fp[1])<1e-6:
        new_fp[1] = 0
    all_fp.append(np.round(new_fp, 6))
    for ii in np.arange(1,how_many_ic):
        new_fp = optimize.fsolve(G_point, ic_set[ii,:], args = (uvw));
        if abs(new_fp[0])<1e-6:
            new_fp[0] = 0
        if abs(new_fp[1])<1e-6:
            new_fp[1] = 0
        #rounding
        new_fp = np.round(new_fp, 6)
        ff = 0
        already_found = [False]*len(all_fp)
        while (ff<len(all_fp)) & (not any(already_found)):
            if (all_fp[ff][0]==new_fp[0]) & (all_fp[ff][1]==new_fp[1]):
                already_found[ff] = True
            ff+=1
        if not any(already_found):
            all_fp.append(new_fp)
    
    # one last sweep to check
    ultimate_fp = []
    for ii in range(len(all_fp)):
        fp = all_fp[ii]
        check_fp = G_point(np.array(fp), uvw)
        if ((abs(check_fp[0])<=1e-5) & (abs(check_fp[1])<=1e-5)):
            ultimate_fp.append(fp)
    # sort according to 1st dim
    ultimate_fp = np.array(ultimate_fp)
    ultimate_fp = ultimate_fp[ultimate_fp[:, 0].argsort()]
    ultimate_fp = eliminate_duplicates(ultimate_fp)
    return ultimate_fp

def point_type(G, J, Fixed_Found, uvw):
    """
    % Written by Francesca Basini
    %
    % Returns the type of critical point (saddle, minimum, maximum)
    %
    % Usage:
    %  point_type_vec = point_type(params)
    %
    % params.G:            Gradient system dX/dt, function handle @(x, my_p)
    % params.J:            Jacobian matrix of system, function handle @(x, my_p)
    % params.Fixed_Found:   Fixed points found  
    % params.uvw:         (u, v, w) parameters of the system
    """
	
    how_many_fp = Fixed_Found.shape[0]
    point_type_vec = np.zeros(how_many_fp)
    for ll in range(how_many_fp):
        the_Jac = J(Fixed_Found[ll,:], uvw)
        eigenVal, eigVec = LA.eig(the_Jac)
        
        if (np.abs(eigenVal[0])<1e-4):
            eigenVal[0] = 0
        if (np.abs(eigenVal[1])<1e-4):
            eigenVal[1] = 0
        
        neg_lambda = np.sum(eigenVal<0)
        if (neg_lambda==2):
            #attractor, min
            point_type_vec[ll] = 0
        elif (neg_lambda==1): 
            #saddle
            point_type_vec[ll] = 1
        elif (neg_lambda==0):
            #maximum
            point_type_vec[ll] = 2
    return point_type_vec


def get_x_initial(xs,xa, xb, stable, J, uvw, s):
    # for the STABLE manifold, the starting point is the eigendirection
    if stable:
        # find eigenvalues and vectors
        lambdas, V = LA.eig(J(xs, uvw))
        #[V, D] = eig(params.J(the_saddle, params.PAR));

        neg_eigendir = V[:,lambdas<0]

        pos_eigendir = V[:,lambdas>0]
        #mult_lambda = linspace(-1,1, 100);
        #eigendirection_saddle = neg_eigendir.*mult_lambda;

        # in case eigendirection has inf m
        if (xs[0]==neg_eigendir[0]):
            # Simple vertical line
            x_init_eigen = np.column_stack([xs[0]*np.ones(len(s)),
                                            (xs[1]-neg_eigendir[1])*s+neg_eigendir[1] + 0.5*s*(1-s)])
        else:
            m_eigendir = (neg_eigendir[1]-xs[1])/(neg_eigendir[0]-xs[0])
            q_eigendir = xs[1]-m_eigendir*xs[0]

            # now build the line
            dxx = np.linspace(min([xs[0], neg_eigendir[0]])-1, max([xs[0], neg_eigendir[0]])+1, len(s))
            fxx = m_eigendir*dxx +q_eigendir
            x_init_eigen = np.column_stack([dxx, fxx])
        return x_init_eigen
    else: 
        Xstart = xa 
        Ystart = xb
        if (Xstart[0]==Ystart[0]):
            # Simple vertical line
            x_initial = np.column_stack([Ystart[0]*np.ones(len(s)),
                                         (Ystart[1]-Xstart[1])*s+Xstart[1] + 0.5*s*(1-s)])
        else:
            m = (Ystart[1] - Xstart[1])/(Ystart[0] - Xstart[0])
            q = Xstart[1] - m*(Xstart[0])
            dx = np.linspace(min([Xstart[0], Ystart[0]]), max([Xstart[0], Ystart[0]]), len(s));
            fx = m*dx +q;
            x_initial = np.column_stack([dx, fx])

        # We always set string in ascending order (no direction from xb to xa considered)
        if not (np.sum(np.round(Xstart, 4) == np.round(x_initial[0,:], 4))==2):
            x_initial = np.flip(x_initial)
        return x_initial

## TO FIX

# interp1 of matlab does it twice, for both columns of V!
def interp1_matlab(x, v, xq):
    interp_dim = v.shape[1]
    ynew = np.empty((xq.shape[0], interp_dim))
    for dd in range(interp_dim):
        # interpolate (x,y) and calculate in xq - query
        f = interpolate.interp1d(x, v[:,dd], kind = "cubic")
        yq = f(xq)
        ynew[:, dd] = yq
    return ynew

# Function to apply string method
# Function to apply string method
def string1_adapted(x_initial, G, uvw, forward, lims, tolerance, iterations):
    """
    Written by Tobias Grafke (grafke@cims.nyu.edu), based on the
    publication - Modified for stable manifold by Francesca Basini
    
    Usage:
    x = string1(x_initial, params)
    
    Returns the string x, which is the heteroclinic orbit connecting
    initial and final point through the saddle, which fulfills
    dx/dt||H_p(x,0). Here, params is a *dictionary* containing the
    fields
    
    params["H_p"]:        The p-derivative of the Hamiltonian as a
                        function handle of x and p
    params["iterations"]: (optional) Total number of iterations
    NO - params.plotevery:  (optional) Do a diagnostic plot every <plotevery>
                         iterations
    params["epsilon"]:    (optional) Relaxation step length
    params["xa","xb","xs"]: (optional) initial, final and saddle point
                         for diagnostic plotting purposes

    params["forward"]:    (optional) logical to integrate forward or backward
                         default: forward
    params["lims"]:       (optional) 2x2matrix, limits for string plotting
    
    params["tolerance"]:    tolerance level after which to stop iterating
    
    """
    
    # default params
    epsilon = 1e-2
	
    # start
    x = x_initial;
    # keep track of all of them
    #X_store = np.zeros((iterations, x_initial.shape[0], x_initial.shape[1]))
    X_store_prev = x
    
    # ends for trimming 
    end_lims = lims
    minValue_X = end_lims[0, 0]
    maxValue_X = end_lims[0, 1]

    minValue_Y = end_lims[1, 0]
    maxValue_Y = end_lims[1, 1]

    xa = x[0,:]
    xb = x[-1,:]
	
    # tuple
    Nt, Nx = x.shape
    s = np.linspace(0,1,Nt)
    ds = s[1]-s[0]

    # reparametrize to standard arclength
    alpha = np.hstack([0, np.cumsum(np.sqrt(np.sum((x[1:,:]-x[0:-1,:])**2, 1)))])
    alpha = alpha/alpha[-1]
    x = interp1_matlab(alpha, x, s)
    #x = interp1(alpha, x, s, 'spline');
    
    # counter
    i = 0
    more_than_tol = True
    if (not forward):
        # integrate backwards
        while (i<iterations) and more_than_tol:
            #x = x - epsilon*H_p(x,0*x, params["PAR"]);
            x = x - epsilon*G(x, uvw);
            # trim X
            indexesInRange_X = (x[:,0] > minValue_X) & (x[:,0] < maxValue_X)
            indexesInRange_Y = (x[:,1] > minValue_Y) & (x[:,1] < maxValue_Y)
            indexesInRange_both = (indexesInRange_X & indexesInRange_Y)
            
            trim_x = x[indexesInRange_both, :]
            
            # not need to set endpoints to fixed
            alpha = np.hstack([0, np.cumsum(np.sqrt(np.sum((trim_x[1:,:]-trim_x[0:-1,:])**2, 1)))])
            alpha = alpha/alpha[-1]
            x = interp1_matlab(alpha, trim_x, s)
            # store it
            #X_store[i,:,:] = x
            
            #computing the average distance
            discrepancy = np.mean(LA.norm(X_store_prev-x, axis = 1))
            more_than_tol = discrepancy>tolerance 
            X_store_prev = x
            i = i+1
            
    else:
        # integrate forward
        while (i<iterations) and more_than_tol:
            #x = x + epsilon*H_p(x,0*x, params["PAR"])
            x = x + epsilon*G(x, uvw)
            
            # reset initial and final points to allow for string computation
            # between points that are not stable fixed points
            x[0,:] = xa
            x[-1,:] = xb
            # reparametrize to standard arclength
            alpha = np.hstack([0, np.cumsum(np.sqrt(np.sum((x[1:,:]-x[0:-1,:])**2, 1)))])
            alpha = alpha/alpha[-1]
            x = interp1_matlab(alpha, x, s)
            # store it
            #X_store[i,:,:] = x
            
             #computing the average distance
            discrepancy = np.mean(LA.norm(X_store_prev-x, axis = 1))
            more_than_tol = discrepancy>tolerance
            X_store_prev = x
            i= i+1
    return x


######## FUNCTIONS EXSISTING

def delta_function(r0):
    """a discrete equivalent of a dirac-delta function centered at r0"""
    r0 = np.atleast_1d(r0)

    def pdf(*args):
        values = np.zeros_like(args[0])

        diff = sum([(r0[i] - args[i])**2 for i in range(len(args))])
        idx = np.unravel_index(np.argmin(diff), diff.shape)
        values[idx] = 1

        return values
        
    return pdf

def gaussian_pdf(center, width):
    """A Gaussian probability distribution function
    -always assuming independence between n dims-
    
    Arguments:
        center    center of Gaussian (scalar or vector)
        width     width of Gaussian (scalar or vector)
    """

    center = np.atleast_1d(center)
    ndim = len(center)
    width = value_to_vector(width, ndim)

    def pdf(*args):
        values = np.ones_like(args[0])

        for i, arg in enumerate(args):
            values *= np.exp(-np.square((arg - center[i])/width[i]))

        return values/np.sum(values)

    return pdf

def uniform_pdf(func=None):
    """A uniform probability distribution function
    
    Arguments:
        func    a boolean function specifying region of uniform probability (default: everywhere)
    """

    def pdf(*args):
        if func is None:
            values = np.ones_like(args[0])
        else:
            values = np.zeros_like(args[0])
            idx = func(*args)
            values[idx] = 1

        return values/np.sum(values)

    return pdf



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

