import numpy as np
# import numba ?
from scipy import interpolate
from scipy import optimize
from numpy import linalg as LA
import matplotlib.pyplot as plt

#import scipy

# for the EuMa sim
#from scipy.stats import multivariate_normal
# for the LNA
#from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline


from fp_lna.functions import ffix_solve, point_type, get_x_initial, string1_adapted


class landscape:
    def __init__(self, potential_V, Gradient_G, G_point, Jacobian_J, uvw, lims, 
                 tolerance = 1e-5, iterations = 400):
        """
        Solve the Fokker-Planck equation
        Arguments:
            potential
            uvw             parameters needed for the distribution
            
        Optional pars:
            tolerance
            iterations
            (epsilon)
        """
        
        self.V = potential_V
        self.G = Gradient_G
        self.G_point = G_point
        self.J = Jacobian_J
        self.uvw = uvw
        self.lims = lims
        self.tolerance = tolerance
        self.iterations = iterations
            
		# automatically returns fixed points
        fixed_points = ffix_solve(self.uvw, self.G_point)
        self.fixed_points = fixed_points
        saddle_attractors = point_type(self.G, self.J, fixed_points, self.uvw)
        
        the_saddle = fixed_points[saddle_attractors==1,:][0]
        self.xs = the_saddle
        attractors = fixed_points[saddle_attractors==0,:]
        self.xa = attractors[0,:]
        self.xb = attractors[1,:] 
		
		# obtaining manifolds
        stable_unstable = np.array([False, True])
        Nt = 2**6
        s = np.linspace(0,1,Nt)
        
        for stable in stable_unstable:
            if stable:
                # if we want the stable, we set integration to backward
                forward = False
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
				# x_stable = x_stable[x_stable[:, 0].argsort()]
                self.x_stable = manifold
            else:
                forward = True
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
                # sort both in ascending order wrt y
                manifold = manifold[manifold[:, 1].argsort()]
                self.x_unstable = manifold
		
		# arclength par
        alpha_length = np.hstack([0,np.cumsum(LA.norm(self.x_unstable[:-1,:] - self.x_unstable[1:,:], axis = 1))])
        # here you flip it, to get a dicent parm?
        spline_s_given_y_arcl = UnivariateSpline(self.x_unstable[:,1], alpha_length, k=3, s=0)
        #spline_t_given_x_t = UnivariateSpline(x_unstable[:,0], alpha_t_arc, k=3, s=0)

        # Recentering the arclength parameterization at the saddle
        sad_arclength = spline_s_given_y_arcl(self.xs[1])
        arclength_s = alpha_length - sad_arclength
        self.arclength_s = arclength_s 
        # New Splines:
        spline_x_s_given_s = UnivariateSpline(arclength_s, self.x_unstable[:,0], k=3, s=0)
        spline_y_s_given_s = UnivariateSpline(arclength_s, self.x_unstable[:,1], k=3, s=0)
        spline_s_given_y_s = UnivariateSpline(self.x_unstable[:,1], arclength_s, k=3, s=0)
        
        self.spline_x_s_given_s = spline_x_s_given_s
        self.spline_y_s_given_s = spline_y_s_given_s
        self.spline_s_given_y_s = spline_s_given_y_s
        
        # extremes of arclength
        Sa, Sb = arclength_s[[0, -1]]
        self.Sa = Sa
        self.Sb = Sb
        
    @staticmethod
    def _d_C_dV_(t, z, Big_Sigma, uvw, G_point, J):
        #z contains the:
        # x and y coord, 
        # the 4 entries of matrix V
        xx, yy = z[0:2]
        C = z[2:4]
        VV = z[4:].reshape(2, 2)

        # doing the F - WRONG!
        dxdy = G_point(np.array([xx, yy]), uvw)

        # doing the Jacobian
        current_J = J(np.array([xx, yy]), uvw)
        dC = current_J @ C

        dV = current_J @ VV + VV @ np.transpose(current_J) + Big_Sigma 
        dv = dV.flatten()

        Syst = np.concatenate((dxdy, dC, dv))
        return Syst
    
     
    def get_LNA(self, m_0, V_t0, Big_Sigma, MaxTime, dt):    
    timespan = np.arange(0, MaxTime+dt, dt)
    
    # m_0 distances are not necessary here
    # are there any other ways to give a default?
    starting_points = len(m_0.shape)
    if starting_points == 1:
        start_lna_datapoints = m_0[1:]
        my_distances = m_0[0]
        how_many_blobs = 1
    else:
        start_lna_datapoints = m_0[:,1:]
        my_distances = m_0[:,0]
        how_many_blobs = len(my_distances)
    
    
    v0 = V_t0.flatten()
    tt_max = len(timespan)-1
    
    x_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
    C_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
    V_sol_all_ic = np.zeros((how_many_blobs, tt_max, 4))
    T_sol_all_ic = np.zeros((how_many_blobs, tt_max, 1))

    initial_blob = np.zeros((how_many_blobs, 2))
    final_blob = np.zeros((how_many_blobs, 2))
    
    if how_many_blobs>1:
        # how many initial conditions
        for current_ic in range(how_many_blobs):
            # no perturbation
            yx_0 = start_lna_datapoints[current_ic,:]
            # original dispacement
            C_0 = yx_0 - yx_0 
            y0 = np.hstack([yx_0, C_0, v0])

            sol = solve_ivp(self.d_C_dV, t_span = (0, MaxTime), t_eval = timespan,
                            y0 = y0, args = (self.uvw, Big_Sigma, self.G_point, self.J))
            YY = sol.y
            TT = sol.t

            # exclude inital time 0
            x_sol= YY[0:2, 1:]
            C_sol= YY[2:4, 1:]
            V_sol = YY[4:, 1:]

            # save ic and final
            #initial_blob[current_ic,:] = yx_0
            #final_blob_current = C_sol[-1, :]
            #final_blob[current_ic,:] = final_blob_current

            # save whole trajectory in cell arrays
            x_sol_all_ic[current_ic, :, :] = np.transpose(x_sol)
            C_sol_all_ic[current_ic, :, :] = np.transpose(C_sol)
            V_sol_all_ic[current_ic, :, :] = np.transpose(V_sol)
            T_sol_all_ic[current_ic, :, :] = TT[1:].reshape((tt_max, 1))
    else:
        # no perturbation
        yx_0 = start_lna_datapoints
        # original dispacement
        C_0 = yx_0 - yx_0 
        y0 = np.hstack([yx_0, C_0, v0])

        sol = solve_ivp(self.d_C_dV, t_span = (0, MaxTime), t_eval = timespan,
                        y0 = y0, args = (self.uvw, Big_Sigma, self.G_point, self.J))
        YY = sol.y
        TT = sol.t

        # exclude inital time 0
        x_sol= YY[0:2, 1:]
        C_sol= YY[2:4, 1:]
        V_sol = YY[4:, 1:]

        # save ic and final
        #initial_blob[current_ic,:] = yx_0
        #final_blob_current = C_sol[-1, :]
        #final_blob[current_ic,:] = final_blob_current

        # save whole trajectory in cell arrays
        x_sol_all_ic[0, :, :] = np.transpose(x_sol)
        C_sol_all_ic[0, :, :] = np.transpose(C_sol)
        V_sol_all_ic[0, :, :] = np.transpose(V_sol)
        T_sol_all_ic[0, :, :] = TT[1:].reshape((tt_max, 1))
   
    LNA_res = {
		"x_t": x_sol_all_ic,
        "C_t": C_sol_all_ic,
        "V_t": V_sol_all_ic,
        "t": T_sol_all_ic}
    return LNA_res 
    
		
# 	def propagate_interval(self, initial, tf, Nsteps=None, dt=None, normalize=True):
#         """Propagate an initial probability distribution over a time interval, return time and the probability distribution at each time-step
#         Arguments:
#             initial      initial probability density function
#             tf           stop time (inclusive)
#             Nsteps       number of time-steps (specifiy Nsteps or dt)
#             dt           length of time-steps (specifiy Nsteps or dt)
#             normalize    if True, normalize the initial probability
#         """
#         p0 = initial(*self.grid)
#         if normalize:
#             p0 /= np.sum(p0)

#         if Nsteps is not None:
#             dt = tf/Nsteps
#         elif dt is not None:
#             Nsteps = np.ceil(tf/dt).astype(int)
#         else:
#             raise ValueError('specifiy either Nsteps or Nsteps')

#         time = np.linspace(0, tf, Nsteps)
#         pf = expm_multiply(self.master_matrix, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True)
#         return time, pf.reshape((pf.shape[0],) + tuple(self.Ngrid))

#     def probability_current(self, pdf):
#         """Obtain the probability current of the given probability distribution"""
#         J = np.zeros_like(self.force_values)
#         for i in range(self.ndim):
#             J[i] = -(self.diffusion[i]*np.gradient(pdf, self.resolution[i], axis=i) 
#                   - self.mobility[i]*self.force_values[i]*pdf)

#         return J