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
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from fp_lna.lna_check_functions import *
from fp_lna.functions import *

class landscape:
    def __init__(self, potential_V, Gradient_G, G_point, Jacobian_J, G_point_lna, J_lna, uvw, lims, 
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
        self.G_point_lna = G_point_lna
        self.J_lna = J_lna
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
        
        vec_one_min_one = np.array([-1, 1])
        for stable in stable_unstable:
            if stable:
                # if we want the stable, we set integration to backward
                forward = False
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
				# x_stable = x_stable[x_stable[:, 0].argsort()]
                self.x_stable = manifold
                # assuming that it's a spline from x to y
                spline_stable = UnivariateSpline(manifold[:,0], manifold[:,1], k=3, s=0)
                p_der_stab = spline_stable.derivative()
                y_der_saddle_stable = p_der_stab(self.xs[0])
                self.m_stable = y_der_saddle_stable
                self.q_stable = self.xs[1] - self.xs[0]*y_der_saddle_stable

                # vector P
                tangent_stable = vec_one_min_one*self.m_stable+self.q_stable
                tang_stab_coord = np.column_stack([vec_one_min_one, tangent_stable])
                # directional vector P
                direction_P = tang_stab_coord[1, :] - tang_stab_coord[0,:]
                self.direction_P_scaled = direction_P/LA.norm(direction_P) 
            else:
                # doing the unstable
                forward = True
                x_initial = get_x_initial(self.xs, self.xa, self.xb, stable, self.J, self.uvw, s)
                manifold = string1_adapted(x_initial, self.G, self.uvw, forward, 
                                           self.lims, self.tolerance, self.iterations)
                # sort both in ascending order wrt y
                manifold = manifold[manifold[:, 1].argsort()]
                self.x_unstable = manifold
                # know we regress on y to x
                spline_unstable = UnivariateSpline(manifold[:,1], manifold[:,0], k=3, s=0)
                p_der = spline_unstable.derivative()
                y_der_saddle=p_der(self.xs[1])
                s1_unstable = y_der_saddle
                self.m_unstable = 1/s1_unstable
                s0_unstable = self.xs[0] - self.xs[1]*s1_unstable
                self.q_unstable = -s0_unstable/s1_unstable

                # direction X and Y
                tangent_unstable = vec_one_min_one*s1_unstable+s0_unstable
                tang_unst_coord = np.column_stack([tangent_unstable, vec_one_min_one])
                direction_X = tang_unst_coord[1,:] - tang_unst_coord[0,:]
                self.direction_X_scaled= direction_X/LA.norm(direction_X) 
                self.direction_Y_scaled = np.array([-self.direction_X_scaled[1], self.direction_X_scaled[0]])
		
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
        
        # more on the mapping geometry
        self.spline_tau_gives_s = get_mapping_from_tau_Tos(self.m_unstable, self.x_unstable, self.xs, self.direction_X_scaled, self.arclength_s)
        
    @staticmethod
    def _d_C_dV_(t, z, Big_Sigma, uvw, G_point_lna, J_lna):
        #z contains the:
        # x and y coord, 
        # the 4 entries of matrix V
        xx, yy = z[0:2]
        C = z[2:4]
        VV = z[4:].reshape(2, 2)

        # doing the F - WRONG!
        dxdy = G_point_lna(xx, yy, uvw)

        # doing the Jacobian
        current_J = J_lna(xx, yy, uvw)
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

                sol = solve_ivp(self._d_C_dV_, t_span = (0, MaxTime), t_eval = timespan,
                                y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
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

            eigen_V_t_all_ic = get_eigen_evolution(V_sol_all_ic)
        else:
            # no perturbation
            yx_0 = start_lna_datapoints
            # original dispacement
            C_0 = yx_0 - yx_0 
            y0 = np.hstack([yx_0, C_0, v0])

            sol = solve_ivp(self._d_C_dV_, t_span = (0, MaxTime), t_eval = timespan,
                            y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
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

            eigen_V_t_all_ic = get_eigen_evolution(V_sol_all_ic)

        LNA_res = {
            "x_t": x_sol_all_ic,
            "C_t": C_sol_all_ic,
            "V_t": V_sol_all_ic,
            "t": T_sol_all_ic,
            "eigen_V_t": eigen_V_t_all_ic}
        return LNA_res 
    
    def LNA_check(self, m_0, V_t0, Big_Sigma, MaxTime, dt, containment, prob = 0.9, thick = 50):
        # JUST RETURN THE LNA CHECK and TIME at which it passes - "text"
        LNA_res = self.get_LNA(m_0, V_t0, Big_Sigma, MaxTime, dt)
        x_t = LNA_res["x_t"]
        C_t = LNA_res["C_t"]
        V_t = LNA_res["V_t"]
        LNA_t = LNA_res["t"]
        eigen_V_t= LNA_res["eigen_V_t"]

        # defining thresholds
        self.Sa_e = self.Sa*containment/100
        self.Sb_e = self.Sb*containment/100


        # closest point to saddle
        close_times = closest_to_saddle(x_t, self.xs)
        how_many_ic = len(m_0.shape)
        if how_many_ic == 1:
            selected_mean = x_t[0,close_times, :].reshape(1, 2)[0]
            selected_cov = V_t[0,close_times, :].reshape(2, 2)
            ellipe_points_all_ic = return_ellipse_points(selected_mean, selected_cov, prob, thick)
            
            # check intersection with stable
            intersect_stable = intersection_check(ellipe_points_all_ic, self.m_stable, self.q_stable)
            if not intersect_stable:
                # fancier writing
                print("LNA Gaussian is not intersecting tangent to stable manifold at its closest to the saddle.")
                print("Returning density at given T")
                # FP not required, return density at a given T
                return intersect_stable, selected_mean, selected_cov
            # CHECK INTERSECTION WITH UNSTABLE
            else: 
                print("LNA density is intersecting the stable manifold. FP-LNA required.")
                print("Returning density with first intersection with unstable manifold.")
                # restart from time 0
                unstable_cross_pos = 0
                not_in_range = True
                while (not_in_range & (unstable_cross_pos<=close_times)):
                    selected_mean_cross = x_t[0,unstable_cross_pos, :].reshape(1, 2)[0]
                    selected_cov_cross = V_t[0,unstable_cross_pos, :].reshape(2, 2)
                    ellipe_points_all_ic = return_ellipse_points(selected_mean_cross, selected_cov_cross, prob, thick)

                    intersect_unstable = intersection_check(ellipe_points_all_ic, self.m_unstable, self.q_unstable)
                    if not intersect_unstable:
                        unstable_cross_pos = unstable_cross_pos+1
                    else:
                        # it crossed, calculate how much
                        projection_vals, proj_range, in_range = condition_on_s(ellipe_points_all_ic, self.m_unstable, self.q_unstable, self.m_stable,
                        self.direction_X_scaled, self.xs, self.arclength_s, self.spline_tau_gives_s, self.Sa_e, self.Sb_e)
                        if in_range:
                            unstable_cross_found = unstable_cross_pos
                            if unstable_cross_pos<=close_times:
                                unstable_cross_pos = unstable_cross_pos+1
                        else:
                            not_in_range = False
                            unstable_cross_pos = np.inf

                selected_mean_cross_found = x_t[0,unstable_cross_found, :].reshape(1, 2)[0]
                selected_cov_cross_found = V_t[0,unstable_cross_found, :].reshape(2, 2)
                print(unstable_cross_found)
                return intersect_stable, selected_mean_cross_found, selected_cov_cross_found

            

        # # no time to do the N-dimensional    
        # else:
        #     how_many_blobs = m_0.shape[0]
        #     ellipe_points_all_ic = np.zeros((how_many_blobs, thick, 2))
        #     for ii in range(how_many_blobs):
        #         selected_mean = x_t[ii,close_times[ii], :].reshape(1, 2)[0]
        #         selected_cov = V_t[ii,close_times[ii], :].reshape(2, 2)
        #         ellipe_points = return_ellipse_points(selected_mean, selected_cov, prob, thick)
        #         ellipe_points_all_ic[ii, :, :] = ellipe_points
        
        # ellipe_points_all_ic

    # def FP_setup(self, ): #takes LNA inputs
    #     """ Should return the setup for the FP solver:  
    #     arclength range s,
    #     initial distribution - p0
    #     spline for drift - s_dot
    #     spline for diffusion - sigma(s)
    #     effective (remaining) - T
    #     """

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