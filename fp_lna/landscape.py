import numpy as np
# import numba ?
from scipy import interpolate
from scipy import optimize

from scipy.stats import multivariate_normal

from numpy import linalg as LA
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#import scipy

# for the EuMa sim
#from scipy.stats import multivariate_normal

# for the LNA
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

from fp_lna.functions import *
from fp_lna.lna_check_functions import *
from fp_lna.fokker_planck_solver import *
from fp_lna.loglik_reconstruction import *

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
                # sort by x to get a spline x to y
                x_stable_sorted_X = manifold[np.argsort(manifold[:,0]), :]
                spline_stable = UnivariateSpline(x_stable_sorted_X[:,0], x_stable_sorted_X[:,1], k=3, s=0)
                self.spline_stable = spline_stable
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
                manifold = manifold[manifold[:, 1].argsort()] # to leave?
                self.x_unstable = manifold
                # sort by Y to get a spline y to x
                #x_unstable_sorted_X = manifold[np.argsort(manifold[:,0]), :]
                #spline_unstable = UnivariateSpline(x_unstable_sorted_X[:,0], x_unstable_sorted_X[:,1], k=3, s=0)
                spline_unstable = UnivariateSpline(manifold[:,1], manifold[:,0], k=3, s=0)
                self.spline_unstable = spline_unstable
                p_der = spline_unstable.derivative()
                y_der_saddle=p_der(self.xs[1])
                s1_unstable = y_der_saddle
                self.m_unstable = 1/s1_unstable
                s0_unstable = self.xs[0] - self.xs[1]*s1_unstable
                self.q_unstable = -s0_unstable/s1_unstable

                # y_der_saddle=p_der(self.xs[0])
                # s1_unstable = y_der_saddle
                # self.m_unstable = s1_unstable
                # self.q_unstable = self.xs[1] - self.xs[0]*y_der_saddle


                # direction X and Y
                # tangent_unstable = vec_one_min_one*s1_unstable+s0_unstable
                # tang_unst_coord = np.column_stack([tangent_unstable, vec_one_min_one])
                # direction_X = tang_unst_coord[1,:] - tang_unst_coord[0,:]
                tangent_unstable = vec_one_min_one*self.m_unstable+self.q_unstable
                tang_unst_coord = np.column_stack([vec_one_min_one, tangent_unstable])
                # directional vector P
                direction_X = tang_unst_coord[1, :] - tang_unst_coord[0,:]
                self.direction_X_scaled= direction_X/LA.norm(direction_X) 
                self.direction_Y_scaled = np.array([-self.direction_X_scaled[1], self.direction_X_scaled[0]])
		
		# arclength par
        alpha_length = np.hstack([0,np.cumsum(LA.norm(self.x_unstable[:-1,:] - self.x_unstable[1:,:], axis = 1))])
        # here you flip it, to get a dicent parm?
        x_unstable_sorted_x = self.x_unstable[self.x_unstable[:, 0].argsort()]
        spline_t_given_x_t = UnivariateSpline(x_unstable_sorted_x[:,0], alpha_length, k=3, s=0)
        # sorted_y
        x_unstable_sorted_Y = self.x_unstable[self.x_unstable[:, 1].argsort()]
        spline_s_given_y_arcl = UnivariateSpline(x_unstable_sorted_Y[:,1], alpha_length, k=3, s=0)

        # Recentering the arclength parameterization at the saddle
        sad_arclength = spline_s_given_y_arcl(self.xs[1])
        arclength_s = alpha_length - sad_arclength
        self.arclength_s = arclength_s 
        # New Splines:
        spline_x_s_given_s = UnivariateSpline(arclength_s, self.x_unstable[:,0], k=3, s=0)
        # 21/08 might not be the same thing?
        spline_y_s_given_s = UnivariateSpline(arclength_s, self.x_unstable[:,1], k=3, s=0)
        # modified below
        #spline_s_given_y_s = UnivariateSpline(self.x_unstable[:,1], arclength_s, k=3, s=0)
        spline_s_given_y_s = UnivariateSpline(x_unstable_sorted_Y[:,1], arclength_s, k=3, s=0)
        #spline_s_given_x_s = UnivariateSpline(x_unstable_sorted_X[:,0], arclength_s, k=3, s=0)
        
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
    
     
    # function that iterates the LNA until fixed point is reached
    def get_t_s(self, m_0, V_t0, Big_Sigma, dt, MaxTime):    
        #timespan = np.arange(0, MaxTime+dt, dt)
        
        # m_0 distances are not necessary here
        # are there any other ways to give a default?
        #starting_points = len(m_0.shape)
        start_lna_datapoints = m_0[1:]
        #my_distances = m_0[0]
        #how_many_blobs = 1
        # else:
        #     start_lna_datapoints = m_0[:,1:]
        #     my_distances = m_0[:,0]
        #     how_many_blobs = len(my_distances)

        v0 = V_t0.flatten()
        #tt_max = len(timespan)-1

        #t_s_all_ic = np.zeros(how_many_blobs)
        
        # if how_many_blobs>1:
        #     # how many initial conditions
        #     for current_ic in range(how_many_blobs):
        #         # no perturbation
        #         yx_0 = start_lna_datapoints[current_ic,:]
        #         # original dispacement
        #         C_0 = yx_0 - yx_0 
        #         y0 = np.hstack([yx_0, C_0, v0])

        #         sol = solve_ivp(self._d_C_dV_, t_span = (0, MaxTime), 
        #                         #t_eval = timespan,
        #                         method = "Radau",
        #                         y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
        #         # just save the rounded up t_s
        #         t_s_blob = np.round(sol.t[-2], 2)

        #         # save whole trajectory in cell arrays
        #         t_s_all_ic[current_ic] = t_s_blob
        # else:
        
        yx_0 = start_lna_datapoints
        # original dispacement
        C_0 = yx_0 - yx_0 
        y0 = np.hstack([yx_0, C_0, v0])

        sol = solve_ivp(self._d_C_dV_, t_span = (0, MaxTime), 
                        #t_eval = timespan,
                        method = "Radau",
                        y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
        YY_radau = sol.y
        # check whether indeed ts and t are the same
        if MaxTime - sol.t[-2] <= dt:
            t_s_blob = MaxTime
        else:
            t_s_blob = np.round(sol.t[-2], 2)
        return t_s_blob, YY_radau[:,-1] #return final solution
    
    def get_LNA(self, m_0, V_t0, Big_Sigma, observed_t, dt, MaxTime = 15):  
        
        t_s = self.get_t_s(m_0, V_t0, Big_Sigma, dt, max(MaxTime, observed_t))[0]
        # right or not?
        max_t_ts = max(observed_t, t_s)

        # m_0 distances are not necessary here
        # are there any other ways to give a default?
        #starting_points = len(m_0.shape)
        #if starting_points == 1:
        start_lna_datapoints = m_0[1:]
        #my_distances = m_0[0]
        #how_many_blobs = 1
        # else:
        #     start_lna_datapoints = m_0[:,1:]
        #     my_distances = m_0[:,0]
        #     how_many_blobs = len(my_distances)
        
        v0 = V_t0.flatten()
        timespan = np.arange(0, max_t_ts+dt, dt)
        tt_max = len(timespan)-1
        
        # x_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
        # C_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
        # V_sol_all_ic = np.zeros((how_many_blobs, tt_max, 4))
        # T_sol_all_ic = np.zeros((how_many_blobs, tt_max, 1))

        # x_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
        # C_sol_all_ic = np.zeros((how_many_blobs, tt_max, 2))
        # V_sol_all_ic = np.zeros((how_many_blobs, tt_max, 4))
        # T_sol_all_ic = np.zeros((how_many_blobs, tt_max, 1))

        # if how_many_blobs>1:
        #     # how many initial conditions
        #     for current_ic in range(how_many_blobs):
        #         # no perturbation
        #         yx_0 = start_lna_datapoints[current_ic,:]
        #         # original dispacement
        #         C_0 = yx_0 - yx_0 
        #         y0 = np.hstack([yx_0, C_0, v0])

        #         sol = solve_ivp(self._d_C_dV_, t_span = (0, MaxTime), t_eval = timespan,
        #                         y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
        #         YY = sol.y
        #         TT = sol.t

        #         # exclude inital time 0
        #         x_sol= YY[0:2, 1:]
        #         C_sol= YY[2:4, 1:]
        #         V_sol = YY[4:, 1:]

        #         # save whole trajectory in cell arrays
        #         x_sol_all_ic[current_ic, :, :] = np.transpose(x_sol)
        #         C_sol_all_ic[current_ic, :, :] = np.transpose(C_sol)
        #         V_sol_all_ic[current_ic, :, :] = np.transpose(V_sol)
        #         T_sol_all_ic[current_ic, :, :] = TT[1:].reshape((tt_max, 1))

        #     eigen_V_t_all_ic, eigen_DIR_V_t = get_eigen_evolution(V_sol_all_ic)
        # else:
            
        yx_0 = start_lna_datapoints
        # original dispacement
        C_0 = yx_0 - yx_0 
        y0 = np.hstack([yx_0, C_0, v0])

        sol = solve_ivp(self._d_C_dV_, t_span = (0, max_t_ts), t_eval = timespan,
                        y0 = y0, args = (Big_Sigma, self.uvw, self.G_point_lna, self.J_lna))
        YY = sol.y
        TT = sol.t

        # exclude inital time 0
        x_sol= np.transpose(YY[0:2, :])
        C_sol= np.transpose(YY[2:4, :])
        V_sol = np.transpose(YY[4:, :])

        # # save whole trajectory in cell arrays
        # x_sol_all_ic[0, :, :] = np.transpose(x_sol)
        # C_sol_all_ic[0, :, :] = np.transpose(C_sol)
        # V_sol_all_ic[0, :, :] = np.transpose(V_sol)
        # T_sol_all_ic[0, :, :] = TT[1:].reshape((tt_max, 1))

        eigen_V_t, eigen_DIR_V_t = get_eigen_evolution(V_sol)

        LNA_res = {
            "x_t": x_sol,
            "C_t": C_sol,
            "V_t": V_sol,
            "t": TT,
            "eigen_V_t": eigen_V_t,
            "eigendir": eigen_DIR_V_t}
        return LNA_res 
    
    def LNA_check(self, m_0, V_t0, Big_Sigma, observed_t, dt, MaxTime = 15, containment= 0.9, prob = 0.9, thick = 50):
        # JUST RETURN THE LNA CHECK and TIME at which it passes - "text"
        LNA_res = self.get_LNA(m_0, V_t0, Big_Sigma, observed_t, dt)
        x_t = LNA_res["x_t"].squeeze()
        #C_t = LNA_res["C_t"].squeeze()
        V_t = LNA_res["V_t"].squeeze()
        LNA_t = LNA_res["t"].squeeze()
        eigen_V_t= LNA_res["eigen_V_t"].squeeze()
        eigen_DIR_V_t = LNA_res["eigendir"].squeeze()

        # identify t_lambda
        t_lambda = get_t_lambda(eigen_V_t[:,1])
        # identify t_close
        t_close = get_closest_t_to_saddle(x_t, self.xs)
        
        # closest point to saddle
        print("Print t_close: {}".format(t_close*dt))
        print("Print t_lambda: {}".format(t_lambda*dt))

        ###########################################
        # # check intersection with stable manifold
        ###########################################

        intersect_stable_count = 0
        
        if t_lambda == t_close:
            selected_mean = x_t[t_close, :].reshape(1, 2)[0]
            selected_cov = V_t[t_close, :].reshape(2, 2)
            ellipe_points_all_ic = return_ellipse_points(selected_mean, selected_cov, prob, thick)
            intersect_stable_count = intersection_check(ellipe_points_all_ic, self.m_stable, self.q_stable)
        else:
            tt = min(t_close, t_lambda)
            max_t_close_t_lambda = max(t_close, t_lambda)
            while tt<=max_t_close_t_lambda:
                selected_mean = x_t[tt, :].reshape(1, 2)[0]
                selected_cov = V_t[tt, :].reshape(2, 2)
                ellipe_points_all_ic = return_ellipse_points(selected_mean, selected_cov, prob, thick)
                intersect_stable = intersection_check(ellipe_points_all_ic, self.m_stable, self.q_stable)
                intersect_stable_count += intersect_stable
                tt +=1
        # intersect_stable_count should be at least 1
        if intersect_stable_count<1:
            # fancier writing
            print("Case 0: LNA Gaussian is not intersecting tangent to stable manifold.")
            print("FP not required, returning density at given T")
            # FP not required, return density at a given T
            
            # fpLNA necessary? False
            return False, x_t[int(observed_t/dt), :].reshape(1, 2), V_t[int(observed_t/dt), :].reshape(2, 2)
        
        else:
            # Don-t need intersection WITH UNSTABLE
            print("Case 1: LNA density is intersecting the stable manifold. FP-LNA required.")
            print("Returning density at critical time t* based on stopping rule.")
            
            # defining thresholds
            self.Sa_e = self.Sa*containment/100
            self.Sb_e = self.Sb*containment/100
         
             # start from time 0 to max(t_close, t_lambda)
             # let's try withouth a maximum

            # NOT STORING projections on s
            #proj_store = np.zeros((int(), 2))
            # Could do an append
            not_in_range = True
            tt = max_t_close_t_lambda
            while (not_in_range & (tt>=0)):
            #while not_in_range:
                selected_mean_stop_find = x_t[tt, :].reshape(1, 2)[0]
                selected_cov_stop_find = V_t[tt, :].reshape(2, 2)
                ellipe_points_all_ic = return_ellipse_points(selected_mean_stop_find, selected_cov_stop_find, prob, thick)

                # who cares about unstable
                # intersect_unstable = intersection_check(ellipe_points_all_ic, self.m_unstable, self.q_unstable)
                # if not intersect_unstable:
                #     unstable_cross_pos = unstable_cross_pos+1
                # else:
                    
                projection_vals, proj_width, in_range = condition_on_s(ellipe_points_all_ic, self.m_unstable, self.q_unstable, self.m_stable,
                    self.direction_X_scaled, self.xs, self.arclength_s, self.spline_tau_gives_s, self.Sa_e, self.Sb_e)
                if in_range:
                    # enough to stop! because we are going bacwards
                    not_in_range = False
                else:
                    tt -=1
            # found critical time of stopping rule
            t_star = np.copy(tt)
            print("t_star: {}".format(t_star*dt))
            # fpLNA necessary? True
            return True, selected_mean_stop_find, selected_cov_stop_find, t_star, LNA_res #return also LNA results

    # function that from mean-variance return initialisation of FP
    def initialise_FP(self, xt_critical, Vt_critical, Big_Sigma, range_fp = 400):
        integrated_g, spline_projection = IntegrateOverP(xt_critical, Vt_critical, self.direction_P_scaled, self.xs, print_ = False)
        p0_s_vals, spline_p0s_given_s, s_tau_gives_s =  get_p0_s(1/self.m_unstable, -1/self.q_unstable/self.m_unstable,
                                                        self.m_stable, self.x_unstable,
                                                        self.xs, xt_critical, self.direction_X_scaled, 
                                                        integrated_g, spline_projection, self.arclength_s)
        dd_s  = get_diffusion_s(self.spline_unstable, p0_s_vals[0], self.spline_y_s_given_s, Big_Sigma)
        spline_dd_s_given_s = UnivariateSpline(p0_s_vals[0], np.round(dd_s, 2) , k=3, s=0)

        # THIS IS OUR RANGE
        arclength_s_bitOut = np.linspace(self.Sa-spline_dd_s_given_s(s_tau_gives_s(self.Sa))*2, 
                                            self.Sb+spline_dd_s_given_s(s_tau_gives_s(self.Sb))*2, range_fp)

        # Dynamics
        spline_vf_given_s = get_dynamics_s(arclength_s_bitOut, self.spline_y_s_given_s, self.spline_x_s_given_s,
                                            self.spline_s_given_y_s, self.spline_unstable, self.G, self.uvw)
        
        return arclength_s_bitOut, spline_p0s_given_s, spline_vf_given_s, spline_dd_s_given_s

    def get_likelihood(self, observed_data, m_0, V_t0, Big_Sigma, observed_t, dt, MaxTime = 15, containment= 0.9, prob = 0.9, thick = 50):
        checks_done = self.LNA_check(m_0, V_t0, Big_Sigma, observed_t, dt, MaxTime, containment, prob, thick)
        check_mean = checks_done[1]
        checks_covar = checks_done[2]
        t_star_checks = checks_done[3]

        # run as long as the difference between t* and observed_t
        t_delta_start_observed = observed_t-t_star_checks*dt
        print("fp running for time {}".format(t_delta_start_observed))
        if checks_done[0]:
            # there must be a smarter way to do this
            arclength_s_bitOut, spline_p0s_given_s, spline_vf_given_s, spline_dd_s_given_s = self.initialise_FP(
                                                                    xt_critical = check_mean, 
                                                                    Vt_critical = checks_covar, Big_Sigma = Big_Sigma,
                                                                    range_fp = 400)

            fp_range_s, fp_solution, datatrack_flux = FP_run(t_delta_start_observed, arclength_s_bitOut, 
                                                            spline_p0s_given_s,spline_vf_given_s, spline_dd_s_given_s,
                                                            # could customise
                                                            interval_track = dt*10)
            # customisable
            interval_track = dt*10
            #this it the no flux time in fp scale
            t_no_flux_fp= (len(datatrack_flux.data)-1)*(interval_track)
            
            #return fp_range_s, fp_solution

            if (t_delta_start_observed-t_no_flux_fp)<dt*10:
                print("No flux condition was not satisfied, data projected on manifold")

                # need to compute the density of projected data
                #project_data
                data_on_Wu = project_observed_on_Wu(observed_data, 1/self.m_unstable, -self.q_unstable/self.m_unstable, 
                                                    self.m_stable, self.direction_X_scaled, self.xs, self.spline_tau_gives_s)
                # interpolate fp with spline
                fp_spline = interp1d(fp_range_s.data, fp_solution.data, kind='cubic')
                # density of data in fp
                data_density_fp = fp_spline(data_on_Wu)
                # loglikelihood
                loglik = np.sum(np.log(data_density_fp))
            else:
                # no flux reached before observed t
                # this is the time of no flux in positional
                t_no_flux = (t_star_checks*dt + t_no_flux_fp)
            
                print("t_no_flux_fp: {}".format(t_no_flux))
                print("2D reconstruction")

                lna2_time = observed_t - t_no_flux
                print("Running the two lnas until observed time, remaining {}".format(lna2_time))

                ## 2D Reconstruction
                s_peak, sigma_firstComp = find_peaks_variances(fp_range_s.data, fp_solution.data)
                fp_mean_manifold_x = self.spline_x_s_given_s(s_peak)
                fp_mean_manifold_y = self.spline_y_s_given_s(s_peak)

                fp_means_manifold = np.column_stack([fp_mean_manifold_x, fp_mean_manifold_y])
                #print("sigma of components bottom {} upper {}".format(sigma_firstComp[0], sigma_firstComp[1]))
                # take LNA final solution
                LNA_res = checks_done[4]
                x_t_final = LNA_res["x_t"].squeeze()[-1,:]
                V_t_final = LNA_res["V_t"].squeeze()[-1,:]
                print("Limiting Variance")
                lna_sigma = np.sqrt(min(V_t_final[0], V_t_final[3]))
                print("sigma associated with LNA mean {}".format(lna_sigma))
                #LNA_t = LNA_res["t"].squeeze()
                #eigen_V_t= LNA_res["eigen_V_t"].squeeze()
                #eigen_DIR_V_t = LNA_res["eigendir"].squeeze()

                dist_a = LA.norm(self.xa - x_t_final)
                dist_b = LA.norm(self.xb - x_t_final)
                if dist_a<dist_b:
                    # need to solve LNA for b
                    around_attractor_start = self.xb + np.sqrt(np.diag(Big_Sigma))
                    radau_solution = self.get_t_s(np.hstack([0, around_attractor_start]) , V_t0, Big_Sigma, dt, max(MaxTime, observed_t))[1]
                    # variances 
                    opposite_sigma = np.sqrt(min(radau_solution[4], radau_solution[7]))
                    print("Sigma of Opposite attractor: {}".format(opposite_sigma))
                    
                    V_ta = np.array([sigma_firstComp[1], 0, 0, lna_sigma])
                    V_tb = np.array([sigma_firstComp[0], 0, 0, opposite_sigma])
                else:
                    # need to solve LNA for a
                    around_attractor_start = self.xa + np.sqrt(np.diag(Big_Sigma))
                    radau_solution = self.get_t_s(np.hstack([0, around_attractor_start]) , V_t0, Big_Sigma, dt, max(MaxTime, observed_t))[1]
                    opposite_sigma = np.sqrt(min(radau_solution[4], radau_solution[7]))
                    print("Sigma of Opposite attractor: {}".format(opposite_sigma))
                    V_ta = np.array([sigma_firstComp[1], 0, 0, opposite_sigma])
                    V_tb = np.array([sigma_firstComp[0], 0, 0, lna_sigma])

                # Now run the two sep LNAs:
                lna_a = self.get_t_s(np.hstack([0, fp_means_manifold[1,:]]), V_ta, Big_Sigma, dt, lna2_time)[1]
                lna_b = self.get_t_s(np.hstack([0, fp_means_manifold[0,:]]), V_tb, Big_Sigma, dt, lna2_time)[1]

                weights_b_a = find_weights(fp_range_s.data, fp_solution.data)
                # loglik = np.sum--- 2dMixture
                gausMix_oneD_pdf(means, variances, weights)
                loglik = 0
        else:
            # compute likelihood for the multivariate gaussian
            loglik = np.sum(np.log(multivariate_normal.pdf(observed_data, check_mean, checks_covar)))
        return loglik


######## OLD VERSION OF STOPPIN CROSS UNSTABLE

            # # restart from time 0
            # unstable_cross_pos = 0
            # # Create a vector to store projections on s
            # proj_store = np.zeros((int(MaxTime/dt), 2))
            # not_in_range = True
            # # 13/09/22
            # # taking away condition that we need to be before closest point
            # while (not_in_range & (unstable_cross_pos<=t_close)):
            # #while not_in_range:
            #     selected_mean_cross = x_t[0,unstable_cross_pos, :].reshape(1, 2)[0]
            #     selected_cov_cross = V_t[0,unstable_cross_pos, :].reshape(2, 2)
            #     ellipe_points_all_ic = return_ellipse_points(selected_mean_cross, selected_cov_cross, prob, thick)

            #     intersect_unstable = intersection_check(ellipe_points_all_ic, self.m_unstable, self.q_unstable)
            #     if not intersect_unstable:
            #         unstable_cross_pos = unstable_cross_pos+1
            #     else:
            #         # it crossed, calculate how much
            #         projection_vals, proj_range, in_range = condition_on_s(ellipe_points_all_ic, self.m_unstable, self.q_unstable, self.m_stable,
            #         self.direction_X_scaled, self.xs, self.arclength_s, self.spline_tau_gives_s, self.Sa_e, self.Sb_e)
            #         if in_range:
            #             unstable_cross_found = unstable_cross_pos
            #             proj_store[unstable_cross_pos, :] = projection_vals
            #             # 13/09 comment if
            #             if unstable_cross_pos<=t_close:
            #                 unstable_cross_pos = unstable_cross_pos+1
            #         else:
            #             not_in_range = False
            #             unstable_cross_pos = np.inf

            # selected_mean_cross_found = x_t[0,unstable_cross_found, :].reshape(1, 2)[0]
            # selected_cov_cross_found = V_t[0,unstable_cross_found, :].reshape(2, 2)
            # print(unstable_cross_found)
            # return intersect_stable, selected_mean_cross_found, selected_cov_cross_found, proj_store

            

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