# to find the order of magnitude
import math
from scipy.stats import multivariate_normal
from scipy.interpolate import UnivariateSpline

import numpy as np
from numpy import linalg as LA

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

# given an initial distribution, we integrate over one direction
def IntegrateOverP(xt_critical, Vt_critical, direction_P_scaled, Sad, print_ = False):
    # tangent passing through the center of the distribution

    #s0_crit = Ct_critical[0] - Ct_critical[1]*s1_saddle_unstable
    # get the new 2d gaussian wrt s, q: (P, P')

    #B = np.column_stack([direction_X_scaled, direction_Y_scaled])

    #Rotation matrix (in 2d)
    direction_P_perp = np.array([-direction_P_scaled[1], direction_P_scaled[0]])
    direction_P_perp_scaled = direction_P_perp/LA.norm(direction_P_perp)

    # ROTATION MATRIX
    B = np.column_stack([direction_P_scaled, direction_P_perp_scaled])
    # I know that the first dimension is the one I am interested
    
    Saddle_rotated = LA.inv(B) @ Sad

    # projected means
    mean_Xpp = LA.inv(B) @ xt_critical

    # mean centred
    # if we work with mean_Xsq we subtract it to get [0,0]
    mean_Xpp_from_centred = LA.inv(B) @ np.zeros(2)

    # variance projected
    V_Xpp = LA.inv(B) @ Vt_critical @ np.transpose(LA.inv(B))

    # # set p perp as dimension with larger var: WRONG
#     pp_V = np.diag(V_Xpp)
#     if (pp_V[0]>pp_V[1]):
#         mean_Xpp = mean_Xpp[np.array([1, 0])]
#         # easy here because it is two dimensional
#         temp_V11 = np.copy(V_Xpp[0,0])
#         V_Xpp[0,0] = np.copy(V_Xpp[1, 1])
#         V_Xpp[1, 1] = np.copy(temp_V11)
        
    if print_:
        plot_2d_ellipse(mean_Xpp, V_Xpp, 2, lw = 2, Col = "red")
        plt.axis('square')
        rotated_P = LA.inv(B) @direction_P_scaled
        plt.plot([0, rotated_P[0]], np.array([0, rotated_P[1]])+Saddle_rotated[1], label = "P")

        rotated_P_perp =  LA.inv(B) @direction_P_perp_scaled
        plt.plot(np.array([0, rotated_P_perp[0]])+Saddle_rotated[0], [0, rotated_P_perp[1]], label = "P'")

        #plt.plot([0, direction_P_scaled[0]], [0, direction_P_scaled[1]])

        rotated_X = LA.inv(B) @direction_X_scaled
        rotated_Y = LA.inv(B) @direction_Y_scaled
        plt.legend()
    
    # calculate the 2d on a grid
    #NOT mean = Saddle_rotated
    proj_lna_gaus = multivariate_normal(mean=[0,0], cov=V_Xpp)

    # on a grid
    sd_p1 = np.sqrt(V_Xpp[0, 0])
    sd_p2 = np.sqrt(V_Xpp[1, 1])

    many_sd_away = 4 
    # WHY DOES IT NEED TO BE CENTERED AT SADDLE? IT DOES NOT
    p1_range_P = np.arange(-sd_p1*many_sd_away, #+Saddle_rotated[0],
                         sd_p1*many_sd_away, 10**(orderOfMagnitude(sd_p1)-1)) # after 4th sd it is all 0
    p2_range_Pperp = np.arange(-sd_p2*many_sd_away ,
                         sd_p2*many_sd_away, 10**(orderOfMagnitude(sd_p2)-1))

    P1, P2_perp = np.meshgrid(p1_range_P, p2_range_Pperp, indexing='xy')
    points = np.stack((P1, P2_perp), axis=-1)

    p_gaus_Grid = proj_lna_gaus.pdf(points)
    # renormalize
    p_gaus_Grid = p_gaus_Grid/np.sum(p_gaus_Grid)
    
    Gaus_2d = [P1, P2_perp, p_gaus_Grid]
    
    # Integration
    proj_gaus_p_perp = np.zeros(len(p2_range_Pperp))
    for pp in range(len(p2_range_Pperp)):
        # should I multiply by *step?
        proj_gaus_p_perp[pp] = np.sum(p_gaus_Grid[pp, :])
    
#     # 29.06. mean_Xpp rearranged
#     # Instead of that, centre around the Saddle

    #print(xt_repar)
    integrated_gaus = [p2_range_Pperp , proj_gaus_p_perp]
#     # make it into a spline
    #print(intersection_x, intersection_y)
    spline_projection = UnivariateSpline(p2_range_Pperp, proj_gaus_p_perp,  k=3, s=0)
    
    return integrated_gaus, spline_projection

# Separate function for getting mapping from tangent to arclength
# get_mapping_from_tau_Tos

# getting initial distribution p0_s
def get_p0_s(s1_saddle_unstable,s0_saddle_unstable, s1_saddle_stable, x_unstable, Sad, xt_critical, direction_X_scaled, integrated_g, spline_projection, arclength_s):
    # tangent space
    # v_vector*tau+sad

    # Rotate tangent and manifold
    thera_rotate = -np.arctan(1/s1_saddle_unstable);
    # with the minus it is graphically more consistent
    R_theta = np.array([[np.cos(thera_rotate), -np.sin(thera_rotate)], [np.sin(thera_rotate), np.cos(thera_rotate)]])

    # rotating the saddle, thus knowing the two extremes
    x_unstable_man_rotated = (R_theta@x_unstable.T).T

    Sad_rotated = (R_theta@Sad.T).T

    v_vector = direction_X_scaled
    v_vector_rotated = R_theta@v_vector
    
    #tangent_sample_rotated = (R_theta@tangent_sample.T).T

    tangent_rotated = np.column_stack([x_unstable_man_rotated[:,0],np.ones(x_unstable.shape[0])*Sad_rotated[1]])
    tangent_original = (LA.inv(R_theta)@tangent_rotated.T).T
    # [x_tan, y_tan] = v* [tau_x, tau_y] + Sad
    # # solving for tau
    #tau_x = (tangent_original[:,0] - Sad[0][0])/v_vector[0]
    tau_y = (tangent_original[:,1] - Sad[1])/v_vector[1]
    spline_tau_given_s = UnivariateSpline(arclength_s, tau_y, k=3, s=0)
    #spline_tau_given_sigma = UnivariateSpline(sigma_arclen, tau_y, k=3, s=0)

    #rho(tau) = s
    s_tau_gives_s = UnivariateSpline(tau_y, arclength_s, k=3, s=0)
    #rho(s) = tau
    s_s_gives_tau = UnivariateSpline(arclength_s, tau_y, k=3, s=0)
    
    
    m2 = 1/s1_saddle_unstable
    q2 = -s0_saddle_unstable/s1_saddle_unstable
    
    q1 = xt_critical[1]- s1_saddle_stable*xt_critical[0]
    #q1 = s0_stab
    m1 = s1_saddle_stable
    
    #  intersection
    intersection_x = (q2-q1)/(m1-m2)
    intersection_y = m1*intersection_x+q1

#     #rotate
    xt_crit_rotated = R_theta@ np.array([intersection_x, intersection_y])
    # Transformation of P(perp)
    #shifted_p2 = integrated_g[0] + Sad_rotated[0]
    shifted_p2 = integrated_g[0] + xt_crit_rotated[0]

    # New tangent range is essentially p2, unless p2 is smaller
    # already centre at saddle

    tangent_ranges_flat = np.linspace(min(np.hstack([tangent_rotated[:,0], shifted_p2])),
                                max(np.hstack([tangent_rotated[:,0], shifted_p2])),
                                max([len(tangent_rotated[:,0]), len(shifted_p2)]))


    # Extended Integrated range: 
    # set to 0 when we get out of bounds
    # p2_proj_density_tangent_range = np.where(spline_projection(tangent_ranges_flat- Sad_rotated[0]) < 0, 0,
    #                                          spline_projection(tangent_ranges_flat- Sad_rotated[0]))
    p2_proj_density_tangent_range = np.where(spline_projection(tangent_ranges_flat- xt_crit_rotated[0]) < 0, 0,
                                              spline_projection(tangent_ranges_flat- xt_crit_rotated[0]))
    

    # Can move it by centring in Sad
    tangent_rotated = np.column_stack([tangent_ranges_flat, [Sad_rotated[1]]*len(tangent_ranges_flat)])

    tangent_extended = (LA.inv(R_theta)@tangent_rotated.T).T

    # extend the tangent but not too much, keep it as in 
    ## SAME
    tau_y_extended = (tangent_extended[:,1]-Sad[1])/v_vector[1]
    tau_y_rotated = (tangent_rotated[:,0] - Sad_rotated[0])/v_vector_rotated[0]

    arcl_s_extended_tau = s_tau_gives_s(tau_y_extended)
    d_s_s_gives_tau = s_s_gives_tau.derivative()

    #Spline from tang to tau
    s_tau_gives_p2flat = UnivariateSpline(tau_y_rotated, tangent_rotated[:,0] , k=3, s=0) 
    d_s_tau_gives_p2flat_d_tau = s_tau_gives_p2flat.derivative()

    s_tau_gives_tang = UnivariateSpline(tau_y_extended, tangent_extended[:,1] , k=3, s=0) 

    # Chain Rule
    p_tau = p2_proj_density_tangent_range * np.abs(d_s_tau_gives_p2flat_d_tau(tau_y_rotated))
    p_tau = p_tau/np.sum(p_tau)


    p_s = p_tau * np.abs(d_s_s_gives_tau(arcl_s_extended_tau))
    p_s = p_s/np.sum(p_s)

    # rho(p) = s    #from rotated p2 or tang to arclength
    s_s_gives_p2 = UnivariateSpline(arcl_s_extended_tau, tangent_rotated[:,0] , k=3, s=0) 
    d_s_s_gives_p2_d_s = s_s_gives_p2.derivative()


    p_s_direct = p2_proj_density_tangent_range * np.abs(d_s_s_gives_p2_d_s(arcl_s_extended_tau))
    p_s_direct = p_s_direct/np.sum(p_s_direct)
    
    values_p0_s = [arcl_s_extended_tau, p_s_direct]
    spline_p0s_given_s = UnivariateSpline(arcl_s_extended_tau, p_s_direct , k=3, s=0)
    
    
    return values_p0_s, spline_p0s_given_s, s_tau_gives_s

# getting diffusion function
def get_diffusion_s(spline_manifold, arcl_t_grid, spline_y_t_given_t, Big_Sigma):
    # points of the manifold: arcl_t_extended_tau
    y_unst = spline_y_t_given_t(arcl_t_grid)
    #x_unst = spline_x_t_given_t(arcl_t_grid)
    #x_unstable_extended = np.column_stack([spline_unstable(y_unst),y_unst])

    # compute the tangents at points of the arclength
    spline_der = spline_manifold.derivative()
    spline_der_vals = spline_der(y_unst) # <- m

    diff_s = np.zeros(len(y_unst))
    #diff_s_David = np.zeros(len(y_unst))
    for i in range(len(y_unst)):
        # angle with x axis
        alph = np.arctan(1/spline_der_vals[i])
        
        # if angle is positive, rotate clockwise
        if alph>=0:
            R_alph = np.array([[np.cos(alph), -np.sin(alph)],
                                        [np.sin(alph), np.cos(alph)]])
        else:
            R_alph = np.array([[np.cos(-alph), np.sin(-alph)],
                                            [-np.sin(-alph), np.cos(-alph)]])

        Sigma_rot_alph = LA.inv(R_alph) @ Big_Sigma @ LA.inv(R_alph).T
        # get the marginal x -  simpy drop the rest?
        marg_tang_sigma_newx = np.sqrt(np.abs(Sigma_rot_alph[0,0]))
        diff_s[i] = marg_tang_sigma_newx
    # get the conditional, x/y=0
    return(diff_s)

def get_dynamics_s(arclength_s_bitOut, spline_y_s_given_s, spline_x_s_given_s, spline_s_given_y_s, spline_unstable, G, uvw):

        # DYNAMICS
        y_unst = spline_y_s_given_s(arclength_s_bitOut)
        x_unstable_extended = np.column_stack([spline_unstable(y_unst),y_unst])

        Omega = G(x_unstable_extended, uvw)

        # from Omega, we get the arclength
        #v1_s = spline_t_given_x_t(Omega[:,0])

        v2_s = spline_s_given_y_s(Omega[:,1]+x_unstable_extended[:,1]) - spline_s_given_y_s(x_unstable_extended[:,1])

        # this is the spline derivative of s(t) = x, arclength
        d_y_s_t_d_s_t = spline_y_s_given_s.derivative()
        d_x_s_t_d_s_t = spline_x_s_given_s.derivative()


        d_y_s_t_d_s_t_computed = d_y_s_t_d_s_t(arclength_s_bitOut)
        #d_x_s_t_d_s_t_computed = d_x_s_t_d_s_t(arclength_s_bitOut)

        s_dot_s = v2_s / d_y_s_t_d_s_t_computed
        spline_vf_given_s = UnivariateSpline(arclength_s_bitOut, s_dot_s , k=3, s = 0)
        return spline_vf_given_s


import sys
#sys.path.append('../..')  # add the pde package to the python path
import pde

from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, SteadyStateTracker
from pde import CallbackTracker, DataTracker

# FP_run, checkin for unstable manifold
def FP_run(t_max, arclength_s_bitOut, spline_p0s_given_s, spline_vf_given_s, spline_dd_s_given_s, interval_track):
    #arclength_s = np.copy(arclength_s_bitOut)
    #arclength_s = arcl_s_extended_tau

    # saddle_point = 0
    saddle_arcl_s = np.abs(arclength_s_bitOut).argmin()
    
    grid_arcl = CartesianGrid([[arclength_s_bitOut[0], arclength_s_bitOut[-1]]], [len(arclength_s_bitOut)], periodic=False)#
    #grid_arcl = CartesianGrid([[arcl_t_extended_tau[0], arcl_t_extended_tau[-25]]], [len(arcl_t_extended_tau[:-25])], periodic=False)#

    s_field = ScalarField.from_expression(grid_arcl, "x")

    # initial values - same
    initial_p = spline_p0s_given_s(s_field.data)
    initial_p[initial_p<0] = 0
    # properly normalized?
    initial_p = initial_p/np.sum(initial_p)
    state_p0 = pde.ScalarField(grid_arcl, data=initial_p, label="p0")

    # solve the equation and store the trajectory
    #storage = MemoryStorage()

    #t_max = 5

    # Saving in Disk
    # storage=FileStorage("FP_pypde_results/ArcLength_1D_Asym{}_sigmae{}_T".format(str(v).replace(".", ""),
    #                                                                  str(se).replace(".", ""),
    #                                                                 t_max))

    storage=MemoryStorage()
    #trackers = [storage.tracker(interval=pt)]
    #controller1 = Controller(ExplicitSolver(eq), t_range=T, tracker=trackers)

    ##############
    # Dynamics projected

    spline_s_vec_field = spline_vf_given_s(s_field.data)
    velocity = pde.VectorField(grid_arcl, data=spline_s_vec_field)

    # Diffusion
    dd_s_field = spline_dd_s_given_s(s_field.data)
    dd_s_field_FP = (dd_s_field**2)/2
    diffusion_s = pde.VectorField(grid_arcl, data=dd_s_field_FP)

    eq = pde.PDE({'P': f'-divergence(P * V) + laplace(P * D)'}, consts={'V': velocity, 'D': diffusion_s}         
                ,bc = [{"value":0},{"value":0}] )

    def get_flux_allGrad(data):
        P = data
        return P.gradient(bc = [{"value":0}])

    def get_flux_atSad(data):
        P = data
        P_grad = P.gradient(bc = [{"value":0}])
        return P_grad.interpolate(0)
    
    def get_flux_atSad_fast(data):
        P = data.data
        P_grad = np.gradient(P)
        flux_sad = P_grad[saddle_arcl_s]
        return flux_sad
        
    def calc_max(data):
        P = data.data
        how_many_higher = np.sum(P>P[saddle_arcl_s])/len(P)
        return how_many_higher<0.1
    
    # simulate the pde
    #data_tracker_flux = DataTracker(get_flux_allGrad, interval=0.1)
    data_tracker_flux_atS = DataTracker(get_flux_atSad_fast, interval = interval_track)
    
    # simulate the pde
    #pde.SteadyStateTracker(interval = 0.1, rtol = 1e-04)
    def check_simulation(data):
        cond = get_flux_atSad_fast(data)
        #print(cond)
        if (np.abs(cond) < 5e-06) & (not calc_max(data)):
            raise pde.base.FinishedSimulation("No flux at saddle")

    tracker_check = CallbackTracker(check_simulation, interval=interval_track)
    
    #adaptive=True, method="explicit_mpi"
    
    sol = eq.solve(state_p0, t_range=t_max, dt=1e-5,
                   tracker=[tracker_check, data_tracker_flux_atS]) # data_tracker_flux_atS, data_tracker_flux
                                                              #storage.tracker(track_t)])
        #data_tracker_flux_atS

    #sol = eq.solve(state_p0, t_range=t_max, dt=1e-5, tracker={"progress", "steady_state", storage.tracker(track_t)})

    #plt.plot(s_field.data, storage.data[-1], label = "t final")

    # plot the trajectory as a space-time plot
    #plot_kymograph(storage)
    
    # trackers save it at t=0 and then every interval
    return s_field, sol, data_tracker_flux_atS