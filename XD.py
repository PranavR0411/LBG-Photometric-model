# This code is based on the Extreme deconvolution algorithm developed by Bovy et al (2011). This code is run on redshifted PCA components of EAZY galaxy templates to get the redshift-dependent coefficients 
# for these templates to explain the galaxies observed in COSMOS. 
# There are some modifications to the original algorithm developed by Bovy et al. Here, in this modified code, we are inferring the coefficients and E(B-V) value for dust simultaneously. 
# Extreme Deconvolution being a linear model, can't handle the exponential nature of dust properly. Hence we use the Extended Kalman Filter (EKF) to linearize the dust using it's jacobian.
# The M step is also modified to give us redshift-dependent coefficients. The mean of cofficient population is modelled as mu(z)=M0+M1z, where M0 and M1 are got through linear regression, which can be changed to other non linear options later.
# This code takes in the redshifted PCA components, and using EKF, simultaneously infers the redshift dependant coefficients and dust value for each galaxy in the COSMOS dataset.

import numpy as np
import pandas as pd
from sedpy.observate import getSED, load_filters
from scipy.interpolate import RegularGridInterpolator
import time
import matplotlib.pyplot as plt
plt.clf()

f_all = np.load('pcaflux_w_redshift_igm.npy') # redshifted PCA components
z_grid1=np.load('zgrid.npy')
f_interp_vec = RegularGridInterpolator((z_grid1,), f_all, bounds_error=False, fill_value=None) # Interpolated components

# data
z = df['ez_z_phot'].to_numpy()
mask = np.isfinite(z) & (z<8)
z = z[mask]
Y = np.nan_to_num(df_flux.to_numpy(float), nan=0.0)[mask]
S = np.nan_to_num(df_err.to_numpy(float), nan=1e6)[mask]
R = S**2
N, M = Y.shape

N_bands = len(fil) 
Kc = 4 

# initialize
theta_new=np.zeros((N,Kc+1)) 
theta_new[:, -1] = np.where(f_ratio > np.median(f_ratio), np.log(0.9/(1-0.9)), np.log(0.3/(1-0.3))) # initialize theta value based on color of filter to overcome dust redshift degenracy
M0 = np.mean(alpha3, axis=0) 
#M0 = np.zeros(Kc) 
M1 = np.zeros(Kc) 
mu_eta=np.mean(theta_new[:,-1])

Sigma = np.zeros((Kc+1,Kc+1))
sig = np.cov(alpha3, rowvar=False)
Sigma[:Kc,:Kc]=sig
Sigma[Kc,Kc]=0.3**2

try:
    Sigma_inv = np.linalg.inv(Sigma)
except:
    Sigma_inv = np.linalg.pinv(Sigma)

num_iterations = 20
convergence_threshold = 1e-10
batch_size = 50000
print("\n--- EM for Redshift-Dependent Coeff Mean & Separate Redshift Mean ---")
for it in range(num_iterations):

    start = time.time()
    M0_old, M1_old= M0.copy(), M1.copy()
    mu_eta_old=mu_eta
    theta_old=theta_new.copy()
    theta_new=np.zeros((N,Kc+1))
    theta_new[:,-1]=theta_old[:,-1]
    sum_t = np.zeros((Kc, 1))
    sum_zt = np.zeros((Kc, 1))
    sum_B = np.zeros((Kc+1, Kc+1))
    sum_t_outer = np.zeros((Kc+1, Kc+1))
    sum_eta=0
    #logL = 0.0

    #bi_all = np.zeros((N, Kc+1))   # store full posterior means
    batch_idx = 0
    for b_start in range(0, N, batch_size):
    
        b_end = min(b_start + batch_size, N)
        Yb, Rb = Y[b_start:b_end], R[b_start:b_end]
        zb, zsigb = z[b_start:b_end], z_sigma[b_start:b_end]
        thetab=theta_old[b_start:b_end]
        B = b_end - b_start
    
        Fz = f_interp_vec(zb)
        Rinv = 1.0 / np.maximum(Rb,1e-6)
        eta=thetab[:,-1]
        E=sigmoid(eta) # model E(B-V) as sigmoid to prevent the algorithm from inferring negative values
        b=thetab[:,:-1]
        lam_rest_A = lam_eff_obs_A[None, :] / (1.0 + zb[:, None])
        k_eff = calzetti_k(lam_rest_A) # Use calzetti dust law to get attenuation
        trans=10**(-0.4*k_eff*E[:,None])
        pred=trans*np.einsum('bkn,bk->bn', Fz, b)

        C=(trans[:,None,:]*Fz)
        dE_deta = (E * (1.0 - E))                   # EKF
        dPred_dE = (-0.4*np.log(10.0) * k_eff * pred)  
        M = (dPred_dE * dE_deta[:, None])[:, None, :]  
        H = np.concatenate([C, M], axis=1) #Jacobian

        HSH = np.einsum('bkn,bn,bjn->bkj', H, Rinv, H)   
        delta_y = Yb - pred
        HSy = np.einsum('bkn,bn->bk', H, Rinv * delta_y)
        try:
            Bi = np.linalg.inv(HSH + Sigma_inv[None, :, :])
        except:
            Bi = np.linalg.pinv(HSH + Sigma_inv[None, :, :])
            
        mu=M0[None, :] + M1[None, :] * zb[:, None]
        mu_etab=np.full((len(zb), 1), mu_eta)
        mu_full = np.hstack([mu, mu_etab])
        
        rhs = HSy+ np.einsum('kj,bj->bk', Sigma_inv, mu_full - thetab)
        dtheta = np.einsum('bkj,bj->bk', Bi, rhs)

        eta_current = thetab[:, -1]
        deta = dtheta[:, -1]
        alpha = np.ones(len(eta_current))
        
        theta_newb = thetab +  alpha[:, None] * dtheta
        theta_new[b_start:b_end]=theta_newb

        sum_t += np.sum(theta_newb[:,:-1], axis=0, keepdims=True).T
        sum_eta += np.sum(theta_newb[:,-1])
        sum_zt += np.sum(zb[:, None] * theta_newb[:,:-1], axis=0, keepdims=True).T
        sum_B += np.sum(Bi, axis=0)
        sum_t_outer += np.einsum('bi,bj->ij', theta_newb, theta_newb)
        #sum_E_outer += np.einsum('b,b->bb', theta_newb[:,-1], theta_newb[:,-1])
    # M step
    sum_z = np.sum(z)
    sum_z2 = np.sum(z**2)

    A_reg = np.array([[N, sum_z], [sum_z, sum_z2]])
    try:
        A_reg_inv = np.linalg.inv(A_reg)
    except:
        A_reg_inv = np.linalg.pinv(A_reg)

    rhs2 = np.vstack([sum_t.T, sum_zt.T])
    # Linear Regression
    M0, M1 = (A_reg_inv @ rhs2)
    
    mu_eta = np.mean(theta_new[:, -1])
    mu_final_b = M0[None, :] + M1[None, :] * z[:, None]
    mu_final_eta = np.full((N, 1), mu_eta)
    mu_final = np.hstack([mu_final_b, mu_final_eta])
    
    # Efficient Sigma Update
    # (theta_outer and B are already accumulated)
    sum_mu_theta = np.einsum('bi,bj->ij', mu_final, theta_new)
    sum_mu_mu = np.einsum('bi,bj->ij', mu_final, mu_final)
    
    Sigma = (sum_B + sum_t_outer - sum_mu_theta - sum_mu_theta.T + sum_mu_mu) / N
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except:
        Sigma_inv = np.linalg.pinv(Sigma)
    # Convergence
    delta_M = np.linalg.norm(M0 - M0_old) + np.linalg.norm(M1 - M1_old) 
    print(f"Iter {it+1:02d} | ΔM = {delta_M:.3e} | E Mean: {sigmoid(mu_eta)}")
    end = time.time()
    
    if delta_M < convergence_threshold:
        print("Converged.")
        break

print("\n--- Fit Complete ---")
print("Final M0c:", M0)
print("Final M1c:", M1)
print("Final mu E:", mu_eta)
print("Final Sigma:", Sigma)
#plt.figure(logL)
        
        
