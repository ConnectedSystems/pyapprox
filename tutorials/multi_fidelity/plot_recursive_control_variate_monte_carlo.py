r"""
Recursive Approximate Control Variate Monte Carlo
=================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and demonstrates that multi-level Monte Carlo and multi-fidelity Monte Carlo are both approximate control variate techniques.


Multi-level Monte Carlo (MLMC)
------------------------------
The multi-level (MLMC) estimator based on :math:`M+1` models :math:`f_0,\ldots,f_M` ordered by decreasing fidelity (note typically MLMC literature reverses this order) is given by

.. math:: Q_0^\mathrm{ML}=\mean{f_M}+\sum_{\alpha=1}^M\mean{f_{\alpha-1}-f_\alpha}

Similarly to ACV we approximate each expectation using Monte Carlo sampling such that

.. math::  Q_{0,N,\mathcal{Z}}^\mathrm{ML}=Q_{M,\hat{\mathcal{Z}}_{M}}+\sum_{\alpha=1}^M\left(Q_{\alpha-1,\hat{\mathcal{Z}}_{\alpha-1}}-Q_{\alpha,\hat{\mathcal{Z}}_{\alpha-1}}\right),

for some sampling sets :math:`\mathcal{Z}=\cup_{\alpha=0}^M\hat{\mathcal{Z}}_{\alpha}`.

The three model MLMC estimator is

.. math:: Q_{0,\mathcal{Z}}^\mathrm{ML}=Q_{2,\hat{\mathcal{Z}_{2}}}+\left(Q_{1,\hat{\mathcal{Z}}_{1}}-Q_{2,\hat{\mathcal{Z}}_{1}}\right)+\left(Q_{0,\hat{\mathcal{Z}}_{0}}-Q_{1,\hat{\mathcal{Z}}_{0}}\right)

By rearranging terms it is clear that this is just a control variate estimator

.. math::

    Q_{0,\mathcal{Z}}^\mathrm{ML}&=Q_{0,\hat{\mathcal{Z}}_{0}} - \left(Q_{1,\hat{\mathcal{Z}}_{0}}-Q_{1,\hat{\mathcal{Z}}_{1}}\right)-\left(Q_{2,\hat{\mathcal{Z}}_{1}}-Q_{2,\hat{\mathcal{Z}}_{2}}\right)\\
   &=Q_{0,\mathcal{Z}_{0}} - \left(Q_{1,\mathcal{Z}_{1,1}}-Q_{1,\mathcal{Z}_{1,2}}\right)-\left(Q_{2,\mathcal{Z}_{2,1}}-Q_{2,\mathcal{Z}_{2,2}}\right)

where in the last line we have used the general ACV notation for sample partitioning. The control variate weights in this case are just :math:`\eta_1=\eta_2=-1`.

The MLMC and ACV IS sample sets are depicted in :ref:`mlmc-sample-allocation` and :ref:`acv-is-sample-allocation-mlmc-comparison`, respectively

By inductive reasoning we get the :math:`M` model ACV version of the MLMC estimator.

.. math:: Q_{0,\mathcal{Z}}^\mathrm{ML}=Q_{0,\mathcal{Z}_{0}} +\sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha-1,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha,2}}\right)

where :math:`\eta_\alpha=-1,\forall\alpha` and :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1,2}`, and :math:`\mu_{\alpha,\mathcal{Z}_{\alpha,2}}=Q_{\alpha,\mathcal{Z}_{\alpha,2}}`
 
.. list-table::

   * - 
       .. _mlmc-sample-allocation:

       .. figure:: ../../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy

     - 
       .. _acv-is-sample-allocation-mlmc-comparison:

       .. figure:: ../../figures/acv_is.png
          :width: 100%
          :align: center

          ACV IS sampling strategy


Lets setup a problem to compute an MLMC estimate of :math:`\mean{f_0}`
"""
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import \
    TunableModelEnsemble, ShortColumnModelEnsemble, PolynomialModelEnsemble
from scipy.stats import uniform
from functools import partial
from scipy.stats import uniform,norm,lognorm
np.random.seed(1)

short_column_model = ShortColumnModelEnsemble()
model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m1,short_column_model.m2])

costs = np.asarray([100, 50, 5])
target_cost = int(1e4)
idx = [0,1,2]
cov = short_column_model.get_covariance_matrix()[np.ix_(idx,idx)]
# generate pilot samples to estimate correlation
# npilot_samples = int(1e4)
# cov = pya.estimate_model_ensemble_covariance(
#    npilot_samples,short_column_model.generate_samples,model_ensemble)[0]

# define the sample allocation
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov, costs, target_cost)[:2]
# generate sample sets
samples,values =pya.generate_samples_and_values_mlmc(
    nhf_samples,nsample_ratios,model_ensemble,
    short_column_model.generate_samples)
# compute mean using only hf data
hf_mean = values[0][0].mean()
# compute mlmc control variate weights
eta = pya.get_mlmc_control_variate_weights(cov.shape[0])
# compute MLMC mean
mlmc_mean = pya.compute_approximate_control_variate_mean_estimate(eta,values)

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MLMC error',abs(mlmc_mean-true_mean))
print('MC error',abs(hf_mean-true_mean))


#%%
#These errors are comparable. However these errors are only for one realiation of the samples sets. To obtain a clearer picture on the benefits of MLMC we need to look at the variance of the estimator.
#
#By viewing MLMC as a control variate we can derive its variance reduction [GGEJJCP2020]_
#
#.. math::  \gamma+1 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}}{\hat{r}_{M}} - \sum_{1=2}^M \frac{1}{\hat{r}_{i-1}}\left( \eta_i^2 \tau_{i}^2 + \tau_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),
#   :label: mlmc-variance-reduction
#
#where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`\hat{r}_\alpha=\lvert\mathcal{Z}_{\alpha,2}\rvert/N` is the ratio of the cardinality of the sets :math:`\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{0,2}`. 
#
#First let us use 2 models. The following code computes the variance reuduction of the estimator by computing the MLMC repeatedly with different realizations of the sample sets. If a :math:`\texttt{get_rsquared_mlmc}` function is available then it also returns the theoretical variance reduction
ntrials=1e1
get_cv_weights_mlmc = pya.get_mlmc_control_variate_weights_pool_wrapper
means1, numerical_var_reduction1, true_var_reduction1 = \
    pya.estimate_variance_reduction(
        model_ensemble, cov[:2,:2], short_column_model.generate_samples,
        pya.allocate_samples_mlmc,
        pya.generate_samples_and_values_mlmc, get_cv_weights_mlmc,
        pya.get_rsquared_mlmc,ntrials=ntrials,max_eval_concurrency=1,
        costs=costs[:2],target_cost=target_cost)
print("Theoretical 2 model MLMC variance reduction",true_var_reduction1)
print("Achieved 2 model MLMC variance reduction",numerical_var_reduction1)


#%%
# The numerical estimate of the variance reduction is consistent with the theory.
# Now let us compute the theoretical variance reduction using 3 models

true_var_reduction2 = 1-pya.get_rsquared_mlmc(cov,nsample_ratios)
print("Theoretical 3 model MLMC variance reduction",true_var_reduction2)

#%%
#The variance reduction obtained using three models is only slightly better than when using two models. The difference in variance reduction is dependent on the correlations between the models and the number of samples assigned to each model. However by looking at :eq:`mlmc-variance-reduction` we can see that the variance reduction is bounded by the CV estimator using the lowest fidelity model with the highest correlation with :math:`f_0`. 
#
#It is also worth empahsizing the MLMC only works when the variance between the model discrepancies decay. One needs to be careful that the variance between discrepancies does indeed decay. Sometimes when the models correspond to different mesh discretizations. Increasing the mesh resolution does not always produce a smaller discrepancy. The following example shows that, for this example, adding a third model actually increases the variance of the MLMC estimator. 

model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m3,short_column_model.m4])
idx = [0,3,4]
cov3 = short_column_model.get_covariance_matrix()[np.ix_(idx,idx)]
true_var_reduction3 = 1-pya.get_rsquared_mlmc(cov3,nsample_ratios)
print("Theoretical 3 model MLMC variance reduction for a pathalogical example",true_var_reduction3)

#%%
#Using MLMC for this ensemble of models creates an estimate with a variance orders of magnitude larger than just using the high-fidelity model.

#%%
#
#Multi-fidelity Monte Carlo (MFMC)
#---------------------------------
#This section of the tutorial introduces another recursive control variate estimator calledMulti-fidelity Monte Carlo (MFMC). To derive this estimator first recall the two model ACV estimator
#
#.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \eta\left(Q_{1,\mathcal{Z}_{0}}-\mu_{1,\mathcal{Z}_{1}}\right)
#
#The MFMC estimator can be derived with the following recursive argument. Partition the samples assigned to each model such that
#:math:`\mathcal{Z}_\alpha=\mathcal{Z}_{\alpha,1}\cup\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\alpha,2}=\emptyset`. That is the samples at the next lowest fidelity model are the samples used at all previous levels plus an additional independent set, i.e. :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1}`. See :ref:`mfmc-sample-allocation`. Note the differences between this scheme and the MLMC scheme.
#
#.. list-table::
#
#   * - 
#       .. _mfmc-sample-allocation:
#
#       .. figure:: ../../figures/mfmc.png
#          :width: 100%
#          :align: center
#
#          MFMC sampling strategy
#
#     -
#       .. _mlmc-sample-allocation-mfmc-comparison:
#
#       .. figure:: ../../figures/mlmc.png
#          :width: 100%
#          :align: center
#
#          MLMC sampling strategy
#
#Starting from two models we introduce the next low fidelity model in a way that reduces the variance of the estimate :math:`\mu_{\alpha}`, i.e.
#
#.. math::
#
#   Q_{0,\mathcal{Z}}^\mathrm{MF}&=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\left(\mu_{1,\mathcal{Z}_{1}}+\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\right)\right)\\
#   &=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\mu_{1,\mathcal{Z}_{1}}\right)+\eta_1\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\\
#
#We repeat this process for all low fidelity models to obtain
#
#.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)
#
#The optimal weights for the MFMC estimator are :math:`\eta=(\eta_1,\ldots,\eta_M)^T`, where for :math:`\alpha=1\ldots,M`
#
#.. math:: \eta_\alpha = -\frac{\covar{Q_0}{Q_\alpha}}{\var{Q_\alpha}}
#
#With this choice of weights the variance reduction obtained is given by
#
#.. math:: \gamma = 1-\rho_1^2\left(\frac{r_1-1}{r_1}+\sum_{\alpha=2}^M \frac{r_\alpha-r_{\alpha-1}}{r_\alpha r_{\alpha-1}}\frac{\rho_\alpha^2}{\rho_1^2}\right)
#
#Let us use MFMC to estimate the mean of our high-fidelity model.

# define the sample allocation
nhf_samples,nsample_ratios = pya.allocate_samples_mfmc(
    cov, costs, target_cost)[:2]
# generate sample sets
samples,values =pya.generate_samples_and_values_mfmc(
    nhf_samples,nsample_ratios,model_ensemble,
    short_column_model.generate_samples)
# compute mean using only hf data
hf_mean = values[0][0].mean()
# compute mlmc control variate weights
eta = pya.get_mfmc_control_variate_weights(cov)
# compute MLMC mean
mfmc_mean = pya.compute_approximate_control_variate_mean_estimate(eta,values)

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MLMC error',abs(mfmc_mean-true_mean))
print('MC error',abs(hf_mean-true_mean))

#%%
#Similarly to MLMC, using multiple models with MFMC only helps increase the speed to which the variance of the MFMC estimator converges to that of the 2 model CV estimator.
#
#Let us compare the variance reduction obtained by MLMC, MFMC and ACV with the MF sampling scheme as we increase the number of samples assigned to the low-fidelity models, while keeping the number of high-fidelity samples fixed.

poly_model = PolynomialModelEnsemble()
cov = poly_model.get_covariance_matrix()
nhf_samples = 10
nsample_ratios_base = [2, 4, 8, 16]
cv_labels = [r'$\mathrm{OCV1}$',r'$\mathrm{OCV2}$',r'$\mathrm{OCV4}$']
cv_rsquared_funcs=[
    lambda cov: pya.get_control_variate_rsquared(cov[:2,:2]),
    lambda cov: pya.get_control_variate_rsquared(cov[:3,:3]),
    lambda cov: pya.get_control_variate_rsquared(cov)]
cv_gammas = [1-f(cov) for f in cv_rsquared_funcs]
for ii in range(len(cv_gammas)):
    plt.axhline(y=cv_gammas[ii],linestyle='--',c='k')
    xloc = -.35
    plt.text(xloc, cv_gammas[ii]*1.1, cv_labels[ii],fontsize=16)

acv_labels = [r'$\mathrm{MLMC}$',r'$\mathrm{MFMC}$',r'$\mathrm{ACV}$-$\mathrm{MF}$']
acv_rsquared_funcs = [
    pya.get_rsquared_mlmc,pya.get_rsquared_mfmc,
    partial(pya.get_rsquared_acv,
            get_discrepancy_covariances=pya.get_discrepancy_covariances_MF)]

nplot_points = 20
acv_gammas = np.empty((nplot_points,len(acv_rsquared_funcs)))
for ii in range(nplot_points):
    nsample_ratios = [r*(2**ii) for r in nsample_ratios_base]
    acv_gammas[ii,:] = [1-f(cov,nsample_ratios) for f in acv_rsquared_funcs]
for ii in range(len(acv_labels)):
    plt.semilogy(np.arange(nplot_points),acv_gammas[:,ii],label=acv_labels[ii])
plt.legend()
plt.xlabel(r'$\log_2(r_i)-i$')
_ = plt.ylabel(r'$\mathrm{Variance}$ $\mathrm{reduction}$ $\mathrm{ratio}$ $\gamma$')

#%%
#As the theory suggests MLMC and MFMC use multiple models to increase the speed to which we converge to the 2 model CV estimator. These two approaches reduce the variance of the estimator more quickly than the ACV estimator, but cannot obtain the optimal variance reduction. 
#
#Before moving on, note that typically we want to choose both the number of low fidelity samples and the number of high-fidelity samples to minimize variance for a fixed budget. We will cover this in the next tutorial.
#
#Accelerated Approximate Control Variate Monte Carlo
#------------------------------------------------------
#Let :math:`K,L \leq M` with :math:`0 \leq L \leq K`. The ACV-KL estimator is
#
#.. math::
#
#   Q^{\mathrm{ACV-KL}}_{0,\mathcal{Z}}=Q_{0,\mathcal{Z}_{0}} + \sum_{\alpha=1}^K\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{0}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)+\sum_{\alpha=K+1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{L}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)
#
#Here we use a modified version of the MFMC sampling scheme shown in :ref:`acv_mf-kl-sample-allocation-kl-comparison`. Note the subtle difference between this sampling scheme and the one used for MFMC. We also note that the sample sets can be chosen in several ways, this is just one choice.
#
#.. list-table::
#
#   * - 
#       .. _mfmc-sample-allocation-kl-comparison:
#
#       .. figure:: ../../figures/mfmc.png
#          :width: 100%
#          :align: center
#
#          MFMC sampling strategy
#
#     - 
#       .. _acv_mf-kl-sample-allocation-kl-comparison:
#
#       .. figure:: ../../figures/acv_kl_22.png
#          :width: 100%
#          :align: center
#
#          ACV KL MF sampling strategy
#
#This estimator differs from the previous recursive estimators because the first two terms in correspond to an ACV-MF estimator with :math:`K` CVs and the last term adds a CV scheme to the ACV-MF estimator.
#
#The inclusion of the ACV-MF estimator enables the ACV-KL estimator to converge to the OCV estimator and the last term reduces the variance of :math:`\mu_{L}`, thereby accelerating convergence of the scheme. The optimal weights and variance reduction for the ACV-KL estimator are now provided.
#
#The matrix :math:`F` used to compute the covariances of the control variate discrepancies, i.e.
#
#.. math::
#
#   \covar{\V{\Delta}}{Q_0}&=N^{-1}\left(\mathrm{diag}\left(F\right)\circ \covar{\V{Q}_\mathrm{LF}}{Q_0}\right)\\
#   \covar{\V{\Delta}}{\V{\Delta}}&=N^{-1}\left(\covar{\V{Q}_\mathrm{LF}}{\V{Q}_\mathrm{LF}}\circ\mathrm{diag}\left(F\right)\right)\\
#
#can be found in  [GGEJJCP2020]_.
#
#Let us add the ACV KL estimator with the optimal choice of K and L to the previous plot. The optimal values can be obtained by a simple grid search, over all possible values of K and L, which returns the combination that results in the smallest estimator variance. This step only requires an estimate of the model covariance which is required for all ACV estimators.

cv_labels = [r'$\mathrm{OCV1}$',r'$\mathrm{OCV2}$',r'$\mathrm{OCV4}$']
cv_rsquared_funcs=[
    lambda cov: pya.get_control_variate_rsquared(cov[:2,:2]),
    lambda cov: pya.get_control_variate_rsquared(cov[:3,:3]),
    lambda cov: pya.get_control_variate_rsquared(cov)]
cv_gammas = [1-f(cov) for f in cv_rsquared_funcs]
xloc = -.35
for ii in range(len(cv_gammas)):
    plt.axhline(y=cv_gammas[ii],linestyle='--',c='k')
    plt.text(xloc, cv_gammas[ii]*1.1, cv_labels[ii],fontsize=16)
plt.axhline(y=1,linestyle='--',c='k')
plt.text(xloc,1,r'$\mathrm{MC}$',fontsize=16)

acv_labels = [r'$\mathrm{MLMC}$',r'$\mathrm{MFMC}$',r'$\mathrm{ACV}$-$\mathrm{MF}$',r'$\mathrm{ACV}$-$\mathrm{KL}$']
acv_rsquared_funcs = [
    pya.get_rsquared_mlmc,pya.get_rsquared_mfmc,
    partial(pya.get_rsquared_acv,
            get_discrepancy_covariances=pya.get_discrepancy_covariances_MF),
    pya.get_rsquared_acv_KL_best]

nplot_points = 20
acv_gammas = np.empty((nplot_points,len(acv_rsquared_funcs)))
for ii in range(nplot_points):
    nsample_ratios = [r*(2**ii) for r in nsample_ratios_base]
    acv_gammas[ii,:] = [1-f(cov,nsample_ratios) for f in acv_rsquared_funcs]
for ii in range(len(acv_labels)):
    plt.semilogy(np.arange(nplot_points),acv_gammas[:,ii],label=acv_labels[ii])
plt.legend()
plt.xlabel(r'$\log_2(r_i)-i$')
_ = plt.ylabel(r'$\mathrm{Variance}$ $\mathrm{reduction}$ $\mathrm{ratio}$ $\gamma$')

#%%
#The variance of the best ACV-KL still converges to the lowest possible variance. But its variance at small sample sizes is better than ACV-MF  and comparable to MLMC.
#
#Make note about how this scheme is useful when one model may have multiple discretizations.!!!!

#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `B. Peherstorfer, K. Willcox, M. Gunzburger, Optimal model management for multifidelity Monte Carlo estimation, SIAM J. Sci. Comput. 38 (2016) 59 A3163–A3194. <https://doi.org/10.1137/15M1046472>`_
#
#.. [CGSTCVS2011] `K.A. Cliffe, M.B. Giles, R. Scheichl, A.L. Teckentrup, Multilevel Monte Carlo methods and applications to elliptic PDEs with random coefficients, Comput. Vis. Sci. 14 (2011) <https://doi.org/10.1007/s00791-011-0160-x>`_
#
#.. [GOR2008] `M.B. Giles, Multilevel Monte Carlo path simulation, Oper. Res. 56 (2008) 607–617. <https://doi.org/10.1287/opre.1070.0496>`_
#
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, In press, (2020) <https://doi.org/10.1016/j.jcp.2020.109257>`_
