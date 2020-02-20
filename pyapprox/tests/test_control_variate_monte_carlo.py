import unittest
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.configure_plots import *
from pyapprox.control_variate_monte_carlo import *
from scipy.stats import uniform,norm,lognorm
from functools import partial

class TunableModelEnsemble(object):
    def __init__(self,theta1,shifts=None):
        """
        Parameters
        ----------
        theta0 : float
            Angle controling 
        Notes
        -----
        The choice of A0, A1, A2 here results in unit variance for each model
        """
        self.A0 = np.sqrt(11)
        self.A1 = np.sqrt(7)
        self.A2 = np.sqrt(3)
        self.nmodels=3
        self.theta0=np.pi/2
        self.theta1=theta1
        self.theta2=np.pi/6
        assert self.theta0>self.theta1 and self.theta1>self.theta2
        self.shifts=shifts
        if self.shifts is None:
            self.shifts = [0,0]
        assert len(self.shifts)==2
        self.models = [self.m0,self.m1,self.m2]
        
    def m0(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A0*(np.cos(self.theta0) * x**5 + np.sin(self.theta0) * y**5))[:,np.newaxis]
    
    def m1(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A1*(np.cos(self.theta1) * x**3 + np.sin(self.theta1) * y**3)+self.shifts[0])[:,np.newaxis]
    
    def m2(self,samples):
        assert samples.shape[0]==2
        x,y=samples[0,:],samples[1,:]
        return (self.A2*(np.cos(self.theta2) * x + np.sin(self.theta2) * y)+self.shifts[1])[:,np.newaxis]

    def get_covariance_matrix(self):
        cov = np.eye(self.nmodels)
        cov[0, 1] = self.A0*self.A1/9*(np.sin(self.theta0)*np.sin(
            self.theta1)+np.cos(self.theta0)*np.cos(self.theta1))
        cov[1, 0] = cov[0,1]
        cov[0, 2] = self.A0*self.A2/7*(np.sin(self.theta0)*np.sin(
            self.theta2)+np.cos(self.theta0)*np.cos(self.theta2))
        cov[2, 0] = cov[0, 2]
        cov[1, 2] = self.A1*self.A2/5*(
            np.sin(self.theta1)*np.sin(self.theta2)+np.cos(
                self.theta1)*np.cos(self.theta2))
        cov[2, 1] = cov[1,2]
        return cov


class ShortColumnModelEnsemble(object):
    def __init__(self):
        self.nmodels=5
        self.nvars=5
        self.functions = [self.m0,self.m1,self.m2,self.m3,self.m4]
    
    def extract_variables(self,samples):
        assert samples.shape[0]==5
        b = samples[0,:]
        h = samples[1,:]
        P = samples[2,:]
        M = samples[3,:]
        Y = samples[4,:]
        return b,h,P,M,Y

    def m0(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - 4*M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:,np.newaxis]
    
    def m1(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - 3.8*M/(b*(h**2)*Y) - ((P*(1 + (M-2000)/4000))/(b*h*Y))**2)[:,np.newaxis]

    def m2(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P/(b*h*Y))**2)[:,np.newaxis]

    def m3(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(b*h*Y))**2)[:,np.newaxis]

    def m4(self,samples):
        b,h,P,M,Y = self.extract_variables(samples)
        return (1 - M/(b*(h**2)*Y) - (P*(1 + M)/(h*Y))**2)[:,np.newaxis]

class TestCVMC(unittest.TestCase):

    def test_standardize_sample_ratios(self):
        nhf_samples, nsample_ratios = 9.8, [2.1]
        nhf_samples_std, nsample_ratios_std = pya.standardize_sample_ratios(
            nhf_samples,nsample_ratios)
        assert np.allclose(nhf_samples_std,10)
        assert np.allclose(nsample_ratios_std,[2])

    def test_MLMC_tunable_example(self):
        example = TunableModelEnsemble(np.pi/4)
        generate_samples = lambda nn: np.random.uniform(-1,1,(2,nn))
        model_emsemble = pya.ModelEnsemble([example.m0,example.m1,example.m2])
        #costs = np.array([1.0, 1.0/100, 1.0/100/100])
        cov, samples , values= pya.estimate_model_ensemble_covariance(
            int(1e3),generate_samples,model_emsemble)
        import seaborn as sns
        from pandas import DataFrame
        df = DataFrame(
            index=np.arange(values.shape[0]),
            data=dict([(r'$z_%d$'%ii,values[:,ii])
                       for ii in range(values.shape[1])]))
        # heatmap does not currently work with matplotlib 3.1.1 downgrade to
        # 3.1.0 using pip install matplotlib==3.1.0
        #sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidth=0.5)
        exact_cov = example.get_covariance_matrix()
        exact_cor = pya.get_correlation_from_covariance(exact_cov)
        print(exact_cor)
        print(df.corr())
        #plt.tight_layout()
        #plt.show()

        theta1 = np.linspace(example.theta2*1.05,example.theta0*0.95,5)
        covs = []
        var_reds = []
        for th1 in theta1:
            example.theta1=th1
            covs.append(example.get_covariance_matrix())
            OCV_var_red = pya.get_variance_reduction(
                pya.get_control_variate_rsquared,covs[-1],None)
            # use model with largest covariance with high fidelity model
            idx = [0,np.argmax(covs[-1][0,1:])+1]
            assert idx == [0,1] #it will always be the first model
            OCV1_var_red = pya.get_variance_reduction(
                pya.get_control_variate_rsquared,covs[-1][np.ix_(idx,idx)],None)
            var_reds.append([OCV_var_red,OCV1_var_red])
        covs = np.array(covs)
        var_reds = np.array(var_reds)

        fig,axs = plt.subplots(1,2,figsize=(2*8,6))
        for ii,jj, in [[0,1],[0,2],[1,2]]:
            axs[0].plot(theta1,covs[:,ii,jj],'o-',
                        label=r'$\rho_{%d%d}$'%(ii,jj))
        axs[1].plot(theta1,var_reds[:,0],'o-',label=r'$\textrm{OCV}$')
        axs[1].plot(theta1,var_reds[:,1],'o-',label=r'$\textrm{OCV1}$')
        axs[1].plot(theta1,var_reds[:,0]/var_reds[:,1],'o-',
                    label=r'$\textrm{OCV/OCV1}$')
        axs[0].set_xlabel(r'$\theta_1$')
        axs[0].set_ylabel(r'$\textrm{Correlation}$')
        axs[1].set_xlabel(r'$\theta_1$')
        axs[1].set_ylabel(r'$\textrm{Variance reduction ratio} \ \gamma$')
        axs[0].legend()
        axs[1].legend()
        #plt.show()

        print('####')
        target_cost = 100
        cost_ratio = 10
        costs = np.array([1,1/cost_ratio,1/cost_ratio**2])
        example.theta0=1.4/0.95
        example.theta2=0.6/1.05
        theta1 = np.linspace(example.theta2*1.05,example.theta0*0.95,5)
        #allocate = pya.allocate_samples_mlmc
        #get_rsquared = pya.get_rsquared_mlmc
        #allocate = pya.allocate_samples_mfmc
        #get_rsquared = pya.get_rsquared_mfmc
        allocate = pya.allocate_samples_acv
        get_rsquared = pya.get_rsquared_acv
        for th1 in theta1:
            example.theta1=th1
            cov = example.get_covariance_matrix()
            nhf_samples, nsample_ratios, log10_var = allocate(
                cov,costs,target_cost)
            var_red = pya.get_variance_reduction(
                get_rsquared,cov,nsample_ratios)
            assert np.allclose(var_red,(10**log10_var)/cov[0,0]*nhf_samples)
            print(var_red)
            assert False

    def test_mlmc_sample_allocation(self):
        # The following will give mlmc with unit variance
        # and discrepancy variances 1,4,4
        target_cost = 81
        cov = np.asarray([[1.00,0.50,0.25],
                          [0.50,1.00,0.50],
                          [0.25,0.50,4.00]])
        # ensure cov is positive definite
        np.linalg.cholesky(cov)
        #print(np.linalg.inv(cov))
        costs = [6,3,1]
        nmodels = len(costs)
        nhf_samples,nsample_ratios, log10_var = pya.allocate_samples_mlmc(
            cov, costs, target_cost)
        assert np.allclose(10**log10_var,1)
        nsamples = np.concatenate([[1],nsample_ratios])*nhf_samples
        lamda = 9
        nsamples_discrepancy = 9*np.sqrt(np.asarray([1/(6+3),4/(3+1),4]))
        nsamples_true = [
            nsamples_discrepancy[0],nsamples_discrepancy[:2].sum(),
            nsamples_discrepancy[1:3].sum()]
        assert np.allclose(nsamples,nsamples_true)

    def test_standardize_sample_ratios(self):
        nhf_samples,nsample_ratios = 10,[2.19,3.32]
        std_nhf_samples, std_nsample_ratios = pya.standardize_sample_ratios(
            nhf_samples,nsample_ratios)
        assert np.allclose(std_nsample_ratios,[2.1,3.3])

    def test_generate_samples_and_values_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = pya.ModelEnsemble(
            [functions.m0,functions.m1,functions.m2])
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        
        nhf_samples = 10
        nsample_ratios = [2,4]
        samples,values =\
            pya.generate_samples_and_values_mfmc(
                nhf_samples,nsample_ratios,model_ensemble,generate_samples)
    
        for jj in range(1,len(samples)):
            assert samples[jj][1].shape[1]==nsample_ratios[jj-1]*nhf_samples
            idx=1
            if jj==1:
                idx=0
            assert np.allclose(samples[jj][0],samples[jj-1][idx])

        npilot_samples = int(1e4)
        cov, samples, weights = pya.estimate_model_ensemble_covariance(
            npilot_samples,generate_samples,model_ensemble)
            
        M = len(nsample_ratios) # number of lower fidelity models
        ntrials=int(1e3)
        means = np.empty((ntrials,2))
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        for ii in range(ntrials):
            samples,values =\
               generate_samples_and_values_mfmc(
                    nhf_samples,nsample_ratios,model_ensemble,generate_samples)
            # compute mean using only hf data
            hf_mean = values[0][0].mean()
            means[ii,0]= hf_mean
            # compute ACV mean
            eta = get_mfmc_control_variate_weights(cov)
            means[ii:,1] = compute_control_variate_mean_estimate(
                eta,values)

        true_var_reduction = 1-pya.get_rsquared_mfmc(
            cov[:M+1,:M+1],nsample_ratios)
        numerical_var_reduction = means[:,1].var(axis=0)/means[:,0].var(axis=0)
        assert np.allclose(true_var_reduction,numerical_var_reduction,atol=4e-2)

    def test_rsquared_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = pya.ModelEnsemble(
            [functions.m0,functions.m3,functions.m4])
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        npilot_samples = int(1e4)
        pilot_samples = generate_samples(npilot_samples)
        config_vars = np.arange(model_ensemble.nmodels)[np.newaxis,:]
        pilot_samples = pya.get_all_sample_combinations(
            pilot_samples,config_vars)
        pilot_values = model_ensemble(pilot_samples)
        pilot_values = np.reshape(
            pilot_values,(npilot_samples,model_ensemble.nmodels))
        cov = np.cov(pilot_values,rowvar=False)
        
        nhf_samples = 10
        nsample_ratios = np.asarray([2,4])

        nsamples_per_model = np.concatenate(
            [[nhf_samples],nsample_ratios*nhf_samples])

        eta = pya.get_mfmc_control_variate_weights(cov)
        cor = pya.get_correlation_from_covariance(cov)
        var_mfmc = cov[0,0]/nsamples_per_model[0]
        for k in range(1,model_ensemble.nmodels):
            var_mfmc += (1/nsamples_per_model[k-1]-1/nsamples_per_model[k])*(
                eta[k-1]**2*cov[k,k]+2*eta[k-1]*cor[0,k]*np.sqrt(
                    cov[0,0]*cov[k,k]))
            
        assert np.allclose(var_mfmc/cov[0,0]*nhf_samples,
                           1-pya.get_rsquared_mfmc(cov,nsample_ratios))
        
    def test_generate_samples_and_values_mlmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = pya.ModelEnsemble(
            [functions.m0,functions.m1,functions.m2])
        univariate_variables = [
            uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
            lognorm(s=0.5,scale=np.exp(5))]
        variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
            
        npilot_samples = int(1e4)
        cov, samples, weights = pya.estimate_model_ensemble_covariance(
            npilot_samples,generate_samples,model_ensemble)

        target_cost = int(1e4)
        costs = np.asarray([100, 50, 5])
        nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
            cov, costs, target_cost, nhf_samples_fixed=10)[:2]
            
        M = len(nsample_ratios) # number of lower fidelity models
        ntrials=int(1e3)
        means = np.empty((ntrials,2))
        generate_samples=partial(
            pya.generate_independent_random_samples,variable)
        for ii in range(ntrials):
            samples,values =\
               generate_samples_and_values_mlmc(
                    nhf_samples,nsample_ratios,model_ensemble,generate_samples)
            # compute mean using only hf data
            hf_mean = values[0][0].mean()
            means[ii,0]= hf_mean
            # compute ACV mean
            eta = get_mlmc_control_variate_weights(M+1)
            means[ii:,1] = compute_control_variate_mean_estimate(
                eta,values)

        true_var_reduction = 1-pya.get_rsquared_mlmc(
            cov[:M+1,:M+1],nsample_ratios)
        numerical_var_reduction = means[:,1].var(axis=0)/means[:,0].var(axis=0)
        assert np.allclose(true_var_reduction,numerical_var_reduction,atol=1e-2)

    def test_MFMC_fixed_nhf_samples(self):
        msg = 'Add test where compute opitmal num samples without fixing nhf_samples. Then compute the optimal num samples when fixing nhf_samples t0 the value previously computed and check the values for the low fidelity models returned are the same as when nhf_samples wass not fixed. Also repeat for MLMC'
        msg += '\nAlso add tests using ACV2 optimizer to find optima for MLMC and MFMC and compare solution to known exact answer.'
        raise Exception(msg)

    def test_CVMC(self):
        pass

    def test_allocate_samples_acv(self):
        cov = np.asarray([[1.00,0.50,0.25],
                          [0.50,1.00,0.50],
                          [0.25,0.50,4.00]])
        
        costs = np.array([4, 2, 1])
        
        target_cost = 200

        #estimator = ACV2(cov)
        #estimator = MFMC(cov)
        estimator = MLMC(cov)

        nhf_samples_exact, nsample_ratios_exact =  allocate_samples_mlmc(
            cov, costs,target_cost,nhf_samples_fixed=None,standardize=False)[:2]
        
        assert np.allclose(nhf_samples_exact*costs[0]+(nsample_ratios_exact*nhf_samples_exact).dot(costs[1:]),target_cost,rtol=1e-12)

        # NOTE check my hypothesis that gradient may not be zero at exact optima because I am using an approximate covariance
        
        #print(cov)
        jacobian = partial(
            acv_sample_allocation_jacobian_all,estimator)
        x0 = np.concatenate([[nhf_samples_exact],nsample_ratios_exact])
        print('optimal x',x0)
        jac = jacobian(x0)
        print('jac',jac)
        objective = partial(acv_sample_allocation_objective_all,estimator)
        errors = pya.check_gradients(
            objective,jacobian,x0)
        from scipy.optimize import approx_fprime
        print(approx_fprime(x0,objective,1e-9))
        # for mlmc ( and perhaps other estimators) need to include a lagrangian in objective
        
        assert np.allclose(jac,0*jac)
        
        assert False

        
        nhf_samples,nsample_ratios,var=allocate_samples_acv(
            cov, costs, target_cost, estimator, nhf_samples_fixed=None)

        print(estimator.variance_reduction(nsample_ratios).item())
        print(estimator.variance_reduction(nsample_ratios_exact).item())

        

        print(nhf_samples_exact,nhf_samples)
        print(nsample_ratios_exact,nsample_ratios)

        

    def test_ACVMC_sample_allocation(self):
        np.random.seed(1)
        ncv = 2
        matr = np.random.randn(3,3)
        cov_should = np.dot(matr, matr.T)
        L = np.linalg.cholesky(cov_should)
        samp = np.dot(np.random.randn(100000, 3),L.T)
        cov = np.cov(samp, rowvar=False)
        cor = torch.tensor(np.corrcoef(samp, rowvar=False), dtype=torch.float)
        
        costs = [4, 2, 1]
        
        target_cost = 20

        nhf = 2
        nhf,ratios,var=allocate_samples_acv(
            cov, costs, target_cost, nhf_samples_fixed=nhf)
        print("opt = ", nhf, ratios, var)

        nhf,ratios,var=allocate_samples_acv(
            cov, costs, target_cost, nhf_samples_fixed=None)
        print("opt = ", nhf, ratios, var)


    def test_ACVMC_objective_jacobian(self):
        
        ncv = 2
        matr = np.random.randn(3,3)
        cov_should = np.dot(matr, matr.T)
        L = np.linalg.cholesky(cov_should)
        samp = np.dot(np.random.randn(100000, 3),L.T)
        cov = np.cov(samp, rowvar=False)
        cor = torch.tensor(np.corrcoef(samp, rowvar=False), dtype=torch.float)
        
        costs = [4, 2, 1]
        
        target_cost = 20

        nhf_samples, nsample_ratios =  pya.allocate_samples_mlmc(
            cov, costs, target_cost, nhf_samples_fixed=2)[:2]
        print(nsample_ratios)
        from functools import partial
        estimator = ACV2(cov)
        errors = pya.check_gradients(
            partial(acv_sample_allocation_objective,estimator),
            partial(acv_sample_allocation_jacobian,estimator),
            nsample_ratios)
        print(errors.min())
        assert errors.min()<1e-8

    
        
    
if __name__== "__main__":    
    cvmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestCVMC)
    unittest.TextTestRunner(verbosity=2).run(cvmc_test_suite)

