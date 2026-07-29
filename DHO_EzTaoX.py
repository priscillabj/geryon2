#!/usr/bin/env python3

#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os,glob, pickle
import jax
import jax.numpy as jnp

# you should always set this
jax.config.update("jax_enable_x64", True)

import numpyro, re
import numpyro.distributions as dist
from eztaox.fitter import random_search
from eztaox.kernels.quasisep import CARMA
from eztaox.models import UniVarModel
from numpyro.handlers import seed as numpyro_seed

import arviz as az
from numpyro.infer import MCMC, NUTS
import multiprocessing as mp

path = os.environ['HOME']#+'/Nextcloud/Doutorado/Forced_Phot/'
filenames = sorted(glob.glob(path+"/LC/corrlc_*.csv"))

p = 2
test_params = {"log_kernel_param": jnp.log(np.array([0.1, 1.1, 1.0, 3.0]))}
k = CARMA.init(
    jnp.exp(test_params["log_kernel_param"][:p]),
    jnp.exp(test_params["log_kernel_param"][p:]),
)


def radec_filename(i):
    
    match = re.search(r'lc_(\d+.\d+)_(-?\d+.\d+)', i)
    if match:
        regex_ra = match.group(1)
        regex_ra = float(regex_ra)
        regex_dec = match.group(2)
        regex_dec = float(regex_dec)

        return regex_ra, regex_dec

def UniVar_initSampler():
    # DHO Alpha & Beta parameters
    log_alpha = numpyro.sample(
        "log_alpha", dist.Uniform(low=-16.0, high=0.0).expand([2])
    )
    log_beta = numpyro.sample("log_beta", dist.Uniform(low=-10.0, high=2.0).expand([2]))

    log_kernel_param = jnp.hstack([log_alpha, log_beta])
    numpyro.deterministic("log_kernel_param", log_kernel_param)

    # mean
    mean = numpyro.sample("mean", dist.Uniform(low=-0.2, high=0.2))

    sample_params = {"log_kernel_param": log_kernel_param, "mean": mean}
    return sample_params

def UniVar_numpyro_model(t, yerr,band="g", y=None):
    log_alpha = numpyro.sample(
        "log_alpha", dist.Uniform(low=-16.0, high=0.0).expand([2])
    )
    log_beta = numpyro.sample("log_beta", dist.Uniform(low=-10.0, high=2.0).expand([2]))

    log_kernel_param = jnp.hstack([log_alpha, log_beta])
    numpyro.deterministic("log_kernel_param", log_kernel_param)

    # mean: use a normal prior for better convergence
    mean = numpyro.sample("mean", dist.Normal(0.0, 0.1))

    sample_params = {"log_kernel_param": log_kernel_param, "mean": mean}

    # the following is different from the initSampler
    zero_mean = False

    m = UniVarModel(t, y, yerr, k, zero_mean=zero_mean)
    m.sample(sample_params)


def DHO_X(lc):
    ts, ys, yerrs = {}, {}, {}
    ys_noisy = {}

    fp_dhofit = []
    zero_mean = False
    mcmc_seed = 0
    nSample = 10_000
    nBest = 10  # it seems like this number needs to be high

    #for lc in filenames[:5]:
    ztf = pd.read_csv(lc)
    fig,ax= plt.subplots(nrows=1, ncols=1,figsize=(12, 8))
    #for band,color in zip("gr",['mediumseagreen', 'firebrick']):
    for band in "g":
        ztf_b = ztf[ztf['filter']==f'ZTF_{band}']
        # add to dict
        #if band == 'g':
        ts[band] = ztf_b['mjd'].values - ztf[ztf['filter']==f'ZTF_g']['mjd'].values[0]
        #else:
        #    ts[band] = ztf_b['mjd'].values - 
        #ts[band] = ztf_b['mjd'].values
        #ys[band] = ztf_b['magtot_clr_corr']
        yerrs[band] = ztf_b['magunc_clr_corr'].values
        # add simulated photometric noise
        ys_noisy[band] = ztf_b['magtot_clr_corr'].values - ztf_b['magtot_clr_corr'].mean()
        regex_ra, regex_dec = radec_filename(lc)
        ax.errorbar(ts[band],ys_noisy[band],yerr=yerrs[band],
                    fmt='.',c='mediumseagreen',label=f'{band}-band')
        #ax.errorbar(ts[band],ys_noisy[band],yerr=yerrs[band],
        #             fmt='.',c=color,label=f'{band}-band')
    #print(ts)
    ax.legend()
    ax.set_ylabel('mag diff', fontsize=15)
    ax.set_xlabel('ts',fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    plt.savefig(path+f'/LC/eztaox_DHO/LC_input_{regex_ra}_{regex_dec}_{band}band.png')
        
    # define univar model
    m = UniVarModel(ts["g"], ys_noisy["g"], yerrs["g"], k, zero_mean=zero_mean)
    
    # # MLE Fitting
    model = m
    sampler = UniVar_initSampler
    fit_key = jax.random.PRNGKey(1)
    
    bestP, ll = random_search(model, UniVar_initSampler, fit_key, nSample, nBest)

    #EzTao best fit 
    #best_fit = carma_fit(ts[band], ys_noisy[band], yerrs[band], p, 1, n_opt=10)
    
    #print("EzTao DHO Best Fit Params (in natual log):")
    #print(np.log(np.hstack([best_fit[:2][::-1], best_fit[2:]])))
    print("EzTaoX MLE DHO Params (in natual log):")
    print(bestP["log_kernel_param"])

    #%%time
    nuts_kernel = NUTS(
        UniVar_numpyro_model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=numpyro.infer.init_to_sample,
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,
        num_samples=5000,
        num_chains=1,
        # progress_bar=False,
    )
    
    mcmc.run(jax.random.PRNGKey(mcmc_seed), ts[band], yerrs[band], y=ys_noisy[band])
    data = az.from_numpyro(mcmc)
    mcmc.print_summary()
    
    median_log_dho_alpha0 = np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=0), 50)
    median_log_dho_alpha1 = np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=1), 50)
    median_log_dho_beta0 = np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=0), 50)
    median_log_dho_beta1 = np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=1), 50)
    #median_log_amp_scale = np.percentile(data.posterior.log_amp_scale[0],50)
    
    var_dict = {'RA': regex_ra, 'DEC':regex_dec,'band':band,
            'median_log_dho_alpha0': median_log_dho_alpha0, 
            'log_alpha0_loerr': median_log_dho_alpha0 -np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=0),15.865),
            'log_alpha0_uperr':np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=0),84.135)- median_log_dho_alpha0 ,
            'median_log_dho_alpha1': median_log_dho_alpha1, 
            'log_alpha1_loerr': median_log_dho_alpha1-np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=1),15.865), 
            'log_alpha1_uperr':np.percentile(data.posterior.log_alpha.isel(log_alpha_dim_0=1),84.135)-median_log_dho_alpha1,
            'median_log_dho_beta0': median_log_dho_beta0, 
            'log_beta0_loerr': median_log_dho_beta0-np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=0),15.865),
            'log_beta0_uperr':np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=0),84.135)-median_log_dho_beta0,
            'median_log_dho_beta1': median_log_dho_beta1, 
            'log_beta1_loerr': median_log_dho_beta1-np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=1),15.865),
            'log_beta1_uperr':np.percentile(data.posterior.log_beta.isel(log_beta_dim_0=1),84.135)-median_log_dho_beta1,
            }

        #fp_dhofit.append(var_dict)

    az.plot_trace(data, var_names=["log_alpha", "log_beta", "mean"])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(path+f'/LC/eztaox_DHO/param_iter_{regex_ra}_{regex_dec}_{band}band.png', dpi=300, bbox_inches='tight')
    
    az.plot_pair(data, var_names=["log_alpha", "log_beta", "mean"])
    plt.savefig(path+f'/LC/eztaox_DHO/posterior_{regex_ra}_{regex_dec}_{band}band.png', dpi=300, bbox_inches='tight')

        #plt.show()
    
    return var_dict

def main_parallel():
    fp_dhofit=[]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(DHO_X, filenames[:2])
        fp_dhofit.append(results)
        #Save to a pickle file

    with open(path+f'LC/batch/calibrated/eztaox_DHO/dho_mcmc_par2.pkl', "wb") as f: # 'wb' for write binary
        pickle.dump(fp_dhofit, f)

    return fp_dhofit

if __name__ == "__main__":
    results = main_parallel()
