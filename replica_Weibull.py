import matplotlib.pyplot as plt
from scipy.integrate import quad, quad_vec
from scipy.stats import gumbel_l
from scipy.stats import norm
import numpy as np
import sympy
import time
K = sympy.EulerGamma.evalf()
import multiprocessing
import argparse


# denoisers for beta: Gaussian and Bernoulli-Gaussian
def den_beta_gaussian(r1, gam1, sigma):
    return r1 * sigma / (sigma + 1.0/gam1)

def den_beta(r1, gam1, la, sigma):
    """
    This function returns the conditional expectation of the coefficients beta given the noisy estimate r1 
    and noise precision gam1. The prior placed on beta is of spike and slab form and is described by sparsity 
    rate (1-la) and variance of the slab part called sigma
    """
    A = (1-la) * norm.pdf(r1, loc=0, scale=np.sqrt(1.0/gam1)) # scale = standard deviation
    B = la * norm.pdf(r1, loc=0, scale=np.sqrt(sigma + 1.0/gam1))
    ratio = gam1 * r1 / (gam1 + 1/sigma) * B / (A + B)
    return ratio

# functions for evaulation of gPout(v, logy) = E[W|logY, V]
def den_w_integrand_expe(w, mu, alpha, v, logy, rho, q):
    return w * gumbel_l.pdf(logy, loc= mu + (np.sqrt(q)*v + np.sqrt(rho-q)*w) + K/alpha, scale=1/alpha) * \
        norm.pdf(w, loc=0, scale=1) * norm.pdf(v, loc=0, scale=1) 

def den_w_integrand_prob_mass(w, mu, alpha, v, logy, rho, q):
    return gumbel_l.pdf(logy, loc=mu + (np.sqrt(q)*v + np.sqrt(rho-q)*w) + K/alpha, scale=1/alpha) * \
        norm.pdf(w, loc=0, scale=1) * norm.pdf(v, loc=0, scale=1) 

def gPout_Weibull(mu, alpha, v, logy, rho, q, int_bound):
    return quad(den_w_integrand_expe, -int_bound, int_bound, args=(mu,alpha,v,logy,rho,q))[0] / quad(den_w_integrand_prob_mass, -int_bound, int_bound, args=(mu,alpha,v,logy,rho,q))[0]

# def new_r_integrand(w, v, logy, rho, q, mu, alpha, int_bound):
#     gPout = gPout_Weibull(mu, alpha, v, logy, rho, q, int_bound)
#     return gPout * gPout * norm.pdf(w, loc=0, scale=1) * norm.pdf(v, loc=0, scale=1) * gumbel_l.pdf(logy, loc=mu + (v + np.sqrt(rho-q)*w) + K/alpha, scale=1/alpha)

# vectorized version of gPout_Weibull
def den_w_integrand_expe_vec(w, mu, alpha, v, logy, rho, q):
    # v, logy are vectors that induce differences between components of function output
    return w * gumbel_l.pdf(logy, loc= mu + (np.sqrt(q)*v + np.sqrt(rho-q)*w) + K/alpha, scale=1/alpha) * \
        norm.pdf(w, loc=0, scale=1) * norm.pdf(v, loc=0, scale=1) 

def den_w_integrand_prob_mass_vec(w, mu, alpha, v, logy, rho, q):
    return gumbel_l.pdf(logy, loc= mu + (np.sqrt(q)*v + np.sqrt(rho-q)*w) + K/alpha, scale=1/alpha) * \
        norm.pdf(w, loc=0, scale=1) * norm.pdf(v, loc=0, scale=1) 

def gPout_Weibull_vec(mu, alpha, v, logy, rho, q, int_bound, num_workers):
   lim = 10000
   epsrel = 1e-05
   epsabs = 1e-50
   quadrature='gk21'
   cache_size = 5000000000
   return np.array( quad_vec(den_w_integrand_expe_vec, -int_bound, int_bound, args=(mu,alpha,v,logy,rho,q), cache_size = cache_size, workers=num_workers, limit=lim, quadrature=quadrature, epsrel = epsrel, epsabs = epsabs)[0] / \
        quad_vec(den_w_integrand_prob_mass_vec, -int_bound, int_bound, args=(mu,alpha,v,logy,rho,q), cache_size = cache_size, workers=num_workers, limit=lim, epsrel=epsrel, quadrature=quadrature, epsabs = epsabs)[0] )

# function that updates signal strength r
def new_r_sampler(rho, q, mu, alpha, num_samples, num_workers):
    int_bound = 4
    w = norm.rvs(loc=0, scale=1, size=num_samples)
    v = norm.rvs(loc=0, scale=1, size=num_samples)
    t0 = time.time()
    logy = np.float64( gumbel_l.rvs(loc=mu + (np.sqrt(q)*v + np.sqrt(rho-q)*w) + K/alpha, scale = 1/alpha) )
    t1 = time.time()
    total_time_sampling = t1-t0
    print(f"total time for sampling inside _new_r_ is {total_time_sampling / 60} mins")
    gPout_evals = np.zeros(num_samples)
    # for i in range(num_samples):
    #     gPout_evals[i] = gPout_Weibull(mu, alpha, v[i], logy[i], rho, q, int_bound)
    t0 = time.time()
    gPout_evals = gPout_Weibull_vec(mu, alpha, v, logy, rho, q, int_bound, num_workers)
    t1 = time.time()
    total_time_gPout = t1-t0
    print(f"total time for sampling inside _new_r_ is {total_time_gPout / 60} mins")
    return np.mean(gPout_evals**2)

def new_r(rho, q, mu, alpha, ratio, num_samples, num_workers):
    # I_WVY = nquad(new_r_integrand, [[-3, 3],[-3, 3],[mu + K/alpha -int_bound, mu + K/alpha + int_bound]], args=(rho,q,mu,alpha,int_bound))[0]
    I_WVY = new_r_sampler(rho, q, mu, alpha, num_samples, num_workers)
    return ratio * I_WVY / (rho - q)

# def new_q_integrand(r1, r, la, sigma):
#     gP0 = den_beta(r1, r, la, sigma)
#     return gP0 * gP0 * ( (1-la) * norm.pdf(r1, loc=0, scale=1/np.sqrt(r)) + la * norm.pdf(r1, loc=0, scale=1/np.sqrt(sigma + r)) )

# function that updates asymptotic correlation q
def new_q_sampler(r, la, sigma, num_samples):
    x0 = norm.rvs(loc=0, scale=np.sqrt(sigma), size=num_samples)
    x0 *= np.random.binomial(1, p=la, size=num_samples)
    r1 = norm.rvs(loc=x0, scale=1.0/np.sqrt(r))
    gP0 = np.zeros(num_samples)
    for i in range(num_samples):
        # gP0[i] = den_beta(r1[i], r, la, sigma)
        gP0[i] = den_beta_gaussian(r1[i], r, sigma)
    return np.mean(gP0**2)

def new_q(r, la, sigma, num_samples):
    # stat_sig = 6
    # return quad(new_q_integrand, -stat_sig*np.sqrt(sigma+r), stat_sig*np.sqrt(sigma+r), args=(r, la, sigma))[0]
    return new_q_sampler(r, la, sigma, num_samples)

def replica_limit(q_init, rho, mu, alpha, ratio, la, sigma, rel_thr=1e-2, maxiter=16, num_samples=10000, num_workers=-1):
    print(f"Using {num_samples} samples in the MCMC approximation..")
    print(f"Max number of iterations is {maxiter}..")
    print(f"Using {num_workers} workers..")
    qs = [q_init]
    q = q_init
    for i in range(maxiter):
        print(f"\n * ITERATION {i} *")
        r = new_r(rho, q, mu, alpha, ratio, num_samples, num_workers)
        print(f"r={r:.9f}")
        qprev = q
        q = new_q(r, la, sigma, num_samples)
        print(f"q={q:.9f}")
        qs.append(q)
        if np.absolute(q-qprev) / qprev < rel_thr:
            break
    return qs 


cpu_count = multiprocessing.cpu_count()
print(f"Using {cpu_count} cores...")

parser = argparse.ArgumentParser()
parser.add_argument("-mu", "--mu", help = "Location parameter of the Weibull model")
parser.add_argument("-alpha", "--alpha", help = "Scale parameter of the Weibull model")
parser.add_argument("-ratio", "--ratio", help = "Ratio of number of samples and number of features")
parser.add_argument("-sigma", "--sigma", help = "Variance of slab part of the prior on the genetic signal")
parser.add_argument("-n", "--n", help = "Number of samples")
parser.add_argument("-la", "--la", help = "Non-sparsity rate")
parser.add_argument("-num_samples", "--num_samples", help = "Number of samples")
parser.add_argument("-q_init_factor", "--q-init-factor", help = "Factor for initialization of q", default=0.95)
parser.add_argument("-num_workers", "--num-workers", help = "Number of workers", default=4)
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
args = parser.parse_args()

# Input parameters
mu = float(args.mu)
alpha = float(args.alpha)
ratio = float(args.ratio)
sigma = float(args.sigma)
n = int(args.n)
# sigma = sigma * n
la = float(args.la)
rho = la * sigma
num_samples = int(args.num_samples)
q_init = float(args.q_init_factor) * rho
num_workers = int(args.num_workers)
iterations = int(args.iterations)
out_dir = args.out_dir

print(f"Parameters: \nmu = {mu} \nalpha = {alpha} \nratio = {ratio} \nsigma = {sigma} \nla = {la} \nn = {n} \nnum_samples = {num_samples} \nq_init = {q_init}")

qs = replica_limit(q_init, rho, mu, alpha, ratio, la, sigma, rel_thr=1e-3, maxiter=iterations, num_samples=num_samples, num_workers = num_workers)
print(qs)

fig = plt.gcf()
qs = np.array(qs)
plt.plot(range(len(qs)), qs, 'bo-')
plt.xlabel("Iterations")
plt.ylabel("Asymptotic correlation")
plt.show()
fig.savefig(out_dir + '/mu_' + str(mu) + '_alpha_' + str(alpha) + \
            '_ratio_' + str(ratio) + '_sigma_' + str(sigma) + '_la_' + str(la) + '_n_' + str(n) + '_num_samples_' + \
                'num_samples_' + str(num_samples) + '_qinit_' + str(q_init) + '.png')

np.save(out_dir + '/mu_' + str(mu) + '_alpha_' + str(alpha) + \
            '_ratio_' + str(ratio) + '_sigma_' + str(sigma) + '_la_' + str(la) + '_n_' + str(n) + '_num_samples_' + \
                'num_samples_' + str(num_samples) + '_qinit_' + str(q_init) + '.npy', qs)