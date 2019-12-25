# Code for slice sampling MCMC for an arbitrary unimodal Gaussian as the sampling distribution P(x)
#     Naman Deep Singh  25-12-2019


# Inverse of the distribution is used in this case

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
mu, var = 30, 85

def prob(x):
    return st.norm.pdf(x, loc = mu, scale = var)

def inverse_prob(p):
    ###  Returns the range to sample from  after computing the inverse of the probability density
    val = np.sqrt(-2*var**2*np.log(p*var*np.sqrt(2*np.pi)))
    return mu - val, mu + val

def slice_sample(num):
    samples = np.zeros(num)
    x = 0
    for i in range(num):
        den = prob(x)
        u = np.random.uniform(0, den)
        low_r, hig_r = inverse_prob(u)
        x = np.random.uniform(low_r, hig_r)
        samples[i] = x

    return samples

samples = slice_sample(num = 1500)
sns.distplot(samples, norm_hist = True)
x = np.arange(-300, 300)
plt.plot(x, prob(x))
plt.show()
