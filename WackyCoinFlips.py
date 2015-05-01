"""
Beta-binomial MLM for fitting results from flipping wacky coins in BDA Lab 14.

Created May 1, 2015 by Tom Loredo
"""
import glob

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import *
from scipy import stats

from stanfitter import StanFitter

# try:
#     import myplot
#     from myplot import close_all, csavefig
#     #myplot.tex_on()
#     csavefig.save = False
# except ImportError:
#     pass


ion()


class CoinCase:

    def __init__(self, ctype, flips):
        """
        Initialize a coin case from the coin type, `ctype` (an int), and
        a string of binary flip results, `flips`.
        """
        self.ctype = ctype
        self.flips = flips
        self.n_flips = len(flips)
        self.n_heads = flips.count('1')


# Create simulated data, for testing data file reading.
if False:
    ctypes = [0,0,0,  1,1,1, 2,2,2,2,  3,3,3,3]
    alphas_sim = [.3,.4,.35,  .5,.6,.5,  .7,.8,.75,.77,  .6,.62,.61,.59]
    flips_sim = [50,30,40,  50,20,40,  50,30,40,50,  50,100,40,20]
    n_coins = len(alphas_sim)
    for i in range(n_coins):
        ctype = ctypes[i]
        alpha = alphas_sim[i]
        # Collect 0/1 flip results in a string.
        s = ''
        for nf in range(flips_sim[i]):
            if np.random.rand() < alpha:
                s += '1'
            else:
                s += '0'
        # Write to a separate file for each coin.
        with open('tjl-{}-sim.txt'.format(i), 'w') as ofile:
            ofile.write('{}\n'.format(ctype))
            if len(s) <= 30:  # to test multi-line input
                ofile.write(s + '\n')
            else:
                ofile.write(s[:30] + '\n')
                ofile.write(s[30:] + '\n')


# Read in data.
files = glob.glob('*-sim.txt')
cases = []
for name in files:
    with open(name) as dfile:
        ctype = int(dfile.readline())
        s = ''
        lines = dfile.readlines()
        for line in lines:
            s += line.strip()
    cases.append(CoinCase(ctype, s))

# Cluster data by coin type.
clusters = {}  # lists of results, keyed by coin type
for case in cases:
    if not clusters.has_key(case.ctype):
        clusters[case.ctype] = []
    clusters[case.ctype].append( (case.n_flips, case.n_heads) )

# Convert to arrays:
for ctype in clusters:
    clusters[ctype] = array(clusters[ctype])


# Stan model beta-binomial (for a single cluster):

bb_code = """
data {
    int<lower=0> n_coins;
    int<lower=0> n_flips[n_coins];
    int<lower=0> n_heads[n_coins];
}

parameters {
    real<lower=0., upper=1.> alpha[n_coins];  // probabilities for heads
    real<lower=0.> mu_a;  // pop'n mean alpha
    real<lower=0.> sigma_a;  // pop'n std dev'n
}

transformed parameters {
    real<lower=0.> a;  // gamma dist'n shape param
    real<lower=0.> b;  // gamma dist'n inv. scale
    a <- mu_a*mu_a / (sigma_a*sigma_a);
    b <- mu_a / (sigma_a*sigma_a);
}

model {
    mu_a ~ uniform(0., 1.);
    sigma_a ~ gamma(1., 2.);  // mean .5, std dev .5
    for (i in 1:n_coins) {
        alpha[i] ~ gamma(a, b);
        n_heads[i] ~ binomial(n_flips[i], alpha[i]);
    }
}
"""

fitter = StanFitter(bb_code)


def fit_cluster(alpha_ax, hyper_ax, cluster):
    """
    Peform a fit to the cluster data in array `cluster` and plot alpha
    marginals on `alpha_ax` and pop'n params on `hyper_ax`.
    """
    n_flips = cluster[:,0]
    n_heads = cluster[:,1]
    data = dict(n_coins=cluster.shape[0], n_flips=cluster[:,0],
        n_heads=cluster[:,1])
    fit = fitter.sample(data=data, n_iter=1000, n_chains=4)

    for i, alpha in enumerate(fit.alpha):
        lbl = '%d' % i
        alpha_ax.hist(alpha.thinned, bins=20, label=lbl, alpha=.3)
    alpha_ax.legend(markerscale=.5, fontsize='small', framealpha=.5)

    hyper_ax.scatter(fit.mu_a.thinned, fit.sigma_a.thinned,
        linewidths=0, alpha=.2)
    xlim(0., 1.)
    ylim(0., 1.)

    return fit


fig = figure(figsize=(15,6))
rc('xtick.major', pad=6)
rc('xtick', labelsize=12)
rc('ytick.major', pad=6)
rc('ytick', labelsize=12)
subplots_adjust(left=.07, top=.95, right=.97, hspace=.2)
n_col = 4
for i, ctype in enumerate(clusters):
    col = i+1
    top = subplot(2, n_col, col)
    bot = subplot(2, n_col, n_col+col)
    fit = fit_cluster(top, bot, clusters[ctype])
    top.set_title('{}'.format(ctype))

    if col == 1:
        top.set_xlabel(r'$\alpha_i$')
        top.set_ylabel('Counts')
        bot.set_xlabel(r'$\mu_f$')
        bot.set_ylabel(r'$\sigma_f$')
    else:
        top.set_xlabel(r'$\alpha_i$')
        bot.set_xlabel(r'$\mu_f$')
