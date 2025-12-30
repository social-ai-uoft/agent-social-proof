import math
from typing import List, Tuple, Optional, Dict
import numpy as np
import statsmodels.api as sm
from fontTools.misc.cython import returns

# scipy or scikit learn for the ols regression

# Commit to the GitHub project

rng = np.random.default_rng(42)

class Agent:
    def __init__(self):
        # Beliefs about willingness to accept
        # utility cost
        self.c0 = c0  # cost per dollar offered
        self.c2 = c2  # extra cost per computer sample
        self.c3 = c3  # reward if accepted

        # sample collected
        self.social_prices = []
        self.computer_prices = []

    @property
    def n_social(self):
        return len(self.social_prices)

    @property
    def n_computer(self):
        return len(self.computer_prices)

    @property
    def n_total(self):
        return self.n_social + self.n_computer

    def take_social_sample(self, price):
        self.social_prices.append(price)

    def take_computer_sample(self, price):
        self.computer_prices.append(price)

    # check either the computer or social dataset
    # come up with a price by regression or average
    # make an offer
    # maybe it gets rejected maybe it does not get rejected
    # repeat

    # ok so basically I want two prior distributions for my agent: one that only uses social samples
    # and one that only uses computer samples
    # I want my agent to have the opportunity to switch between samples


def ols_regression(computer_sample, x_0):
    X = []
    y = []
    for cpu, ram, ssd, price in computer_sample:
        X.append([cpu, ram, ssd])
        y.append(price)
    X = sm.add_constant(np.array(X))  # add intercept
    y = np.array(y)
    model = sm.OLS(y, X).fit()
    x0_row = sm.add_constant(
        np.array([[x0[0], x0[1], x0[2]]]),
        has_constant='add'
    )
    pred = model.get_prediction(x0_row)
    mean_price = float(pred.predicted_mean[0])
    se_mean = float(pred.se_mean[0])
    return mean_price, se_mean


def mean(social_sample):
    # Extract prices
    prices = [row[3] for row in social_sample]
    n = len(prices)

    # Sample mean
    mean_int = sum(prices) / n

    # Sample variance (unbiased estimator)
    var = sum((p - mean_int)**2 for p in prices) / (n - 1)

    # Sample standard deviation
    std = math.sqrt(var)

    # Standard Error of the Mean (SEM)
    sem = std / math.sqrt(n)

    return mean_int, sem

def utility_function(c0, c2, c3, offer_price, clerk_wtta, number_of_computer_samples):
    if offer_price >= clerk_wtta:
        acceptance = 1
    else:
        acceptance = 0
    utility = - c0*offer_price - c2*number_of_computer_samples + c3*acceptance
    return utility

def clerk_acceptance_price(laptop_wanted):
    price = 300*laptop_wanted[0] + 30*laptop_wanted[1] + 2*laptop_wanted[2] # turn the numerical values into constants
    clerk_wita = rng.normal(loc=price, scale=price*0.12)
    return clerk_wita
    # replace with gaussian_pricing_scheme

def gaussian_pricing_scheme(cpu_tier, ram_size, ssd_size):
    price = 300*cpu_tier + 30*ram_size + 2*ssd_size
    random_price = rng.normal(loc= price, scale=price*0.12) # 12% was randomly chosen
    return random_price

def setup_computers():
    computer_list = []
    cpu_tiers = [1,2]
    ram_sizes = [4,8]
    ssd_sizes = [128,256]
    for cpu_tier in cpu_tiers:
        for ram_size in ram_sizes:
            for ssd_size in ssd_sizes:
                for i in range(30):
                    rand_price = gaussian_pricing_scheme(cpu_tier, ram_size, ssd_size)  # create 30 random prices for each computer specification
                    computer_list.append((cpu_tier, ram_size, ssd_size, rand_price))
                    # 30 of each same specification with some added stochasticity
    # do this part in numpy (less computationally expensive)
    return computer_list

def social_sampling(computer_optionz,x_0):
    social_samples = []
    for computer in computer_optionz:
        if computer[0] == x_0[0] and computer[1] == x_0[1] and computer[2] == x_0[2]:
            social_samples.append(computer)
    return social_samples

def computer_sampling(computer_optionz,x_0):
    computer_samples = []
    for computer in computer_optionz:
        if computer[0] != x_0[0] or computer[1] != x_0[1] or computer[2] != x_0[2]:
            computer_samples.append(computer)
    return computer_samples

from scipy.stats import norm

def compute_o(alpha, sigma_tau, gamma, mu_tau, fallback_o=0.0, verbose=True):
    k = (alpha * sigma_tau * np.sqrt(2 * np.pi)) / gamma

    # e^{-z^2/2} must be in (0, 1], so k must be in (0,1]
    if k <= 0 or k > 1:
        if verbose:
            print(
                f"[compute_o] No interior optimum for "
                f"alpha={alpha}, sigma_tau={sigma_tau}, gamma={gamma}. "
                f"k={k:.4f} not in (0, 1]. Returning fallback o={fallback_o}."
            )
        return fallback_o

    root_term = -2 * np.log(k)  # >= 0 in the valid range
    z = np.sqrt(root_term)      # + root
    o = mu_tau + sigma_tau * z
    return o

def one_round(x0, c0, c2, c3,):
    clerk_wta = clerk_acceptance_price(x0)

    social_samples = social_sampling(setup_computers(), x0)
    computer_samples = computer_sampling(setup_computers(), x0)

    n_total_social = len(social_samples)
    n_total_computer = len(computer_samples)

    social_mu, social_sigma = mean(social_samples)
    computer_mu, computer_sigma = ols_regression(computer_samples,x0)
    print('social mean:', social_mu)
    print('computer mean:', computer_mu)
    print('social standard deviation:', social_sigma)
    print('computer standard deviation:', computer_sigma)

    offering_price_social = compute_o(c0, social_sigma, c3, social_mu)
    offering_price_computer = compute_o(c0, computer_sigma, c3, computer_mu)

    u_social = utility_function(
        c0, c2, c3,
        offer_price=offering_price_social,
        clerk_wtta=clerk_wta,
        number_of_computer_samples=0  # using only social in this scenario
    )

    u_computer = utility_function(
        c0, c2, c3,
        offer_price=offering_price_computer,
        clerk_wtta=clerk_wta,
        number_of_computer_samples=n_total_computer  # all samples are computer-based
    )

    #print(f"Clerk WTA (hidden): {clerk_wta:.2f}")
    #print(f"Offer (social):     {offering_price_social:.2f}, utility: {u_social:.2f}")
    #print(f"Offer (computer):   {offering_price_computer:.2f}, utility: {u_computer:.2f}")

    if u_social > u_computer:
        chosen_method = "social"
    else:
        chosen_method = "computer"

    print("Method with higher utility this round:", chosen_method)

    return chosen_method
# have an outlier treatment condition maybe for the agent
if __name__ == "__main__":
    x0 = (2, 4, 128)
    # c0 = 1.0
    # cost per dollar in offer c1 = 0.1
    # cost per total sample c2 = 0.1
    # extra cost per computer sample c3 = 500.0
    # reward if deal accepted
    total_social_accepted_c0 = 0
    total_computer_accepted_c0 = 0
    prev_c0 = ""
    total_social_accepted_c2 = 0
    total_computer_accepted_c2 = 0
    prev_c2 = ""
    total_social_accepted_c3 = 0
    total_computer_accepted_c3 = 0
    prev_c3 = ""

    for c0 in range(1, 21, 1):
        method = one_round(x0, c0, 1,50.0)
        if prev_c0 == "":
            prev_c0 = method
        elif prev_c0 != method:
            print("Method changed at c0 = ", c0)
            prev_c0 = method
        if method == "social":
            total_social_accepted_c0 = total_social_accepted_c0 + 1
        else:
            total_computer_accepted_c0 = total_computer_accepted_c0 + 1
    print("=========================================================")
    for c2 in range(1, 21, 10):
        method = one_round(x0, 1, c2,50.0)
        if prev_c2 == "":
            prev_c2 = method
        elif prev_c2 != method:
            print("Method changed at c0 = ", c2)
            prev_c2 = method
        if method == "social":
            total_social_accepted_c2 = total_social_accepted_c2 + 1
        else:
            total_computer_accepted_c2 = total_computer_accepted_c2 + 1
    print("=========================================================")

    for c3 in range(50, 201, 10):
        method = one_round(x0, 1, 1,c3)
        if prev_c3 == "":
            prev_c3 = method
        elif prev_c3 != method:
            print("Method changed at c3 = ", c3)
            prev_c3 = method
        if method == "social":
            total_social_accepted_c3 = total_social_accepted_c3 + 1
        else:
            total_computer_accepted_c3 = total_computer_accepted_c3 + 1


'''
social mean: 978.0483494449371
computer mean: 967.5676357225287
social standard deviation: 23.397382708972597
computer standard deviation: 22.80101731816566
Clerk WTA (hidden): 1011.69
Offer (social):     1026.49, utility: -526.49
Offer (computer):   1015.06, utility: -536.06
Method with higher utility this round: social

'''