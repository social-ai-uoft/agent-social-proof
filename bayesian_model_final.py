import math
from typing import List, Tuple, Optional, Dict
import numpy as np
import statsmodels.api as sm
from fontTools.misc.cython import returns

# scipy or scikit learn for the ols regression

# Commit to the GitHub project

rng = np.random.default_rng(42)

class BayesianAgent:
    def __init__(self,
                 prior_mean,
                 prior_var,
                 obs_var_social,
                 obs_var_computer,
                 c0, c1, c2, c3):
        # Beliefs about willingness to accept
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.post_mean = prior_mean
        self.post_var = prior_var

        # assumptions about variance in social and computer samples
        self.obs_var_social = obs_var_social
        self.obs_var_computer = obs_var_computer

        # utility cost
        self.c0 = c0  # cost per dollar offered
        self.c1 = c1  # cost per total sample
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

    def update_posterior(self, mode="combined"):
        # start from the prior every time for clarity
        mu = self.prior_mean
        var = self.prior_var

        if mode in ("combined", "social"):
            if self.social_prices:
                mu, var = normal_normal_posterior(
                    prior_mean=mu,
                    prior_var=var,
                    samples=self.social_prices,
                    obs_var=self.obs_var_social
                )

        if mode in ("combined", "computer"):
            if self.computer_prices:
                mu, var = normal_normal_posterior(
                    prior_mean=mu,
                    prior_var=var,
                    samples=self.computer_prices,
                    obs_var=self.obs_var_computer
                )

        self.post_mean = mu
        self.post_var = var

    def propose_price_from_computer(computer_sample, x0):
        mean_price, se_mean = ols_regression(computer_sample, x0)

        # Draw one candidate WTP from Normal(mean_price, se_mean)
        offer = rng.normal(loc=mean_price, scale=se_mean if se_mean > 0 else 1.0)
        return max(0.0, offer)  # ensure non-negative

    def propose_price_from_social(social_sample):
        mean_price, std_price = mean(social_sample)

        # Draw one candidate WTP from Normal(mean_price, std_price)
        offer = rng.normal(loc=mean_price, scale=std_price if std_price > 0 else 1.0)
        return max(0.0, offer)

    # check either the computer or social dataset
    # come up with a price by regression or average
    # make an offer
    # maybe it gets rejected maybe it does not get rejected
    # repeat

    # ok so basically I want two prior distributions for my agent: one that only uses social samples
    # and one that only uses computer samples
    # I want my agent to have the opportunity to switch between samples

def normal_normal_posterior(prior_mean, prior_var, samples, obs_var):
    n = len(samples)
    if n == 0:
        return prior_mean, prior_var

    sum_y = sum(samples)

    precision_prior = 1.0 / prior_var # precision of distribution
    precision_lik = n / obs_var # precision likelihood
        # High variance implies low precision
        # Low variance implies high precision

    post_var = 1.0 / (precision_prior + precision_lik)
    post_mean = post_var * (precision_prior * prior_mean + (sum_y / obs_var))

    return post_mean, post_var



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
    mean_database = []
    mean_int = 0
    for social_samp in social_sample:
        mean_database.append(social_samp[3])
    for item in mean_database:
        mean_int = mean_int + item
    mean_int = mean_int/len(social_sample)
    mse = 0
    for item in mean_database:
        mse = mse + ((item - mean_int)**2)
    standard_error = math.sqrt(mse)
    return mean_int, standard_error

def utility_function(c0, c1, c2, c3, offer_price, clerk_wtta,number_of_total_samples, number_of_computer_samples):
    if offer_price >= clerk_wtta:
        acceptance = 1
    else:
        acceptance = 0
    utility = - c0*offer_price - c1*number_of_total_samples - c2*number_of_computer_samples + c3*acceptance
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

# have an outlier treatment condition maybe for the agent
if __name__ == "__main__":
    #computer_options = setup_computers()
    #x0 = (2,4,128) # the laptop that you are looking to purchase (want to find optimal price)
    #clerk_wta = clerk_acceptance_price(x0)
    #social_samples = social_sampling(computer_options,x0)
    #computer_samples = computer_sampling(computer_options, x0)
    x0 = (2, 4, 128)  # laptop the agent wants

    # True (hidden) clerk WTA for this run
    clerk_wta = clerk_acceptance_price(x0)

    # 2. Get candidate pools
    social_samples = social_sampling(setup_computers(), x0)
    computer_samples = computer_sampling(setup_computers(), x0)

    print(ols_regression(computer_samples,x0))

    # 3. Cost parameters for utility
    c0 = 1.0  # cost per dollar in offer
    c1 = 0.1  # cost per total sample
    c2 = 0.1  # extra cost per computer sample
    c3 = 500.0  # reward if deal accepted

    n_total_social = len(social_samples)
    n_total_computer = len(computer_samples)

    # 4. Agent proposes offers from each method
    offer_social = BayesianAgent.propose_price_from_social(social_samples)
    offer_computer = BayesianAgent.propose_price_from_computer(computer_samples, x0)

    # 5. Compute utilities for each method
    u_social = utility_function(
        c0, c1, c2, c3,
        offer_price=offer_social,
        clerk_wtta=clerk_wta,
        number_of_total_samples=n_total_social,
        number_of_computer_samples=0  # using only social in this scenario
    )

    u_computer = utility_function(
        c0, c1, c2, c3,
        offer_price=offer_computer,
        clerk_wtta=clerk_wta,
        number_of_total_samples=n_total_computer,
        number_of_computer_samples=n_total_computer  # all samples are computer-based
    )

    print(f"Clerk WTA (hidden): {clerk_wta:.2f}")
    print(f"Offer (social):     {offer_social:.2f}, utility: {u_social:.2f}")
    print(f"Offer (computer):   {offer_computer:.2f}, utility: {u_computer:.2f}")

    # so i was able to do one round but cannot do mutliple roudns not too sure how to impletmenet this
    if u_social > u_computer:
        chosen_method = "social"
    else:
        chosen_method = "computer"

    print("Method with higher utility this round:", chosen_method)
