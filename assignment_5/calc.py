# %%
import numpy as np
import math
import matplotlib.pyplot as plt

def calc_success_prob(M:float, p:float) -> float:
    """
    Calculate the probability, that the majority is making the right decision.
    """
    return 1 - np.sum([math.comb(M, i) * p**i * (1-p)**(M-i) for i in range(0, math.floor(M/2+1))])



# %%
calc_success_prob(19, 0.6)
# %%

max_jury_size = 100
min_jury_size = 1
jury_increment = 1
max_p = 0.99
min_p = 0.01
p_increment = 0.01

probabilities = []
for jury_size in range(min_jury_size, max_jury_size+jury_increment, jury_increment):
    current_probs = []
    p = min_p
    while p<=max_p:
        current_probs.append(calc_success_prob(jury_size, p))
        p = round(p + p_increment, 2)
    probabilities.append(current_probs)

# %%
for j in range(len(probabilities)):
    plt.plot([i for i in range(len(probabilities[j]))], probabilities[j])

# %%
probs = [calc_success_prob(i, 0.6) for i in range(19,40)]
plt.plot([i for i in range(19,40)], probs)
# %%

def prob_bounded(M:float, p:float, successes:int) -> float:
    """
    Calculate the probability, that the majority is making the right decision.
    """
    return np.sum([math.comb(M, i) * p**i * (1-p)**(M-i) for i in range(0, successes)])

def success_prob(M:float, p:float, majority:int) -> float:
    """
    Calculate the probability, that the majority is making the right decision.
    """
    return 1 - np.sum([math.comb(M, i) * p**i * (1-p)**(M-i) for i in range(0, majority)])

def weighted_majority(weight:int, p_strong:float, p_weak:float) -> float:
    # there are 10 weak classifiers 
    n_weak = 10
    # the theoretical number of votes is the sum of weak classifiers and the weight of the strong classifier
    theoretical_vote_cap = n_weak + weight
    # the majority is based on the theoretical vote cap
    n_majority = math.floor(theoretical_vote_cap / 2) + 1
    print("Majority:", n_majority)
    
    # The first way to gain a majority is to have enough weak classifiers voting correctly
    if n_majority <= 10:
        weak_majority = success_prob(n_weak, p_weak, n_majority)
        print("Probability of weak majority:", weak_majority)
    else:
        weak_majority = 0
    
    # The second way to gain a majority is for the strong classifier to vote correctly and have enough support from the weak classifiers
    # For that we need to calculate the probability of the lower bound and subtract it from the probability of the upper bound of supporting classifiers
    lower_bound = n_majority - weight
    if lower_bound > 0:
        upper_bound = n_majority
        p_lower = prob_bounded(n_weak, p_weak, lower_bound)
        p_upper = prob_bounded(n_weak, p_weak, upper_bound)
        strong_majority = (p_upper - p_lower) * p_strong
        print("Probability of strong majority:", strong_majority)
    else:
        strong_majority = p_strong
    
    return weak_majority + strong_majority

print("Weight: 1, Probability:", weighted_majority(1, 0.8, 0.6))
# %%
probabilities = [weighted_majority(weight, 0.8, 0.6) for weight in range(1, 11)]

plt.plot([i for i in range(1,11)], probabilities)
# %%

print(math.log((1-0.2)/0.2))
print(math.log((1-0.4)/0.4))
# %%
