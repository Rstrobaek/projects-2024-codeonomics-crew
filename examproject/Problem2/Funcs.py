import numpy as np
import scipy  
import matplotlib.pyplot as plt

from Model import graduate_model
model = graduate_model()
model.setup()
par = model.par


def sim_utility(par):
    '''Simulate utility
    
    Args:
        None
    
    Returns:
        realised_utility: Array of actual utilities
        expected_utility: Array of expected utilities given their friends
        realised_utility_mean: Array of average actual utilities'''
    np.random.seed(42)          # Set seed

    # Arrays to store utilities
    expected_utility = np.zeros((par.J))
    realised_utility = np.zeros((par.K, par.J))
    epsilon_storage = np.zeros((par.K, par.J))

    for k in range(par.K):      # Loop over simulations

        

        for j in range(par.J):  # Loop over careers

            # Calculate errors terms
            epsilon = np.random.normal(0, par.sigma)
            # Add epsilon to storage
            epsilon_storage[k, j] = epsilon

            # Calculate utility
            realised_utility[k, j] = par.v[j] + epsilon
            

    # Calculate mean of the epsilons:
    mean_epsilon = np.mean(epsilon_storage, axis=0)
    # Calculate expected utility
    for j in range(par.J):
        expected_utility[j] = par.v[j] + mean_epsilon[j]
    
    # Calculate average across realised utilities
    realised_utility_mean = np.mean(realised_utility, axis=0)

    return realised_utility, expected_utility, realised_utility_mean


def friend_utility(par):
    '''Simulate utility for graduates when they consider their friends
    
    Args:
        None
    
    Returns:
        actual_epsilon_storage: Storage of actual epsilons (used for testing)
        expected_utility: Array of expected utilities given their friends
        career_choice: Array of career choices
        actual_utility: Array of actual utilities given their friends'''

    np.random.seed(42)

    # Arrays to store results
    actual_epsilon_storage = np.zeros((par.N, par.K, par.J))
    expected_utility = np.zeros((par.N, par.K, par.J))
    career_choice = np.zeros((par.N, par.K), dtype=int)
    actual_utility = np.zeros((par.N, par.K, par.J))

    for i in range(par.N):    # Loop over individuals
        
        # We are told that individual i has i friends. We add 1 to account for Python indexing
        Fi = i + 1

        for k in range(par.K):    # Loop over simulations
            for j in range(par.J):  # Loop over careers
                
                # Draw epsilon for each friend and sum column-wise
                draws = np.random.normal(0, par.sigma, (Fi, par.J))
                epsilon_friend = np.sum(draws, axis=0)  
                # We now have that for person 1, there is only one epsilon in each column, while for person 2 it is the sum of 2 epsilons etc.

                # Draw actual epsilons
                epsilon_actual = np.random.normal(0, par.sigma, par.J)
                # Add epsilon to storage
                actual_epsilon_storage[i, k, :] = epsilon_actual

                # Calculate expected utility (calling it temp as it is not yet stored in the array)
                expected_utility_temp = (epsilon_friend)/Fi+par.v
                # Add expected utility to storage
                expected_utility[i, k, :] = expected_utility_temp

                # Find career choice that maximizes expected utility
                career_choice[i, k] = np.argmax(expected_utility_temp)

                # Calculate actual utility
                actual_utility[i, k, :] = par.v + epsilon_actual


    return actual_epsilon_storage, expected_utility, career_choice, actual_utility


def analyze(par, career_choice, expected_utility, actual_utility):
    '''Analyze the results of the simulation
    
    Args:
        career_choice: Array of career choices
        expected_utility: Array of expected utilities given their friends
        actual_utility: Array of actual utilities given their friends
    
    Returns:
        career_shares: Array of shares of graduates choosing each career for each type of graduate
        expected_average_utility: Array of expected average utility for each type of graduate
        actual_average_utility: Array of actual average utility for each type of graduate'''
    # Initialize arrays to store results for each type of graduate (i)
    career_shares = np.zeros((par.N, par.J))
    expected_average_utility = np.zeros(par.N)
    actual_average_utility = np.zeros(par.N)

    for i in range(par.N):  # Loop over types of graduates
        # Calculate the shares of graduates choosing each career for this type of graduate
        career_counts = np.zeros(par.J)     # Empty array to store number of people in each career
        for j in range(par.J):
            career_counts[j] = np.sum(career_choice[i, :] == j)
        # Calculate shares by dividing by total number of graduates
        career_shares[i, :] = career_counts / par.K

        # Calculate the expected average utility from choosing the best career for this type of graduate
        expected_utility_sum = 0
        actual_utility_sum = 0
        for k in range(par.K):
            expected_utility_sum += expected_utility[i, k, career_choice[i, k]]
            actual_utility_sum += actual_utility[i, k, career_choice[i, k]]
        expected_average_utility[i] = expected_utility_sum / par.K
        actual_average_utility[i] = actual_utility_sum / par.K

    return career_shares, expected_average_utility, actual_average_utility




def plotting(career_shares, expected_average_utility, actual_average_utility):
    '''Function for plotting results'''
    x = np.arange(1, par.N+1)

    plt.figure(figsize=(12, 6))

    # plot career choices
    plt.subplot(1,2,1)
    for j in range(par.J):
        plt.plot(x, career_shares[:,j], label=f'Career {j+1}')
    plt.xlabel('Number of friends')
    plt.ylabel('Share of Gradutes choosing career')
    plt.title('Career choice by number of friends')
    plt.legend()

    # Plot utilities
    plt.subplot(1, 2, 2)
    plt.plot(x, expected_average_utility, label='Average Subjective Utility')
    plt.plot(x, actual_average_utility, label='Average Realized Utility')
    plt.xlabel('Number of Friends (Fi)')
    plt.ylabel('Utility')
    plt.title('Utilities by Number of Friends')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



def switch(par, career_choice, expected_utility, actual_utility):
    '''Function to calculate career switching
    
    Args:
        career_choice: Array of career choices
        expected_utility: Array of expected utilities given their friends
        actual_utility: Array of actual utilities given their friends
    
    Returns:
        choice_utility: Utility of their choice
        new_career_choice: New career choice
        career_shares: Array of shares of graduates choosing each career
        expected_average_utility: Array of expected average utility for each type of graduate
        actual_average_utility: Array of actual average utility for each type of graduate'''

    # Containers for results
    choice_utility = expected_utility.copy()    # We make a copy so we can alter the values without affecting the expected utility before switching
    new_career_choice = np.zeros((par.N, par.K), dtype=int)
    
    for i in range(par.N):
        for k in range(par.K):
            chosen_career = career_choice[i, k]
            for j in range(par.J):
                if j != chosen_career:  # If the career is not the chosen career, subtract the cost
                    choice_utility[i, k, j] -= par.c
                if j == chosen_career:  # If the career is the chosen career, substitute with realised utility
                    choice_utility[i, k, j] = actual_utility[i, k, j]
                
            # Find career choice that gives highest utility
            new_career_choice[i, k] = np.argmax(choice_utility[i, k])    


    # Initialize arrays to store results for each type of graduate (i)
    career_shares = np.zeros((par.N, par.J))
    expected_average_utility = np.zeros(par.N)
    actual_average_utility = np.zeros(par.N)
    

    for i in range(par.N):  # Loop over types of graduates
        # Calculate the shares of graduates choosing each career for this type of graduate
        career_counts = np.zeros(par.J)
        for j in range(par.J):
            career_counts[j] = np.sum(new_career_choice[i, :] == j)
        career_shares[i, :] = career_counts / par.K

        # Calculate the expected average utility from choosing the best career for this type of graduate
        choice_utility_sum = 0
        actual_utility_sum = 0
        for k in range(par.K):
            choice_utility_sum += choice_utility[i, k, new_career_choice[i, k]]
            actual_utility_sum += actual_utility[i, k, new_career_choice[i, k]]
        expected_average_utility[i] = choice_utility_sum / par.K
        actual_average_utility[i] = actual_utility_sum / par.K

    return choice_utility, new_career_choice, career_shares, expected_average_utility, actual_average_utility

# The following plotting function is made using Copilot
def plot_switch_shares(par, career_choice, new_career_choice):
    '''Plot the share of graduates that switch careers conditional on their initial career choice'''
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    axes = axes.flatten()
    
    # Initialize variable and array
    max_switch_share = 0
    switch_shares_all = np.zeros((par.N, par.J))

    for i in range(par.N):
        switch_shares = np.zeros(par.J)
        initial_choice = career_choice[i, :]
        new_choice = new_career_choice[i, :]

        for j in range(par.J):
            # Count the number of graduates that initially chose career j
            initial_count = np.sum(initial_choice == j)
            # Count the number of graduates that initially chose career j and switched to another career
            switch_count = np.sum((initial_choice == j) & (new_choice != j))
            # Calculate the share of graduates that switched careers
            if initial_count > 0:
                switch_shares[j] = switch_count / initial_count

        # Store the shares of graduates that switched careers
        switch_shares_all[i] = switch_shares
        # Update the maximum share of graduates that switched careers
        max_switch_share = max(max_switch_share, max(switch_shares))

    for i in range(par.N):
        x = np.arange(par.J)
        axes[i].bar(x, switch_shares_all[i], tick_label=[f'Career {j + 1}' for j in range(par.J)])
        axes[i].set_xlabel('Initial Career Choice')
        axes[i].set_ylabel('Share of Graduates Switching Careers')
        axes[i].set_title(f'Person {i + 1}')
        axes[i].set_ylim(0, max_switch_share)

    plt.tight_layout()
    plt.show()
