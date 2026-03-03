import numpy as np
import scipy
from Model import *


# Firms
def labor_demand(par, p):
    '''Labor demand
    
    Args:
        p: price of goods
    Returns:
        Labor demand
    '''
    return ((p*par.A*par.gamma)/par.w)**(1/(1-par.gamma))


def production(par, p):
    '''Production
    
    Args:
        p: price of goods
    Returns:
        Production
    '''
    return par.A*labor_demand(par, p)**par.gamma

def profit(par, p):
    '''Profit
    
    Args:
        p: price of goods
    Returns:
        Profit
    '''
    return (1-par.gamma)/par.gamma*par.w*((p*par.A*par.gamma)/par.w)**(1/(1-par.gamma))

# Households
def utility(c1, c2, l, par):
    '''Utility
    
    Args:
        l: Labor
        p1: price of good 1
        p2: price of good 2
    Returns:
        Utility
    '''

    return np.log(c1**par.alpha*c2**(1-par.alpha))-par.nu*(l**(1+par.epsilon))/(1+par.epsilon)

def labor_supply(par, p1, p2, tau, T):
    '''Labor supply
    
    Args:
        p1: price of good 1
        p2: price of good 2
    Returns:
        Labor supply
    '''

    def objective_function(l):
        c1 = par.alpha*(par.w*l+ T + profit(par, p1)+profit(par, p2))/p1
        c2 = (1-par.alpha)*(par.w*l+ T + profit(par, p1)+profit(par, p2))/(p2+tau)
        obj = utility(c1, c2, l, par)
        return -obj
    
    # Perform optimization
    initial_guess = 0.5
    l = scipy.optimize.minimize(objective_function, initial_guess).x

    return l.item()

def demand(par, p1, p2, tau, T):
    '''Demand
    
    Args:
        l: Labor
    Returns:
        Demand
    '''
    c1 = par.alpha*(par.w*labor_supply(par,p1,p2,tau,T)+T+profit(par, p1)+profit(par, p2))/p1
    c2 = (1-par.alpha)*(par.w*labor_supply(par,p1,p2,tau,T)+T+profit(par, p1)+profit(par, p2))/(p2+tau)

    return c1, c2

def utility_SWF(par, tau, T, p1, p2):
    '''Utility
    
    Args:
        p1: price of good 1
        p2: price of good 2
        tau: tax
        T: transfer
    Returns:
        Utility
    '''
    l = labor_supply(par, p1, p2, tau, T)
    c1, c2 = demand(par, p1, p2, tau, T)

    return np.log(c1**par.alpha*c2**(1-par.alpha))-par.nu*(l**(1+par.epsilon))/(1+par.epsilon)

def SWF(tau, T, p1, p2, par):
    '''Social welfare function
    
    Args:
        p1: price of good 1
        p2: price of good 2
    Returns:
        Social welfare function
    '''

    return utility_SWF(par, tau, T, p1, p2) - par.kappa*production(par, p2)

def Walras_law(prices, tau, T, par):
    '''Use Walras law to get market clearing
    
    Args:
        p1: price of good 1
    Returns:
        Market clearing
    '''

    p1, p2 = prices

    # Labor demanded by the firms
    l1 = labor_demand(par, p1)
    l2 = labor_demand(par, p2)

    # Optimal production by the firms
    y1 = production(par, p1)
    y2 = production(par, p2)

    # Labor supply
    ell = labor_supply(par, p1, p2, tau, T)

    # Household consumption
    c1 , c2 = demand(par, p1, p2, tau, T)
    if c1 < 0 or c2 < 0:
        return np.inf, np.inf

    # Market clearings
    labor_market_clear = ell - l1 - l2
    good1_market_clear = y1 - c1
    good2_market_clear = y2 - c2

    return good2_market_clear, labor_market_clear








