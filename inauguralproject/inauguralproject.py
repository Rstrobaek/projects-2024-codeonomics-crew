
from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        par.w1B = 1 - par.w1A   #Consumer B endownment normalisation 
        par.w2B = 1 - par.w2A   #Consumer B endownment normalisation

    def utility_A(self,x1A,x2A):
        """Defines the utility for consumer A
        
        Args:
            self: Parameters
            x1A: Number of good 1
            x2A: Number of good 2

        

        """
        u_A = x1A**(par.alpha) * x2A**(1-par.alpha)
        return u_A


    def utility_B(self,x1B,x2B):
        """
        
        """

        u_B = x1B**par.beta * x2B**(1-par.beta)
        return u_B

    def demand_A(self,p1):
        """
        
        """

        x1A = par.alpha*(p1*par.w1A+p2*par.w2A)/p1      #Demand for good 1
        x2A = (1-par.alpha)*(p1*par.w1A+p2*par.w2A)/1   #Demand for good 2

        return x1A, x2A

    def demand_B(self,p1):
        """
        
        """

        x1B = par.beta*(p1*par.w1B+p2*par.w2B)/p1      #Demand for good 1
        x2A = (1-par.beta)*(p1*par.w1B+p2*par.w2B)/1   #Demand for good 2

        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2