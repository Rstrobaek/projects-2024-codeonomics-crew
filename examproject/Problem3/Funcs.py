import numpy as np
import scipy  
import matplotlib.pyplot as plt



def f(x):
    return x[0] * x[1]

def algorithm1(X, y):

    A, B, C, D = find_points(X, y)    

    try:
        r1_ABC, r2_ABC, r3_ABC, r1_CDA, r2_CDA, r3_CDA = barycentric_coordinates(A,B,C,D,y)

        if is_in_triangle(r1_ABC, r2_ABC, r3_ABC):
            return r1_ABC*f(A) + r2_ABC*f(B) + r3_ABC*f(C)
        
        elif is_in_triangle(r1_CDA, r2_CDA, r3_CDA):
            return r1_CDA*f(C) + r2_CDA*f(D) + r3_CDA*f(A)
    
        else:
            return None
    except TypeError:
        return None
    
def barycentric_coordinates(A,B,C,D,y):
    denominator1 = (B[1]-C[1])*(A[0]-C[0])+(C[0]-B[0])*(A[1]-C[1])
    denominator2 = (D[1]-A[1])*(C[0]-A[0])+(A[0]-D[0])*(C[1]-A[1])

    r1_ABC = ((B[1]-C[1])*(y[0]-C[0])+(C[0]-B[0])*(y[1]-C[1]))/denominator1
    r2_ABC = ((C[1]-A[1])*(y[0]-C[0])+(A[0]-C[0])*(y[1]-C[1]))/denominator1
    r3_ABC = 1 - r1_ABC - r2_ABC

    r1_CDA = ((D[1]-A[1])*(y[0]-A[0])+(A[0]-D[0])*(y[1]-A[1]))/denominator2
    r2_CDA = ((A[1]-C[1])*(y[0]-A[0])+(C[0]-A[0])*(y[1]-A[1]))/denominator2
    r3_CDA = 1 - r1_CDA - r2_CDA

    return r1_ABC, r2_ABC, r3_ABC, r1_CDA, r2_CDA, r3_CDA

def is_in_triangle(r1, r2, r3):
    # Check values are between 0 and 1
    if (r1 >= 0 and r1 <= 1) and (r2 >= 0 and r2 <= 1) and (r3 >= 0 and r3 <= 1):
        return True
    else:
        return False
    
def find_points(X, y):
    A = min((x for x in X if x[0] > y[0] and x[1] > y[1]), key=lambda x: np.linalg.norm(x - y), default=None)
    B = min((x for x in X if x[0] > y[0] and x[1] < y[1]), key=lambda x: np.linalg.norm(x - y), default=None)
    C = min((x for x in X if x[0] < y[0] and x[1] < y[1]), key=lambda x: np.linalg.norm(x - y), default=None)
    D = min((x for x in X if x[0] < y[0] and x[1] > y[1]), key=lambda x: np.linalg.norm(x - y), default=None)
    return A, B, C, D


