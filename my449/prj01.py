import numpy as np
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1():
    arry = np.zeros([10,5],dtype=int)
    for i in range(10):
        for j in range(5):
            arry[i][j]=i+j
    return arry 

# Add two numpy arrays together.
def Q2(a, b):
    arry = a+b
    return arry

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
    arry = np.dot(a,b)
    return arry

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    max_index = np.argmax(a,axis=1)
    return max_index

# Solve Ax = b.
def Q5(A, b):
    arry =np.linalg.solve(A,b)
    return arry

# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input("a")
    b = nn.Input("b")
    c =a+b
    return c

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input("a")
    b = nn.Input("b")
    c = nn.Input("c")
    result = a + b*c
    return result

# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A, b):
    A = nn.Const(A)
    b = nn.Const(b)
    x = nn.Input("x")
    result = A*x+b
    return result

# Given n, return an EasyNN expression for x**n.
def Q9(n):
    x = nn.Input("x")
    result = nn.Const(1)
    for i in range(n):
        result = x*result
    return result

# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    x = nn.Input("x")
    relu = nn.ReLU()
    result = relu(x) + relu(-x)
    return result
