#Matthew McCarthy, 2022

#---Imports---
from random import randint
from time import time
import matplotlib.pyplot as plt
import numpy as np

#---Functions---
def initialise_nn_matrix(dimension: int) -> list[list[int]]:
    '''Returns an (n * n) matrix of random integers between 0 and n, where n is the dimension of the matrix.'''
    return [[randint(0, dimension) for _ in range(dimension)] for _ in range(dimension)]

    #Optional function to test using numpy module
def initialise_nn_matrix_np(dimension: int) -> np.ndarray:
    '''Returns an (n * n) ndarray of random integers between 0 and n, where n is the dimension of the matrix.'''
    return np.random.randint(0, dimension, (dimension, dimension))

def multiply_matrices(mat_A: list[list[int]], mat_B: list[list[int]]) -> list[list[int]]:
    ''' Takes two matrices as input and outputs their product.
        e.g:
                    mat_A = [[x y]
                            [w z]]

                    mat_B = [[a b]
                            [c d]]

            output_matrix = [[x*a + y*c   x*b + y*d]
                            [w*a + z*c   w*b + z*d]]'''
    
    output_matrix = []

    #Don't try to make sense of this if you value your sanity. Just know that it multiplies mat_A and mat_B and it hasn't broken yet.
    for row in mat_A:
        row_array = []
        for i in range(len(mat_B)):
            col = []
            for j in range(len(mat_B[i])):
                col.append(mat_B[j][i])
            
            temp = []
            for i in range(len(row)):
                temp.append(row[i] * col[i])

            row_array.append(sum(temp))
        output_matrix.append(row_array)

    return output_matrix

def run_test(n: int, matrix_mode: int = 1) -> list:
    #Settings
    options = { 1 : [initialise_nn_matrix, multiply_matrices],  #Settings for either using manual matrix multiplication or
                2 : [initialise_nn_matrix_np, np.matmul] }      # numpy matrix multiplication.
    create, mul = options[matrix_mode] #Defines our 'create' and 'multiply' functions based on the mode we pass as a parameter to this function.
    
    '''Runs through all integers from 2 to n and records the time taken at each value to mutliply two matrices of that size. Returns list of times.'''
    
    times = [] #Initialised list for storing the time taken at each value of n.
    for dim in range(2, n + 1):
        mat_A, mat_B = create(dim), create(dim) #Create two matrices at current indexed size (dim).

        start = time()
        try:
            mul(mat_A, mat_B) #Multiplies mat_A with mat_B, does nothing with their output.
        except: #Basic error handling. Ignore pls
            print('Error')
            break
        end = time()

        progress_bar(dim, n)
        times.append(end - start) #Append the length of time taken to compute to our list of times.

    return times

def plot_test(n: int, test_mode: int = 1, fit_line: bool = False) -> None:
    '''Calls run_test() over n values and records output list as 'y_data'.
        test_mode:  1 = use my manual matrix multiplication
                    2 = use numpy's matrices and matrix multiplication
        fit_line:   True = will try to create line of best fit for plot
                    False = turn off line of best fit (recommended)'''

    start = time() #Optional timing for programme total
    x_data = range(2, n + 1) #Creates list from 2 to n inclusive, which are the values used in run_test().
    y_data = run_test(n, test_mode) 
    end = time()
    
    plt.ylabel('Time (seconds)')
    plt.xlabel('n')
    plt.plot(x_data, y_data, 'r-')

    print(f'\nProcess took {end - start} seconds to run in total.\n')

    
    if fit_line: #Creates a line roughly matching the curve expected. Very optional
        const = ((end - start) * (4/(n ** 4))) ** (1/3) #Constant needed to scale our line to fit data. Don't try to understand because it's not written out well. 

        best_fit_data = [(const * x) ** 3 for x in x_data] #Raising to the third power to match expected complexity, O(n^3), arising from analysis of my multiply_matrices() function 
        plt.plot(x_data, best_fit_data, 'g--')

    plt.show()
    
    #Optional for marking progress in terminal. Credit: https://www.youtube.com/watch?v=x1eaT88vJUA&ab_channel=NeuralNine
def progress_bar(progress: int, total: int) -> None:
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * int(100 - percent)
    print(f"\r|{bar}| {percent:.2f}%", end = "\r")

if __name__ == '__main__':
    #---Test---
    '''NB: will take approx. 25 seconds to run if n = 150 in test_mode 1;
        set test_mode = 2 for faster results.'''
    plot_test(n = 150, test_mode = 1, fit_line = False)