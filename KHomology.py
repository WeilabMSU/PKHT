# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:59:40 2024

@author: lshen
"""
import numpy as np
from itertools import combinations
from KHcomplex import SmoothingStateGenerator, StateMap, generate_state_map  # Import your SmoothingStateGenerator
import sympy as sp


# def left_null_space(A):# very slow, but can be very accurate for calculating the quantum degree
#     matrix= sp.Matrix(A)
#     left_null_space = matrix.T.nullspace()
#     LNS = np.array([]) if len(left_null_space)==0 else np.vstack([np.array(vec).reshape(1,-1) for vec in left_null_space])    
#     return LNS

def left_null_space(A, tol=1e-10): #fast, but not accurate
    """
    Computes the left null space of a matrix A using Singular Value Decomposition (SVD).
    
    Parameters:
        A (numpy.ndarray): The input matrix.
        tol (float): Tolerance for determining zero singular values.
        
    Returns:
        numpy.ndarray: The left null space of A as rows of a matrix.
    """
    # Perform SVD decomposition
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    
    # Identify the rank based on tolerance
    rank = np.sum(S > tol)
    
    # Extract the left null space from U (the last columns correspond to null space)
    left_null_space = U[:, rank:].T
    
    return left_null_space

#svd null space
def pointwise_product_sum(list1, list2):
    """
    Compute the pointwise product of two lists and return the sum of the products.
    
    Parameters:
        list1 (list or np.ndarray): First list of numbers.
        list2 (list or np.ndarray): Second list of numbers.
        
    Returns:
        float: The sum of the pointwise product.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    
    # Convert to NumPy arrays for efficient computation
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Calculate pointwise product and sum
    return np.sum(array1 * array2)


def sum_elements(lst):
    """
    Sum all elements in a list.
    
    Parameters:
        lst (list or np.ndarray): List of numbers.
        
    Returns:
        float: The sum of all elements.
    """
    # Convert to NumPy array for efficient computation
    array = np.array(lst)
    
    # Calculate sum
    return np.sum(array)




def calculate_quantum_theta(generator):
    theta=0
    for item in generator:
        if item[0] =='v_+':
            theta+= 1
        else:
            theta+=-1
    return theta

class KhovanovHomology_tangle:
    def __init__(self, pd_code):
        """Initialize the class with the PD code and generate the smoothing state generator."""
        self.pd_code = pd_code
        self.smoothing_state_generator = SmoothingStateGenerator(pd_code)
        #get quantum_key
        self.quantum_key={}
        for k in range(len(self.pd_code)+1):
            k_strings = self.get_k_strings(k, len(self.pd_code))
            self.quantum_key[k] = []
            for k_string in k_strings:
                for generator in self.smoothing_state_generator.get_smoothing_state(k_string).generators:
                    self.quantum_key[k].append(calculate_quantum_theta(generator)+k)
        
        #calculate boundary_matrices
        self.boundary_matrices={}
        for k in range(len(self.pd_code)):
            self.boundary_matrices[k]=self.generate_boundary_matrix(k)
        ##add d_-1,d_n
        mm = self.boundary_matrices[0].shape[1]
        nn = self.boundary_matrices[len(self.pd_code)-1].shape[0]
        self.boundary_matrices[-1] = np.zeros((mm,0),dtype=int)
        self.boundary_matrices[len(self.pd_code)] = np.zeros((0,nn),dtype=int)
        ##calculate homology generators and quantumed generators.
        self.homology()
        self.get_quantumed_generator()
 
    def get_k_strings(self, k, total_length):
        """Generate all k-strings with exactly k '1's, ordered lexicographically."""
        # Generate all combinations of indices where the '1's can be placed
        if k < 0 or k > total_length:
            raise ValueError(f"k must be between 0 and {total_length}, but got {k}.")
        indices_combinations = combinations(range(total_length), k)
        
        k_strings = []
        for indices in indices_combinations:
            # Create a list of '0's
            binary_string = ['0'] * total_length
            # Place '1's at the appropriate positions
            for index in indices:
                binary_string[index] = '1'
            # Convert the list back to a string and append to the result list
            k_strings.append(''.join(binary_string))
        
        return k_strings

    def is_valid_transition(self, pre_string, post_string):
        """Check if the transition between pre_string and post_string is valid."""
        return StateMap.is_valid_map(pre_string, post_string)[0]  # Returns True if valid

    def generate_boundary_matrix(self, k):
        """Generate the boundary matrix from dimension k to k+1."""
        total_length = len(self.pd_code)  # The number of crossings, i.e., the length of state strings
        
        # Get all pre-strings (k '1's) and post-strings (k+1 '1's), lexicographically ordered
        pre_strings = self.get_k_strings(k, total_length)
        post_strings = self.get_k_strings(k + 1, total_length)
        #print(pre_strings,post_strings)
        
        # Initialize the boundary matrix with appropriate block sizes
        boundary_matrix_height = sum(len(self.smoothing_state_generator.get_smoothing_state(post).generators) for post in post_strings)
        boundary_matrix_width = sum(len(self.smoothing_state_generator.get_smoothing_state(pre).generators) for pre in pre_strings)
        boundary_matrix = np.zeros((boundary_matrix_height, boundary_matrix_width),dtype=int)
        

        
        
        # Set up row and column indices for block insertion
        post_row_offset = 0
        for i, post_string in enumerate(post_strings):
            post_state = self.smoothing_state_generator.get_smoothing_state(post_string)
            post_block_height = len(post_state.generators)
        
            pre_col_offset = 0
            for j, pre_string in enumerate(pre_strings):
                pre_state = self.smoothing_state_generator.get_smoothing_state(pre_string)
                pre_block_width = len(pre_state.generators)
        
                if self.is_valid_transition(pre_string, post_string):
                    # Generate StateMap for valid transitions
                    state_map = generate_state_map(self.smoothing_state_generator, pre_string, post_string)
                    block_matrix = state_map.mapping_matrix
                    #print(pre_string,post_string,"\n",block_matrix)
                else:
                    # Invalid transition results in a zero matrix with appropriate dimensions
                    block_matrix = np.zeros((post_block_height, pre_block_width))
                    #print(pre_string,post_string,"\n",block_matrix)
        
                # Insert the block matrix into the boundary matrix at the correct position
                boundary_matrix[post_row_offset:post_row_offset + post_block_height,
                             pre_col_offset:pre_col_offset + pre_block_width] = block_matrix
        
                # Update the pre_col_offset for the next block in the same row
                pre_col_offset += pre_block_width
        
            # Update the post_row_offset for the next block in the next row
            post_row_offset += post_block_height
        
        return boundary_matrix
    
    def homology(self):
        self.KH_generators={}
        for dim in range(len(self.pd_code)+1):
            self.KH_generators[dim] = left_null_space(np.hstack((self.boundary_matrices[dim-1],self.boundary_matrices[dim].T)))
        
    def get_quantumed_generator(self):
        self.quantumed_generators={}
        tolerance=1e-5
        for dim in range(len(self.pd_code)+1):
            self.quantumed_generators[dim] = []
            for generator in self.KH_generators[dim]:
                #print(self.quantum_key[dim],generator,dim)
                quantum_value = pointwise_product_sum(self.quantum_key[dim], generator)/sum_elements(generator)
                
                self.quantumed_generators[dim].append(quantum_value)

                

# Example PD code
# pdcode = [
#     ["3", "10|", "4", "9"],
#     ["|1", "8", "2", "|7"],    
#     ["9", "4", "8", "5"],
#     ["2", "5", "3", "6|"]
# ]

# pdcode=[
#     ["4","1","3","2"],
#     ["2","3","1","4"]
#     ]

# pdcode=[
#     ["3","1","4","6"],
#     ["1","5","2","4"],
#     ["5","3","6","2"]
#     ]
# pdcode=[
#     ["1","3","6","4"],
#     ["5","1","4","2"],
#     ["3","5","2","6"]
#     ]
#X6172 X12,7,13,8 X4,13,1,14 X10,6,11,5 X8493 X14,10,5,9 X2,12,3,11
# pdcode =[
#     ["6","1","7","2"],
#     ["12","7","13","8"],
#     ["4","13","1","14"],
#     ["10","6","11","5"],
#     ["8","4","9","3"],
#     ["14","10","5","9"],
#     ["2","12","3","11"]    
#     ]
#X6172 X14,7,15,8 X4,15,1,16 X10,6,11,5 X8493 X18,11,19,12 X20,17,5,18 X12,19,13,20 X16,10,17,9 X2,14,3,13
# pdcode = [
#     ["6","1","7","2"],
#     ["14","7","15","8"],
#     ["4","15","1","16"],
#     ["10","6","11","5"],
#     ["8","4","9","3"],
#     ["18","11","19","12"],
#     ["20","17","5","18"],
#     ["12","19","13","20"],
#     ["16","10","17","9"],
#     ["2","14","3","13"]
#     ]




# Create the generator object
# import time
# start_time = time.time()
# KHT = KhovanovHomology_tangle(pdcode)
# print("quantumed generator","\n",KHT.quantumed_generators,"\n")
# # print("boundary_matrices","\n",KHT.boundary_matrices,"\n")
# print("generators","\n",KHT.KH_generators,"\n")
# end_time = time.time()
# time_cost = end_time - start_time

# # Display the time cost
# print(f"Time cost: {time_cost:.6f} seconds")