# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:00:00 2024

@author: lshen
"""
import numpy as np
from itertools import product

# Helper function 
def generate_state_map(smoothing_state_generator, pre_state_string, post_state_string):
    is_valid, change_index = StateMap.is_valid_map(pre_state_string, post_state_string)
    if is_valid:
        pre_state = smoothing_state_generator.get_smoothing_state(pre_state_string)
        post_state = smoothing_state_generator.get_smoothing_state(post_state_string)
        return StateMap(pre_state, post_state, change_index)
    else:
        raise ValueError("Invalid state transition: only one '0' can change to '1'")
        
def check_connected_elements_consistency(pre_generator, post_generator, unchanged_connections):
    """
    Check if the connected unchanged elements in the pre_generator and post_generator are consistent.
    If connected elements are not both represented by v_+, v_-, or w, return 0, else return 1.
    """
    for (pre_elem, post_elem) in unchanged_connections:
        pre_type = None
        post_type = None

        # Find the type in pre_generator
        for gen_type, element in pre_generator:
            if element == pre_elem:
                pre_type = gen_type
                break

        # Find the type in post_generator
        for gen_type, element in post_generator:
            if element == post_elem:
                post_type = gen_type
                break

        # If types don't match or are not both valid, return 0
        if pre_type != post_type or pre_type is None or post_type is None:
            return 0
    
    return 1  # If all connected elements are consistent

def generate_tensor_generators(state_elements):
    """Generate all possible tensor products for a list of StateElements."""
    # Generate the list of generators for each StateElement
    generators = []
    for element in state_elements:
        element_generators = []
        if element.w:
            element_generators.append(('w', element))
        if element.v_plus:
            element_generators.append(('v_+', element))
        if element.v_minus:
            element_generators.append(('v_-', element))
        generators.append(element_generators)
    
    # Calculate the Cartesian product of all generators
    tensor_products = list(product(*generators))
    
    return tensor_products


class SmoothingStateGenerator:
    def __init__(self, pdcode):
        self.pdcode = pdcode
        self.num_positions = len(pdcode)
        self.smoothing_states = self.generate_all_smoothing_states()

    def generate_all_smoothing_states(self):
        """Generate all possible smoothing states based on the pdcode."""
        state_strings = [''.join(state) for state in product('01', repeat=self.num_positions)]
        smoothing_states = {state: generate_smoothing_state(self.pdcode, state) for state in state_strings}
        return smoothing_states

    def get_smoothing_state(self, state_string):
        """Retrieve a specific smoothing state by its state string."""
        return self.smoothing_states.get(state_string, None)

    def get_all_smoothing_states(self):
        """Retrieve all generated smoothing states."""
        return self.smoothing_states


# StateElement Class
class StateElement:
    def __init__(self, representative, state_string):
        self.representative = representative  # List of all labels in this class
        self.state_string = state_string      # Remember the state string
        self.type = self.determine_type()  
        self.v_plus = self.type == 'circle'  # True if circle, False otherwise
        self.v_minus = self.type == 'circle'  # True if circle, False otherwise
        self.w = self.type == 'arc'  # True if arc, False otherwise
    def determine_type(self):
        """Determine the type of the equivalence class based on its labels."""
        for label in self.representative:
            if "|" in str(label):
                return "arc"
        return "circle"

    def __str__(self):
        return f"{'~'.join(map(str, sorted(self.representative)))} (Type: {self.type})"

# SmoothingState Class
class SmoothingState:
    def __init__(self, state_elms, state_string):
        self.state_elms = state_elms  # List of all StateElements (arcs, circles)
        self.state_string = state_string  # The smoothing string that generated this state
        self.generators = self.generate_generators()  # Generators based on the state elements

    def generate_generators(self):
            """Generate the tensor products of generators for each state element."""
            generator_groups = []
            for elm in self.state_elms:
                generators = []
                if elm.v_plus:  # If the element has v_plus and v_minus
                    generators.append(('v_+', elm))
                    generators.append(('v_-', elm))
                if elm.w:  # If the element has w
                    generators.append(('w', elm))
                
                generator_groups.append(generators)
    
            # Create the tensor products of generators
            state_generators = list(product(*generator_groups))
            return state_generators

    def __str__(self):
        return f"Smoothing State: {self.state_string}\n" + "\n".join(map(str, self.state_elms))


# Functions to handle smoothing
def apply_smoothing(pdcode, smoothing_type):
    """Applies the specified smoothing type (0 or 1) to the PDcode."""
    i, j, k, l = pdcode
    if smoothing_type == '0':
        return [(i, j), (k, l)]  # 0-smoothing: i~j and k~l
    elif smoothing_type == '1':
        return [(i, l), (j, k)]  # 1-smoothing: i~l and j~k
    else:
        raise ValueError("Smoothing type must be '0' or '1'.")

def find_representative(equivalence_map, element):
    """Find the representative of the element using path compression."""
    if equivalence_map[element] != element:
        equivalence_map[element] = find_representative(equivalence_map, equivalence_map[element])
    return equivalence_map[element]

def union_elements(equivalence_map, element1, element2):
    """Unite two elements into the same equivalence class."""
    root1 = find_representative(equivalence_map, element1)
    root2 = find_representative(equivalence_map, element2)
    if root1 != root2:
        equivalence_map[root2] = root1

def generate_smoothing_state(pdcode_list, state_string):
    """Generates a SmoothingState object from the PDcode and state string."""
    equivalence_map = {}

    for pdcode, smoothing in zip(pdcode_list, state_string):
        pairs = apply_smoothing(pdcode, smoothing)
        for a, b in pairs:
            if a not in equivalence_map:
                equivalence_map[a] = a
            if b not in equivalence_map:
                equivalence_map[b] = b
            union_elements(equivalence_map, a, b)

    class_map = {}
    for element in equivalence_map:
        root = find_representative(equivalence_map, element)
        if root not in class_map:
            class_map[root] = []
        class_map[root].append(element)

    state_elms = []
    for representative, members in class_map.items():
        state_elms.append(StateElement(members, state_string))

    return SmoothingState(state_elms, state_string)


# StateMap Class
class StateMap:
    def __init__(self, pre_state, post_state, smoothing_index):
        self.pre_state = pre_state
        self.post_state = post_state
        self.pre_state_string = pre_state.state_string  # Obtain from SmoothingState
        self.post_state_string = post_state.state_string  # Obtain from SmoothingState
        self.smoothing_index = smoothing_index  # Store the index where 0 changes to 1
        self.pre_changed, self.post_changed, self.pre_unchanged, self.post_unchanged = self.identify_changed_elements()  # Include unchanged elements
        self.pre_generators = pre_state.generators  # Use generators directly from the pre_state
        self.post_generators = post_state.generators  # Use generators directly from the post_state
        self.unchanged_connections = self.identify_unchanged_connections()
        self.coefficients = self.calculate_all_coefficients()
        self.mapping_matrix = self.generate_mapping_matrix()  # Add matrix generation here

    @staticmethod
    def is_valid_map(pre_state_string, post_state_string):
        """Check if the map is valid (exactly one '0' changes to '1') and return the index."""
        change_index = -1
        for i, (pre, post) in enumerate(zip(pre_state_string, post_state_string)):
            if pre == '0' and post == '1':
                if change_index != -1:
                    return False, -1  # More than one change detected
                change_index = i
            elif pre != post:
                return False, -1  # Invalid change detected
        return change_index != -1, change_index

    def identify_changed_elements(self):
        """Identify the elements that have changed or remained the same between pre_state and post_state."""
        pre_elements = {tuple(sorted(elm.representative)): elm for elm in self.pre_state.state_elms}
        post_elements = {tuple(sorted(elm.representative)): elm for elm in self.post_state.state_elms}

        pre_changed = []
        post_changed = []
        pre_unchanged = []
        post_unchanged = []

        for rep, elm in pre_elements.items():
            if rep not in post_elements:
                pre_changed.append(elm)
            else:
                pre_unchanged.append(elm)

        for rep, elm in post_elements.items():
            if rep not in pre_elements:
                post_changed.append(elm)
            else:
                post_unchanged.append(elm)

        return pre_changed, post_changed, pre_unchanged, post_unchanged

    def get_transition_type(self):
        """Determine the transition type based on the pre_changed and post_changed elements."""
    
        if len(self.pre_changed) == 2 and len(self.post_changed) == 1:
            if all(elm.type == 'circle' for elm in self.pre_changed) and self.post_changed[0].type == 'circle':
                return 'merge_circle'
            if ((self.pre_changed[0].type == 'arc' and self.pre_changed[1].type == 'circle') or
                (self.pre_changed[0].type == 'circle' and self.pre_changed[1].type == 'arc')) and self.post_changed[0].type == 'arc':
                return 'merge_arc_circle'
        
        elif len(self.pre_changed) == 1 and len(self.post_changed) == 2:
            if self.pre_changed[0].type == 'circle' and all(elm.type == 'circle' for elm in self.post_changed):
                return 'split_circle'
            if self.pre_changed[0].type == 'arc' and any(elm.type == 'arc' for elm in self.post_changed) and \
               any(elm.type == 'circle' for elm in self.post_changed):
                return 'split_arc_circle'
        
        elif len(self.pre_changed) == 2 and len(self.post_changed) == 2:
            if all(elm.type == 'arc' for elm in self.pre_changed) and all(elm.type == 'arc' for elm in self.post_changed):
                return 'saddle'
        
        return 'unsupported'
    
    def identify_unchanged_connections(self):
        """Identify connections between unchanged elements."""
        unchanged_connections = []
        pre_elements = {tuple(sorted(elm.representative)): elm for elm in self.pre_unchanged}
        post_elements = {tuple(sorted(elm.representative)): elm for elm in self.post_unchanged}
    
        for rep in pre_elements:
            if rep in post_elements:
                unchanged_connections.append((pre_elements[rep], post_elements[rep]))
    
        return unchanged_connections
    
    def determine_coefficient(self, pre_generator, post_generator, transition_type):
        """Determine the coefficient (1, 0, or -1) based on the transition type and the given generators."""
        # First, check the consistency of connected unchanged elements
        if check_connected_elements_consistency(pre_generator, post_generator, self.unchanged_connections) == 0:
            return 0
    
        # Then apply the transition type logic
        if transition_type == 'merge_circle':
            # Check the pre-generator pairs and map them to the appropriate post-generator
            if pre_generator[0][0] == 'v_+' and pre_generator[1][0] == 'v_+':
                return 1 if post_generator[0][0] == 'v_+' else 0
            elif pre_generator[0][0] == 'v_+' and pre_generator[1][0] == 'v_-':
                return 1 if post_generator[0][0] == 'v_-' else 0
            elif pre_generator[0][0] == 'v_-' and pre_generator[1][0] == 'v_+':
                return 1 if post_generator[0][0] == 'v_-' else 0
            elif pre_generator[0][0] == 'v_-' and pre_generator[1][0] == 'v_-':
                return 0  # Maps to 0
    
        elif transition_type == 'split_circle':
            # Check if the pre-generator maps to the appropriate tensor product in the post-generator
            if pre_generator[0][0] == 'v_+':
                if post_generator[0][0] == 'v_+' and post_generator[1][0] == 'v_-':
                    return 1
                elif post_generator[0][0] == 'v_-' and post_generator[1][0] == 'v_+':
                    return 1
            if pre_generator[0][0] == 'v_-':
                if post_generator[0][0] == 'v_-' and post_generator[1][0] == 'v_-':
                    return 1
    
        elif transition_type == 'saddle':
            # Check if w tensor w in pre-generator maps to 0 tensor 0 in post-generator
            if pre_generator[0][0] == 'w' and pre_generator[1][0] == 'w':
                return 0  # Maps to 0 tensor 0
    
        elif transition_type == 'merge_arc_circle':
            # Check if w tensor v_+ maps to w or 0 in the post-generator
            if set([pre_generator[0][0],pre_generator[1][0]]) == set(['w','v_+']):
                return 1 if post_generator[0][0] == 'w' else 0
    
        elif transition_type == 'split_arc_circle':
            # Check if w maps to w tensor v_-
            if pre_generator[0][0] == 'w':
                return 1 if set([post_generator[0][0],post_generator[1][0]]) == set(['w','v_-']) else 0
    
        return 0  # Default to 0 if no conditions are met
    
    def reorder_generators(self, generators, elements_order):
        """Reorder the generators to match the order of elements in elements_order."""
        reordered = []
        for element in elements_order:
            for gen in generators:
                if gen[1] == element:
                    reordered.append(gen)
                    break
        return reordered
    
    def calculate_all_coefficients(self):
        """Calculate the coefficients for all combinations of pre_generators and post_generators."""
        transition_type = self.get_transition_type()
        coefficients = {}
        
        # Determine the correct order of elements
        pre_order = self.pre_changed + self.pre_unchanged
        post_order = self.post_changed + self.post_unchanged
        
        for pre_gen in self.pre_generators:
            for post_gen in self.post_generators:
                # Reorder the generators to match the order of pre_order and post_order
                reordered_pre_gen = self.reorder_generators(pre_gen, pre_order)
                reordered_post_gen = self.reorder_generators(post_gen, post_order)
                
                # Calculate the coefficient
                coefficient = self.determine_coefficient(reordered_pre_gen, reordered_post_gen, transition_type)
                coefficients[(pre_gen, post_gen)] = coefficient
        
        return coefficients
    
    def generate_mapping_matrix(self):
        """Generate the matrix from pre_state to post_state using the calculated coefficients."""
        num_pre = len(self.pre_generators)
        num_post = len(self.post_generators)
        
        matrix = np.zeros((num_post, num_pre),dtype=int)
        
        for i, post_gen in enumerate(self.post_generators):  
            for j, pre_gen in enumerate(self.pre_generators): 
                matrix[i, j] = self.coefficients[(pre_gen, post_gen)]
                
        k = self.pre_state_string[:self.smoothing_index].count('1')
        
         
        return (-1)**k*matrix

def print_state_map(state_map):
    # Print the transition type
    transition_type = state_map.get_transition_type()
    print(f"Transition Type: {transition_type}\n")
    
    # Print the pre-state and post-state elements
    print("Pre-State Elements:")
    for element in state_map.pre_state.state_elms:
        print(f"  - Type: {element.type}, Representative: {element.representative}")
    
    print("\nPost-State Elements:")
    for element in state_map.post_state.state_elms:
        print(f"  - Type: {element.type}, Representative: {element.representative}")
    
    # Print the calculated coefficients
    print("\nCalculated Coefficients:")
    for (pre_gen, post_gen), coeff in state_map.coefficients.items():
        pre_gen_str = " ⊗ ".join([f"{gen[0]}({gen[1].representative})" for gen in pre_gen])
        post_gen_str = " ⊗ ".join([f"{gen[0]}({gen[1].representative})" for gen in post_gen])
        print(f"  {pre_gen_str} → {post_gen_str} : Coefficient = {coeff}")
        
        
# # Define the pdcode
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
# # pdcode=[
# #     ["1","3","6","4"],
# #     ["5","1","4","2"],
# #     ["3","5","2","6"]
# #     ]

# smoothing_state_generator = SmoothingStateGenerator(pdcode)
# state_map = generate_state_map(smoothing_state_generator, "110", "111")
# coefficients = state_map.coefficients
# print_state_map(state_map)
# print(state_map.mapping_matrix)



