# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:38:12 2024

@author: lshen
"""
def close_components(gauss_code, is_closed_list):
    closed_gauss_code = gauss_code.copy()
    for comp_idx, component in enumerate(closed_gauss_code):
        if is_closed_list[comp_idx]:
            # Duplicate and attach the first code to the end of the component
            if component:  # Ensure the component is not empty
                component.append(component[0])
    return closed_gauss_code

def remove_last_for_closed(closed_gauss_code, is_closed_list):
    gauss_code = closed_gauss_code.copy()
    for comp_idx, component in enumerate(gauss_code):
        if is_closed_list[comp_idx] and component:  # Ensure the component is closed and not empty
            component.pop()  # Remove the last code
    return gauss_code


def reorder_gauss_code(gauss_code):
    # Collect unique crossing values from the Gauss code
    unique_crossings = sorted(set(int(code[1:-1]) for comp in gauss_code for code in comp))
    crossing_map = {old_val: new_val for new_val, old_val in enumerate(unique_crossings)}

    # Apply the new ordering to the Gauss code
    reordered_gauss_code = []
    for comp in gauss_code:
        reordered_comp = [
            f"{code[0]}{crossing_map[int(code[1:-1])]}{code[-1]}" for code in comp
        ]
        reordered_gauss_code.append(reordered_comp)

    return reordered_gauss_code

def remove_and_reorder(gauss_code, crossing_values):
    # Remove specified crossings from all components
    for comp_idx, component in enumerate(gauss_code):
        gauss_code[comp_idx] = [code for code in component if code[1:-1] not in crossing_values]
    
    # Reorder the Gauss code after removal
    reordered_gauss_code = reorder_gauss_code(gauss_code)
    return reordered_gauss_code

def R1_move(gauss_code):
    for comp_idx, component in enumerate(gauss_code):
        for i in range(len(component) - 1):
            # Check if two adjacent codes have the same numerical value and different positions
            if component[i][1:-1] == component[i + 1][1:-1] and component[i][-1] != component[i + 1][-1]:
                # Get the crossing value to remove
                crossing_value = component[i][1:-1]
                # Use the remove_and_reorder function to remove and reorder the Gauss code
                reordered_gauss_code = remove_and_reorder(gauss_code, {crossing_value})
                # Get the sign of the removed crossing
                sign = component[i][0]
                return sign, reordered_gauss_code  # Return removed sign and reordered Gauss code
    return None, gauss_code  # If no R1_move is possible

def R2_move(gauss_code):
    r2_waitlist=[]
    for comp_idx, component in enumerate(gauss_code):
        i = 0
        for i in range(len(component) - 1):
            # Check if two adjacent codes have the same position and different numerical values
            if component[i][-1] == component[i + 1][-1] and component[i][1:-1] != component[i + 1][1:-1]:
                crossing_set = {component[i][1:-1], component[i + 1][1:-1]}
                
                if crossing_set in r2_waitlist:
                    # Use the remove_and_reorder function to remove and reorder the Gauss code
                    reordered_gauss_code = remove_and_reorder(gauss_code, crossing_set)
                    return None, reordered_gauss_code  # Return None and remaining Gauss code
                else:
                    # Add the set to R2_waitlist if not found
                    r2_waitlist.append(crossing_set)
                    
    return None, gauss_code  # If no R2_move is possible


def do_R1_R2(gauss_code):
    plus_count = 0  # Counter for "+" signs
    minus_count = 0  # Counter for "-" signs
    
    while True:
        # Initial length of Gauss code
        initial_length = sum(len(comp) for comp in gauss_code)
        
        # Perform R1 moves until no change in length
        while True:
            removed_sign, new_gauss_code = R1_move(gauss_code)
            if removed_sign == "+":
                plus_count += 1
            elif removed_sign == "-":
                minus_count += 1

            if sum(len(comp) for comp in new_gauss_code) == initial_length:
                break  # Stop R1 moves if no change in length
            gauss_code = new_gauss_code  # Update the Gauss code
            initial_length = sum(len(comp) for comp in gauss_code)  # Update initial length
        
        # Perform R2 moves until no change in length
        while True:
            _, new_gauss_code = R2_move(gauss_code)
            if sum(len(comp) for comp in new_gauss_code) == initial_length:
                break  # Stop R2 moves if no change in length
            gauss_code = new_gauss_code  # Update the Gauss code
            initial_length = sum(len(comp) for comp in gauss_code)  # Update initial length
        
        # Perform one-time R1 and R2 moves to check for changes
        removed_sign_r1, new_gauss_code_r1 = R1_move(gauss_code)
        _, new_gauss_code_r2 = R2_move(gauss_code)
        
        # Update sign counters for one-time R1 move
        if removed_sign_r1 == "+":
            plus_count += 1
        elif removed_sign_r1 == "-":
            minus_count += 1

        # Check if no change occurs in both R1 and R2 moves
        if (new_gauss_code_r1 == gauss_code) and (new_gauss_code_r2 == gauss_code):
            break  # Exit if no change occurs in both moves
        
        # Update the Gauss code to the latest changes
        gauss_code = new_gauss_code_r1 if new_gauss_code_r1 != gauss_code else new_gauss_code_r2

    return gauss_code, plus_count, minus_count

def simplify(gauss_code, is_closed_list):
    # Step 1: Close the components if specified in is_closed_list
    closed_gauss_code = close_components(gauss_code, is_closed_list)
    
    # Step 2: Perform iterative R1 and R2 moves to simplify the Gauss code
    simplified_gauss_code, positive_R1, negative_R1 = do_R1_R2(closed_gauss_code)
    
    # Step 3: Remove the last position for closed components to obtain the final format
    final_gauss_code = remove_last_for_closed(simplified_gauss_code, is_closed_list)
    
    return final_gauss_code,positive_R1,negative_R1
