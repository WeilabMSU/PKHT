# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:00:04 2024

@author: lshen
"""
from get_gausscode import *
from simplify_gausscode import simplify
def generate_pd_crossings(gauss_codes, is_closed_list):
    """
    Generates PDcrossings objects based on the Gauss codes.
    
    Parameters:
    - gauss_codes (list): A list of Gauss codes.
    - is_closed_list (list): A list of boolean values indicating if the corresponding curve is closed.
    
    Returns:
    - list: A list of PDcrossing objects containing the crossing information.
    """
    pd_crossings = []
    cur_line_index = 0
    for gauss_code, is_closed in zip(gauss_codes, is_closed_list):
        cur_line_index += 1
        init_index = cur_line_index
        for i, code in enumerate(gauss_code):
            sign = code[0]  # '+' or '-'
            crossing_index = int(code[1:-1])  # The index of the crossing
            position = code[-1]  # 'O' for over or 'U' for under

            # Create a PDcrossing object if it doesn't exist
            pd_crossing = next((crossing for crossing in pd_crossings if crossing.index == crossing_index), None)
            if pd_crossing is None:
                pd_crossing = PDcrossing(crossing_index, sign)
                pd_crossings.append(pd_crossing)

            # Fill input information
            current_segment = f"|{cur_line_index}" if not is_closed and i == 0 else str(cur_line_index)
            if position == 'O':
                pd_crossing.Oi = current_segment
            elif position == 'U':
                pd_crossing.Ui = current_segment

            # Update the segment index for the next crossing
            
            # Fill output information
            if is_closed:
                if  i == len(gauss_code) - 1:
                    next_segment = str(init_index)
                else:
                    cur_line_index +=1
                    next_segment = str(cur_line_index)
            else:
                cur_line_index += 1
                if i ==len(gauss_code)-1:
                    next_segment = f"{cur_line_index}|"
                else:
                    next_segment = str(cur_line_index)

            if position == 'O':
                pd_crossing.Oo = next_segment
            elif position == 'U':
                pd_crossing.Uo = next_segment
    
    return pd_crossings


class PDcode:
    def __init__(self, curves,simplified=True):
        """
        Initializes the PDcode with the given curves.
        
        Parameters:
        - curves (list): A list of Curve objects.
        """
        self.curves = curves
        self.qcircle_count = 0  # Count of qcircles (closed curves with no crossings)
        self.qarc_count = 0  # Count of qarcs (open curves with no crossings)
        self.pd_crossings = []  # List to store PDcrossings objects
        self.positive_R1=0
        self.negative_R1=0
        self.gauss_codes = gauss_codes = [curve.gauss_code for curve in curves]
        self.raw_gauss_codes=self.gauss_codes
        self.is_closed_list = [curve.is_closed for curve in curves]
        # simplify gauss_codes
        if simplified:
            self.gauss_codes,self.positive_R1,self.negative_R1=simplify(self.gauss_codes, self.is_closed_list)


        # Count qcircles and qarcs
        for gauss_code, is_closed in zip(self.gauss_codes, self.is_closed_list):
            if not gauss_code:# if gauss_code != []
                if is_closed:
                    self.qcircle_count += 1
                else:
                    self.qarc_count += 1

        # Generate PDcrossings using an independent function
        

            
        self.pd_crossings = generate_pd_crossings(self.gauss_codes, self.is_closed_list)
        for crossing in self.pd_crossings:
            if crossing.sign == '+':
                crossing.code = [crossing.Ui, crossing.Oo, crossing.Uo, crossing.Oi]
            else:
                crossing.code = [crossing.Ui, crossing.Oi, crossing.Uo, crossing.Oo]
        # Calculate the number of positive and negative crossings
        self.positive_crossings = int(sum(1 for gauss_code in self.gauss_codes for code in gauss_code if code[0] == '+')/2)
        self.negative_crossings = int(sum(1 for gauss_code in self.gauss_codes for code in gauss_code if code[0] == '-')/2)
    def get_pdcode(self):
        """
        Returns the PDcode information for all crossings.
        
        Returns:
        - list: A list of dictionaries representing the PDcode for each crossing.
        """
        return [crossing.code for crossing in self.pd_crossings]        



class PDcrossing:
    def __init__(self, index, sign):
        """
        Initializes a PDcrossing object.
        
        Parameters:
        - index (int): The index of the crossing.
        - sign (str): The sign of the crossing ('+' or '-').
        """
        self.index = index
        self.sign = sign  # Sign of the crossing
        self.Oi = None  # Over input
        self.Ui = None  # Under input
        self.Oo = None  # Over output
        self.Uo = None  # Under output

    def to_dict(self):
        """
        Returns the PDcrossing information as a dictionary.
        
        Returns:
        - dict: A dictionary containing the PDcrossing information.
        """
        return {
            'index': self.index,
            'sign': self.sign,
            'Oi': self.Oi,
            'Ui': self.Ui,
            'Oo': self.Oo,
            'Uo': self.Uo
        }


