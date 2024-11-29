import numpy as np

def get_tangle(curves,direction):
    tangle = [Curve(points=curve,direction=direction) for curve in curves]
    planar_gen= PlanarGenerator(tangle, direction=direction)
    Crossings = planar_gen.find_crossings()
    
    return tangle,Crossings



def line_intersection(a, b, c, d):
    """
    Determines the parameters t and u for the intersection of two line segments defined by endpoints a, b and c, d.
    """
    denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    
    # Check if denominator is zero (lines are parallel)
    if denominator == 0:
        return None  # Lines are parallel

    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / denominator
    u = ((a[0] - c[0]) * (a[1] - b[1]) - (a[1] - c[1]) * (a[0] - b[0])) / denominator

    # Check if line segments actually intersect within the range [0, 1]
    if (0 <= t < 1) and (0 <= u < 1):
        return t, u  # Return parameters t and u
    else: 
        return None  # No intersection

class Crossing:
    def __init__(self, index, curve1, curve1_segment_index, curve2, curve2_segment_index, t, u):
        """
        Initializes a Crossing object to store crossing information.
        
        Parameters:
        - index: Unique index for the crossing.
        - curve1: The first Curve object.
        - curve1_segment_index: Segment index in the first curve.
        - curve2: The second Curve object.
        - curve2_segment_index: Segment index in the second curve.
        - t: Parameter for the first curve.
        - u: Parameter for the second curve.
        """
        self.index = index
        self.curve1 = curve1
        self.curve1_segment_index = curve1_segment_index
        self.curve2 = curve2
        self.curve2_segment_index = curve2_segment_index
        self.t = t
        self.u = u
        self.handedness = self.calculate_handedness()
        self.OU_position = self.get_position()
        self.update_curves_with_gauss_code()

    def calculate_z_values(self):
        """
        Calculates the z-values for the crossing points in the two curves.
        
        Returns:
        A tuple containing the z-values for curve1 and curve2.
        """
        z_value1 = self.curve1.points[self.curve1_segment_index][2] * (1 - self.t) + \
                   self.curve1.points[self.curve1_segment_index + 1][2] * self.t
        z_value2 = self.curve2.points[self.curve2_segment_index][2] * (1 - self.u) + \
                   self.curve2.points[self.curve2_segment_index + 1][2] * self.u
        return z_value1, z_value2

    def get_position(self):
        """
        Determines the top and bottom positions based on the z-values.
        
        Returns:
        A tuple containing the positions of the two curves ('O' for over and 'U' for under).
        """
        z_value1, z_value2 = self.calculate_z_values()
        position1 = "O" if z_value1 > z_value2 else "U"
        position2 = "U" if position1 == "O" else "O"
        return position1, position2

    def calculate_handedness(self):
        """
        Calculates the handedness of the crossing based on the cross product of the segments.
        
        Returns:
        The handedness sign (-1, 0, 1).
        """
        a = self.curve1.points[self.curve1_segment_index:self.curve1_segment_index + 2]
        b = self.curve2.points[self.curve2_segment_index:self.curve2_segment_index + 2]
        sign = np.sign(np.cross(a[1] - a[0], b[1] - b[0]).dot(a[0] - b[0]))
        if sign == 0:
            print("Warning: The two lines are intersecting in 3D space. Setting handedness to 1.")
            sign = 1
        return sign

    def update_curves_with_gauss_code(self):
        """
        Updates the curves with the crossing Gauss code based on handedness, index, and position.
        """
        gauss_code_curve1 = f"{'+' if self.handedness == 1 else '-'}{self.index}{self.OU_position[0]}"
        gauss_code_curve2 = f"{'+' if self.handedness == 1 else '-'}{self.index}{self.OU_position[1]}"
        self.curve1.gauss_code.append((gauss_code_curve1, self.curve1_segment_index + self.t))
        self.curve2.gauss_code.append((gauss_code_curve2, self.curve2_segment_index + self.u))
        
class Curve:
    def __init__(self, points, direction=None, tolerance=1e-7):
        self.points = np.array(points)
        if self.points.shape[1] != 3:
            raise ValueError("Points must be in 3D space.")
        
        self.is_closed = np.allclose(self.points[0], self.points[-1], atol=tolerance)

        if direction is not None:
            self.projected_points = self.project_onto_plane(direction)
        else:
            self.projected_points = None
        
        self.gauss_code = []  # Initialize a list to collect Gauss codes

    def project_onto_plane(self, direction):
        direction = np.array(direction)
        direction_normalized = direction / np.linalg.norm(direction)

        if np.allclose(direction_normalized, [1, 0, 0]):
            basis1 = np.array([0, 1, 0])  # Arbitrary basis if direction is along x-axis
        else:
            basis1 = np.cross(direction_normalized, [1, 0, 0])
        basis1 /= np.linalg.norm(basis1)

        basis2 = np.cross(direction_normalized, basis1)  # Ensure orthogonality

        projected_points = []
        for point in self.points:
            projection = point - np.dot(point, direction_normalized) * direction_normalized
            x_prime = np.dot(projection, basis1)
            y_prime = np.dot(projection, basis2)
            projected_points.append((x_prime, y_prime))
        
        return np.array(projected_points)

    def get_segments(self):
        """Generates segments from the original 3D points."""
        segments = []
        for i in range(len(self.points) - 1):
            segments.append((self.points[i], self.points[i + 1]))
        return segments

    def get_segments_from_projected_points(self):
        """Generates segments from the projected points (if needed)."""
        segments = []
        for i in range(len(self.projected_points) - 1):
            segments.append((self.projected_points[i], self.projected_points[i + 1]))
        return segments

    def sort_gauss_code(self):
        """Sorts the Gauss code based on the distance position."""
        self.gauss_code.sort(key=lambda x: x[1])
        self.gauss_code = [code for code, _ in self.gauss_code]


    def __repr__(self):
        return f"Curve(points={self.points}, is_closed={self.is_closed}, projected_points={self.projected_points}, gauss_code={self.gauss_code})"

class PlanarGenerator:
    def __init__(self, curves, direction):
        """
        Initializes the PlanarGenerator with multiple curves and a single direction vector.

        Parameters:
        curves (list): A list of Curve objects.
        direction (array-like): A direction vector for projecting the curves onto a plane.
        """
        self.curves = curves
        self.direction = np.array(direction)

        # Project every Curve during initialization
        for curve in self.curves:
            curve.projected_points = curve.project_onto_plane(self.direction)

    def find_crossings(self):
        """
        Finds crossings between segments of two curves, skipping adjacent segments 
        only within the same curve and avoiding duplicate checks. Each crossing is stored in the 
        Gauss code attribute of the respective Curve.
        """
        Crossings = []
        crossing_index = 0
        for i, curve1 in enumerate(self.curves):
            projected_segments1 = curve1.get_segments_from_projected_points()
            for j, curve2 in enumerate(self.curves):
                # Only consider crossings if i <= j to avoid duplicate checks
                if i > j:
                    continue
                
                projected_segments2 = curve2.get_segments_from_projected_points()

                for seg1_index, seg1 in enumerate(projected_segments1):
                    for seg2_index, seg2 in enumerate(projected_segments2):
                        # Skip adjacent segments if comparing within the same curve
                        if i == j and abs(seg1_index - seg2_index) == 1:
                            continue
                        # Skip if segments are the same
                        if i == j and seg1_index >= seg2_index:
                            continue

                        intersection = line_intersection(seg1[0], seg1[1], seg2[0], seg2[1])
                        if intersection is not None:
                            t, u = intersection  # Unpack t and u

                            # Create Crossing object
                            crossing = Crossing(crossing_index, curve1, seg1_index, curve2, seg2_index, t, u)
                            Crossings.append(crossing)
                            crossing_index += 1
        # Sort the Gauss codes for each curve
        for curve in self.curves:
            curve.sort_gauss_code()
        return Crossings



