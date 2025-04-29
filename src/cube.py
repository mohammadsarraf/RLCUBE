import kociemba as koc
import copy

class Cube:
    """
    A class that represents a 3x3 Rubik's cube.
    
    The cube is represented as 6 faces, each with 9 stickers.
    Faces are indexed as follows:
    0: Up (White)
    1: Left (Orange)
    2: Front (Green)
    3: Right (Red)
    4: Back (Blue)
    5: Down (Yellow)
    
    Each face is a 3x3 grid of stickers indexed from 0 to 8:
    0 1 2
    3 4 5
    6 7 8
    """
    
    def __init__(self):
        # Initialize a solved cube
        # Each face has 9 stickers of the same color
        self.faces = [
            ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],  # Up (White)
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],  # Left (Orange)
            ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],  # Front (Green)
            ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],  # Right (Red)
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],  # Back (Blue)
            ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y']   # Down (Yellow)
        ]
    
    def copy(self):
        """Create a deep copy of the cube object."""
        new_cube = Cube()
        new_cube.faces = copy.deepcopy(self.faces)
        return new_cube
    
    def __str__(self):
        """Return a string representation of the cube."""
        result = []
        
        # Up face
        for i in range(0, 9, 3):
            result.append("      " + " ".join(self.faces[0][i:i+3]))
        
        # Left, Front, Right, Back faces side by side
        for row in range(3):
            row_str = []
            for face in range(1, 5):
                row_str.extend(self.faces[face][row*3:row*3+3])
                if face < 4:
                    row_str.append(" ")
            result.append(" ".join(row_str))
        
        # Down face
        for i in range(0, 9, 3):
            result.append("      " + " ".join(self.faces[5][i:i+3]))
            
        return "\n".join(result)
    
    def _rotate_face_clockwise(self, face_idx):
        """Rotate a face clockwise."""
        face = self.faces[face_idx]
        self.faces[face_idx] = [
            face[6], face[3], face[0],
            face[7], face[4], face[1],
            face[8], face[5], face[2]
        ]
    
    def _rotate_face_counterclockwise(self, face_idx):
        """Rotate a face counterclockwise."""
        face = self.faces[face_idx]
        self.faces[face_idx] = [
            face[2], face[5], face[8],
            face[1], face[4], face[7],
            face[0], face[3], face[6]
        ]
    
    def _rotate_face_180(self, face_idx):
        """Rotate a face 180 degrees."""
        face = self.faces[face_idx]
        self.faces[face_idx] = [
            face[8], face[7], face[6],
            face[5], face[4], face[3],
            face[2], face[1], face[0]
        ]
    
    def U(self):
        """Up face clockwise."""
        self._rotate_face_clockwise(0)
        
        # Save the top row of Front
        temp = self.faces[2][:3]
        
        # Front gets top row from Right
        self.faces[2][:3] = self.faces[3][:3]
        
        # Right gets top row from Back
        self.faces[3][:3] = self.faces[4][:3]
        
        # Back gets top row from Left
        self.faces[4][:3] = self.faces[1][:3]
        
        # Left gets saved top row from Front
        self.faces[1][:3] = temp
    
    def U_prime(self):
        """Up face counterclockwise."""
        self._rotate_face_counterclockwise(0)
        
        # Save the top row of Front
        temp = self.faces[2][:3]
        
        # Front gets top row from Left
        self.faces[2][:3] = self.faces[1][:3]
        
        # Left gets top row from Back
        self.faces[1][:3] = self.faces[4][:3]
        
        # Back gets top row from Right
        self.faces[4][:3] = self.faces[3][:3]
        
        # Right gets saved top row from Front
        self.faces[3][:3] = temp
    
    def U2(self):
        """Up face 180 degrees."""
        self._rotate_face_180(0)
        
        # Swap Front and Back top rows
        self.faces[2][:3], self.faces[4][:3] = self.faces[4][:3], self.faces[2][:3]
        
        # Swap Left and Right top rows
        self.faces[1][:3], self.faces[3][:3] = self.faces[3][:3], self.faces[1][:3]
    
    def D(self):
        """Down face clockwise."""
        self._rotate_face_clockwise(5)
        
        # Save the bottom row of Front
        temp = self.faces[2][6:9]
        
        # Front gets bottom row from Left
        self.faces[2][6:9] = self.faces[1][6:9]
        
        # Left gets bottom row from Back
        self.faces[1][6:9] = self.faces[4][6:9]
        
        # Back gets bottom row from Right
        self.faces[4][6:9] = self.faces[3][6:9]
        
        # Right gets saved bottom row from Front
        self.faces[3][6:9] = temp
    
    def D_prime(self):
        """Down face counterclockwise."""
        self._rotate_face_counterclockwise(5)
        
        # Save the bottom row of Front
        temp = self.faces[2][6:9]
        
        # Front gets bottom row from Right
        self.faces[2][6:9] = self.faces[3][6:9]
        
        # Right gets bottom row from Back
        self.faces[3][6:9] = self.faces[4][6:9]
        
        # Back gets bottom row from Left
        self.faces[4][6:9] = self.faces[1][6:9]
        
        # Left gets saved bottom row from Front
        self.faces[1][6:9] = temp
    
    def D2(self):
        """Down face 180 degrees."""
        self._rotate_face_180(5)
        
        # Swap Front and Back bottom rows
        self.faces[2][6:9], self.faces[4][6:9] = self.faces[4][6:9], self.faces[2][6:9]
        
        # Swap Left and Right bottom rows
        self.faces[1][6:9], self.faces[3][6:9] = self.faces[3][6:9], self.faces[1][6:9]
    
    def L(self):
        """Left face clockwise."""
        self._rotate_face_clockwise(1)
        
        # Save the left column of Up
        up_left = [self.faces[0][0], self.faces[0][3], self.faces[0][6]]
        
        # Up gets left column from Back
        self.faces[0][0], self.faces[0][3], self.faces[0][6] = self.faces[4][8], self.faces[4][5], self.faces[4][2]
        
        # Back gets left column from Down
        self.faces[4][8], self.faces[4][5], self.faces[4][2] = self.faces[5][0], self.faces[5][3], self.faces[5][6]
        
        # Down gets left column from Front
        self.faces[5][0], self.faces[5][3], self.faces[5][6] = self.faces[2][0], self.faces[2][3], self.faces[2][6]
        
        # Front gets saved left column from Up
        self.faces[2][0], self.faces[2][3], self.faces[2][6] = up_left
    
    def L_prime(self):
        """Left face counterclockwise."""
        self._rotate_face_counterclockwise(1)
        
        # Save the left column of Up
        up_left = [self.faces[0][0], self.faces[0][3], self.faces[0][6]]
        
        # Up gets left column from Front
        self.faces[0][0], self.faces[0][3], self.faces[0][6] = self.faces[2][0], self.faces[2][3], self.faces[2][6]
        
        # Front gets left column from Down
        self.faces[2][0], self.faces[2][3], self.faces[2][6] = self.faces[5][0], self.faces[5][3], self.faces[5][6]
        
        # Down gets left column from Back
        self.faces[5][0], self.faces[5][3], self.faces[5][6] = self.faces[4][8], self.faces[4][5], self.faces[4][2]
        
        # Back gets saved left column from Up
        self.faces[4][8], self.faces[4][5], self.faces[4][2] = up_left
    
    def L2(self):
        """Left face 180 degrees."""
        self._rotate_face_180(1)
        
        # Swap Up and Down left columns
        up_left = [self.faces[0][0], self.faces[0][3], self.faces[0][6]]
        self.faces[0][0], self.faces[0][3], self.faces[0][6] = self.faces[5][0], self.faces[5][3], self.faces[5][6]
        self.faces[5][0], self.faces[5][3], self.faces[5][6] = up_left
        
        # Swap Front and Back left/right columns
        front_left = [self.faces[2][0], self.faces[2][3], self.faces[2][6]]
        self.faces[2][0], self.faces[2][3], self.faces[2][6] = self.faces[4][8], self.faces[4][5], self.faces[4][2]
        self.faces[4][8], self.faces[4][5], self.faces[4][2] = front_left
    
    def R(self):
        """Right face clockwise."""
        self._rotate_face_clockwise(3)
        
        # Save the right column of Up
        up_right = [self.faces[0][2], self.faces[0][5], self.faces[0][8]]
        
        # Up gets right column from Front
        self.faces[0][2], self.faces[0][5], self.faces[0][8] = self.faces[2][2], self.faces[2][5], self.faces[2][8]
        
        # Front gets right column from Down
        self.faces[2][2], self.faces[2][5], self.faces[2][8] = self.faces[5][2], self.faces[5][5], self.faces[5][8]
        
        # Down gets right column from Back
        self.faces[5][2], self.faces[5][5], self.faces[5][8] = self.faces[4][6], self.faces[4][3], self.faces[4][0]
        
        # Back gets saved right column from Up
        self.faces[4][6], self.faces[4][3], self.faces[4][0] = up_right
    
    def R_prime(self):
        """Right face counterclockwise."""
        self._rotate_face_counterclockwise(3)
        
        # Save the right column of Up
        up_right = [self.faces[0][2], self.faces[0][5], self.faces[0][8]]
        
        # Up gets right column from Back
        self.faces[0][2], self.faces[0][5], self.faces[0][8] = self.faces[4][6], self.faces[4][3], self.faces[4][0]
        
        # Back gets right column from Down
        self.faces[4][6], self.faces[4][3], self.faces[4][0] = self.faces[5][2], self.faces[5][5], self.faces[5][8]
        
        # Down gets right column from Front
        self.faces[5][2], self.faces[5][5], self.faces[5][8] = self.faces[2][2], self.faces[2][5], self.faces[2][8]
        
        # Front gets saved right column from Up
        self.faces[2][2], self.faces[2][5], self.faces[2][8] = up_right
    
    def R2(self):
        """Right face 180 degrees."""
        self._rotate_face_180(3)
        
        # Swap Up and Down right columns
        up_right = [self.faces[0][2], self.faces[0][5], self.faces[0][8]]
        self.faces[0][2], self.faces[0][5], self.faces[0][8] = self.faces[5][2], self.faces[5][5], self.faces[5][8]
        self.faces[5][2], self.faces[5][5], self.faces[5][8] = up_right
        
        # Swap Front and Back right/left columns
        front_right = [self.faces[2][2], self.faces[2][5], self.faces[2][8]]
        self.faces[2][2], self.faces[2][5], self.faces[2][8] = self.faces[4][6], self.faces[4][3], self.faces[4][0]
        self.faces[4][6], self.faces[4][3], self.faces[4][0] = front_right
    
    def F(self):
        """Front face clockwise."""
        self._rotate_face_clockwise(2)
        
        # Save the bottom row of Up
        up_bottom = self.faces[0][6:9]
        
        # Up gets bottom row from Left (right column)
        self.faces[0][6], self.faces[0][7], self.faces[0][8] = self.faces[1][8], self.faces[1][5], self.faces[1][2]
        
        # Left gets right column from Down
        self.faces[1][2], self.faces[1][5], self.faces[1][8] = self.faces[5][0], self.faces[5][1], self.faces[5][2]
        
        # Down gets top row from Right
        self.faces[5][0], self.faces[5][1], self.faces[5][2] = self.faces[3][6], self.faces[3][3], self.faces[3][0]
        
        # Right gets left column from Up
        self.faces[3][0], self.faces[3][3], self.faces[3][6] = up_bottom[0], up_bottom[1], up_bottom[2]
    
    def F_prime(self):
        """Front face counterclockwise."""
        self._rotate_face_counterclockwise(2)
        
        # Save the bottom row of Up
        up_bottom = self.faces[0][6:9]
        
        # Up gets bottom row from Right
        self.faces[0][6], self.faces[0][7], self.faces[0][8] = self.faces[3][0], self.faces[3][3], self.faces[3][6]
        
        # Right gets left column from Down
        self.faces[3][0], self.faces[3][3], self.faces[3][6] = self.faces[5][2], self.faces[5][1], self.faces[5][0]
        
        # Down gets top row from Left
        self.faces[5][0], self.faces[5][1], self.faces[5][2] = self.faces[1][2], self.faces[1][5], self.faces[1][8]
        
        # Left gets right column from Up
        self.faces[1][2], self.faces[1][5], self.faces[1][8] = up_bottom[2], up_bottom[1], up_bottom[0]
    
    def F2(self):
        """Front face 180 degrees."""
        self._rotate_face_180(2)
        
        # Swap Up bottom row and Down top row
        self.faces[0][6:9], self.faces[5][0:3] = list(reversed(self.faces[5][0:3])), list(reversed(self.faces[0][6:9]))
        
        # Swap Left right column and Right left column
        left_right = [self.faces[1][2], self.faces[1][5], self.faces[1][8]]
        right_left = [self.faces[3][0], self.faces[3][3], self.faces[3][6]]
        self.faces[1][2], self.faces[1][5], self.faces[1][8] = right_left[2], right_left[1], right_left[0]
        self.faces[3][0], self.faces[3][3], self.faces[3][6] = left_right[2], left_right[1], left_right[0]
    
    def B(self):
        """Back face clockwise."""
        self._rotate_face_clockwise(4)
        
        # Save the top row of Up
        up_top = self.faces[0][:3]
        
        # Up gets top row from Right
        self.faces[0][0], self.faces[0][1], self.faces[0][2] = self.faces[3][2], self.faces[3][5], self.faces[3][8]
        
        # Right gets right column from Down
        self.faces[3][2], self.faces[3][5], self.faces[3][8] = self.faces[5][8], self.faces[5][7], self.faces[5][6]
        
        # Down gets bottom row from Left
        self.faces[5][6], self.faces[5][7], self.faces[5][8] = self.faces[1][0], self.faces[1][3], self.faces[1][6]
        
        # Left gets left column from Up
        self.faces[1][0], self.faces[1][3], self.faces[1][6] = up_top[2], up_top[1], up_top[0]
    
    def B_prime(self):
        """Back face counterclockwise."""
        self._rotate_face_counterclockwise(4)
        
        # Save the top row of Up
        up_top = self.faces[0][:3]
        
        # Up gets top row from Left
        self.faces[0][0], self.faces[0][1], self.faces[0][2] = self.faces[1][6], self.faces[1][3], self.faces[1][0]
        
        # Left gets left column from Down
        self.faces[1][0], self.faces[1][3], self.faces[1][6] = self.faces[5][6], self.faces[5][7], self.faces[5][8]
        
        # Down gets bottom row from Right
        self.faces[5][6], self.faces[5][7], self.faces[5][8] = self.faces[3][8], self.faces[3][5], self.faces[3][2]
        
        # Right gets right column from Up
        self.faces[3][2], self.faces[3][5], self.faces[3][8] = up_top[0], up_top[1], up_top[2]
    
    def B2(self):
        """Back face 180 degrees."""
        self._rotate_face_180(4)
        
        # Swap Up top row and Down bottom row
        self.faces[0][:3], self.faces[5][6:9] = list(reversed(self.faces[5][6:9])), list(reversed(self.faces[0][:3]))
        
        # Swap Left left column and Right right column
        left_left = [self.faces[1][0], self.faces[1][3], self.faces[1][6]]
        right_right = [self.faces[3][2], self.faces[3][5], self.faces[3][8]]
        self.faces[1][0], self.faces[1][3], self.faces[1][6] = right_right[2], right_right[1], right_right[0]
        self.faces[3][2], self.faces[3][5], self.faces[3][8] = left_left[2], left_left[1], left_left[0]
    
    def M(self):
        """Middle slice clockwise (same direction as L)."""
        # Save the middle column of Up
        up_mid = [self.faces[0][1], self.faces[0][4], self.faces[0][7]]
        
        # Up gets middle column from Back (flipped)
        self.faces[0][1], self.faces[0][4], self.faces[0][7] = self.faces[4][7], self.faces[4][4], self.faces[4][1]
        
        # Back gets middle column from Down (flipped)
        self.faces[4][7], self.faces[4][4], self.faces[4][1] = self.faces[5][1], self.faces[5][4], self.faces[5][7]
        
        # Down gets middle column from Front
        self.faces[5][1], self.faces[5][4], self.faces[5][7] = self.faces[2][1], self.faces[2][4], self.faces[2][7]
        
        # Front gets middle column from Up
        self.faces[2][1], self.faces[2][4], self.faces[2][7] = up_mid
    
    def M_prime(self):
        """Middle slice counterclockwise (opposite direction as L)."""
        # Save the middle column of Up
        up_mid = [self.faces[0][1], self.faces[0][4], self.faces[0][7]]
        
        # Up gets middle column from Front
        self.faces[0][1], self.faces[0][4], self.faces[0][7] = self.faces[2][1], self.faces[2][4], self.faces[2][7]
        
        # Front gets middle column from Down
        self.faces[2][1], self.faces[2][4], self.faces[2][7] = self.faces[5][1], self.faces[5][4], self.faces[5][7]
        
        # Down gets middle column from Back (flipped)
        self.faces[5][1], self.faces[5][4], self.faces[5][7] = self.faces[4][7], self.faces[4][4], self.faces[4][1]
        
        # Back gets middle column from Up (flipped)
        self.faces[4][7], self.faces[4][4], self.faces[4][1] = up_mid
    
    def E(self):
        """Equatorial slice clockwise (same direction as D)."""
        # Save the middle row of Front
        front_mid = self.faces[2][3:6]
        
        # Front gets middle row from Left
        self.faces[2][3:6] = self.faces[1][3:6]
        
        # Left gets middle row from Back
        self.faces[1][3:6] = self.faces[4][3:6]
        
        # Back gets middle row from Right
        self.faces[4][3:6] = self.faces[3][3:6]
        
        # Right gets middle row from Front
        self.faces[3][3:6] = front_mid
    
    def E_prime(self):
        """Equatorial slice counterclockwise (opposite direction as D)."""
        # Save the middle row of Front
        front_mid = self.faces[2][3:6]
        
        # Front gets middle row from Right
        self.faces[2][3:6] = self.faces[3][3:6]
        
        # Right gets middle row from Back
        self.faces[3][3:6] = self.faces[4][3:6]
        
        # Back gets middle row from Left
        self.faces[4][3:6] = self.faces[1][3:6]
        
        # Left gets middle row from Front
        self.faces[1][3:6] = front_mid
    
    def S(self):
        """Standing slice clockwise (same direction as F)."""
        # Save the middle row of Up
        up_mid = self.faces[0][3:6]
        
        # Up gets middle row from Left (rotated)
        self.faces[0][3], self.faces[0][4], self.faces[0][5] = self.faces[1][7], self.faces[1][4], self.faces[1][1]
        
        # Left gets middle row from Down
        self.faces[1][1], self.faces[1][4], self.faces[1][7] = self.faces[5][3], self.faces[5][4], self.faces[5][5]
        
        # Down gets middle row from Right (rotated)
        self.faces[5][3], self.faces[5][4], self.faces[5][5] = self.faces[3][7], self.faces[3][4], self.faces[3][1]
        
        # Right gets middle row from Up
        self.faces[3][1], self.faces[3][4], self.faces[3][7] = up_mid[0], up_mid[1], up_mid[2]
    
    def S_prime(self):
        """Standing slice counterclockwise (opposite direction as F)."""
        # Save the middle row of Up
        up_mid = self.faces[0][3:6]
        
        # Up gets middle row from Right
        self.faces[0][3], self.faces[0][4], self.faces[0][5] = self.faces[3][1], self.faces[3][4], self.faces[3][7]
        
        # Right gets middle row from Down (rotated)
        self.faces[3][1], self.faces[3][4], self.faces[3][7] = self.faces[5][5], self.faces[5][4], self.faces[5][3]
        
        # Down gets middle row from Left
        self.faces[5][3], self.faces[5][4], self.faces[5][5] = self.faces[1][1], self.faces[1][4], self.faces[1][7]
        
        # Left gets middle row from Up (rotated)
        self.faces[1][1], self.faces[1][4], self.faces[1][7] = up_mid[2], up_mid[1], up_mid[0]
    
    # Whole cube rotations
    def x(self):
        """Rotate the entire cube around the x-axis (same as R, M', L')."""
        self.R()
        self.M_prime()
        self.L_prime()
    
    def x_prime(self):
        """Rotate the entire cube around the x-axis counterclockwise."""
        self.R_prime()
        self.M()
        self.L()
    
    def y(self):
        """Rotate the entire cube around the y-axis (same as U, E', D')."""
        self.U()
        self.E_prime()
        self.D_prime()
    
    def y_prime(self):
        """Rotate the entire cube around the y-axis counterclockwise."""
        self.U_prime()
        self.E()
        self.D()
    
    def z(self):
        """Rotate the entire cube around the z-axis (same as F, S, B')."""
        self.F()
        self.S()
        self.B_prime()
    
    def z_prime(self):
        """Rotate the entire cube around the z-axis counterclockwise."""
        self.F_prime()
        self.S_prime()
        self.B()
    
    def is_solved(self):
        """Check if the cube is solved."""
        # For each face, check if all stickers are the same
        return all(all(face[0] == sticker for sticker in face) for face in self.faces)
    
    def scramble(self, num_moves=20):
        """Apply a random sequence of moves to scramble the cube.
        
        Returns:
            tuple: (self, scramble_str) where scramble_str is the applied moves
        """
        import random
        
        moves = [
            self.U, self.U_prime, self.U2,
            self.D, self.D_prime, self.D2,
            self.L, self.L_prime, self.L2,
            self.R, self.R_prime, self.R2,
            self.F, self.F_prime, self.F2,
            self.B, self.B_prime, self.B2
        ]
        
        move_names = [
            "U", "U'", "U2",
            "D", "D'", "D2",
            "L", "L'", "L2",
            "R", "R'", "R2",
            "F", "F'", "F2",
            "B", "B'", "B2"
        ]
        
        # Apply random moves and track them
        applied_moves = []
        for _ in range(num_moves):
            move_idx = random.randrange(len(moves))
            moves[move_idx]()
            applied_moves.append(move_names[move_idx])
        
        # Return both the cube and the scramble sequence
        scramble_str = " ".join(applied_moves)
        return self, scramble_str

    def apply_algorithm(self, algorithm):
        """
        Apply a sequence of moves from a string notation.
        
        Examples:
        - "R U R'" applies R, then U, then R'
        - "F2 B2 L' D" applies F2, then B2, then L', then D
        
        Valid move notations:
        - Basic face turns: U, D, L, R, F, B
        - Inverted turns: U', D', L', R', F', B'
        - Double turns: U2, D2, L2, R2, F2, B2
        - Middle slice moves: M, E, S and their inverses/doubles
        - Whole cube rotations: x, y, z and their inverses
        """
        # Dictionary mapping move notation to corresponding method
        move_map = {
            "U": self.U,
            "U'": self.U_prime,
            "U2": self.U2,
            "D": self.D,
            "D'": self.D_prime,
            "D2": self.D2,
            "L": self.L,
            "L'": self.L_prime,
            "L2": self.L2,
            "R": self.R,
            "R'": self.R_prime,
            "R2": self.R2,
            "F": self.F,
            "F'": self.F_prime,
            "F2": self.F2,
            "B": self.B,
            "B'": self.B_prime,
            "B2": self.B2,
            "M": self.M,
            "M'": self.M_prime,
            "E": self.E,
            "E'": self.E_prime,
            "S": self.S,
            "S'": self.S_prime,
            "x": self.x,
            "x'": self.x_prime,
            "y": self.y,
            "y'": self.y_prime,
            "z": self.z,
            "z'": self.z_prime
        }
        
        # Split the algorithm into individual moves
        moves = algorithm.split()
        
        # Apply each move
        for move in moves:
            if move in move_map:
                move_map[move]()
            else:
                raise ValueError(f"Invalid move notation: {move}")
        
        return self

    def to_kociemba_string(self):
        """
        Convert cube state to Kociemba string notation.
        
        Kociemba uses the following conventions:
        - Each face is represented by its center color
        - The order of faces is: Up, Right, Front, Down, Left, Back
        - Each face is read from top-left to bottom-right
        
        Returns:
            str: A 54-character string representing the cube state
        """
        # Mapping from our internal face representation to Kociemba order
        # Our order: Up(0), Left(1), Front(2), Right(3), Back(4), Down(5)
        # Kociemba order: Up(0), Right(3), Front(2), Down(5), Left(1), Back(4)
        kociemba_face_order = [0, 3, 2, 5, 1, 4]
        
        # Get the center color of each face to map color to face in Kociemba notation
        centers = [face[4] for face in self.faces]
        
        # Map our colors to Kociemba's URFDLB face representation
        # In Kociemba: U=white, R=green, F=red, D=yellow, L=blue, B=orange
        # This mapping ensures compatibility with the solver
        color_map = {
            'W': 'U',  # Up is White
            'R': 'R',  # Right is Red
            'G': 'F',  # Front is Green
            'Y': 'D',  # Down is Yellow
            'O': 'L',  # Left is Orange
            'B': 'B'   # Back is Blue
        }
        
        # Construct the Kociemba string with mapped colors
        kociemba_str = ""
        for face_idx in kociemba_face_order:
            for color in self.faces[face_idx]:
                kociemba_str += color_map[color]
            
        return kociemba_str
    
    def verify_kociemba_compatibility(self):
        """
        Verify that the cube state is valid for Kociemba solver.
        
        The Kociemba solver requires:
        1. Exactly one center piece of each color
        2. Exactly 4 edge pieces of each color
        3. Exactly 4 corner pieces of each color
        4. Properly oriented and permuted pieces (solvable state)
        
        Returns:
            bool: True if the cube state is valid for Kociemba, False otherwise
        """
        kociemba_str = self.to_kociemba_string()
        
        # 1. Check string length (must be 54 characters)
        if len(kociemba_str) != 54:
            return False
        
        # 2. Count occurrences of each color (must be exactly 9 of each)
        color_counts = {}
        for color in kociemba_str:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        if any(count != 9 for count in color_counts.values()) or len(color_counts) != 6:
            return False
        
        # 3. Check that centers are different colors
        centers = [
            kociemba_str[4],   # Up center
            kociemba_str[13],  # Right center
            kociemba_str[22],  # Front center
            kociemba_str[31],  # Down center
            kociemba_str[40],  # Left center
            kociemba_str[49]   # Back center
        ]
        
        if len(set(centers)) != 6:
            return False
        
        # Note: Full validation of a Rubik's cube state is complex and would
        # require checking corner and edge piece parity, which is quite involved.
        # This function performs basic checks only.
        
        return True
        
    def solve_with_kociemba(self):
        """
        Generate a solution using the Kociemba algorithm.
        
        Returns:
            str: A sequence of moves that solves the cube, or an error message
                 if no solution is found
        
        Note: This requires the 'kociemba' package to be installed.
        Install it using: pip install kociemba
        """
        try:
            import kociemba
            
            # Verify that the cube state is valid for Kociemba
            if not self.verify_kociemba_compatibility():
                return "Error: Cube state is not valid for Kociemba solver"
            
            # Get the cube state in Kociemba format
            kociemba_str = self.to_kociemba_string()
            
            # Get the solution
            solution = kociemba.solve(kociemba_str)
            
            return solution
        except ImportError:
            return "Error: kociemba package is not installed. Install it using: pip install kociemba"
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Try to import the solve_from_input function from advanced_solver
    try:
        from advanced_solver import solve_from_input
        # Call the function to get input and solve the cube
        solve_from_input()
    except ImportError:
        print("Error: Could not import solve_from_input from advanced_solver.py")
        print("Make sure advanced_solver.py is in the same directory.")
        # Fall back to basic input
        scramble = input("Enter a scramble: ")
        cube = Cube()
        try:
            cube.apply_algorithm(scramble)
            print(f"Applied scramble: {scramble}")
            print("Cube state:")
            print(cube)
            
            # Try to solve with kociemba as fallback
            solution = cube.solve_with_kociemba()
            print(f"\nSolution: {solution}")
        except Exception as e:
            print(f"Error: {str(e)}")