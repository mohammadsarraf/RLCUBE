import unittest
from cube import Cube
import kociemba as koc

class TestCubeIsSolved(unittest.TestCase):
    
    def test_is_solved(self):
        """Test the is_solved method of the Cube class."""
        # Create a new cube (should be solved)
        cube = Cube()
        solution = koc.solve(cube.to_kociemba_string())
        cube.apply_algorithm(solution)
        self.assertTrue(cube.is_solved())
        
        # Make a move to scramble the cube
        cube.apply_algorithm("R U R' F' R U R' U' R' F R2 U' R' U'")
        cube.apply_algorithm("R U R' F' R U R' U' R' F R2 U' R' U'")
        solution = koc.solve(cube.to_kociemba_string())
        cube.apply_algorithm(solution)
        self.assertTrue(cube.is_solved())
        
        # Sexy Move 6 times seprately
        cube.apply_algorithm("R U R' U'")
        cube.apply_algorithm("R U R' U'")
        cube.apply_algorithm("R U R' U'")
        cube.apply_algorithm("R U R' U'")
        cube.apply_algorithm("R U R' U'")
        cube.apply_algorithm("R U R' U'")
        solution = koc.solve(cube.to_kociemba_string())
        cube.apply_algorithm(solution)
        self.assertTrue(cube.is_solved())
        
        # Sexy Move 6 times in one call
        cube.apply_algorithm("R U R' U' R U R' U' R U R' U' R U R' U' R U R' U' R U R' U'")
        solution = koc.solve(cube.to_kociemba_string())
        cube.apply_algorithm(solution)
        self.assertTrue(cube.is_solved())

        # Random scramble
        cube.scramble(20)
        solution = koc.solve(cube.to_kociemba_string())
        cube.apply_algorithm(solution)
        self.assertTrue(cube.is_solved())

if __name__ == "__main__":
    unittest.main()
