# sudoku_solver
This Python project implements a complete pipeline for solving Sudoku puzzles from images. It combines computer vision techniques with classic backtracking logic to detect, interpret, and solve Sudoku boards.

Main Features:

1. Image Preprocessing:  
   Reads an image of a Sudoku puzzle and applies preprocessing to locate and isolate the Sudoku grid.

2. Grid & Cell Extraction:  
   Detects the structure of the board, divides it into 9x9 cells, and prepares each cell for digit analysis.

3. Digit Detection & Classification:  
   For each cell, the software determines whether it contains a digit. If so, it classifies the digit using a trained digit recognizer (CNN).

4. Sudoku Solving:  
   Uses constraint propagation and backtracking to solve the puzzle logically.

5. Solution Visualization:  
   Overlays the solved digits back onto the original image for visualization.

Technologies Used:  
- OpenCV (image processing)  
- NumPy  
- ensorFlow (for digit classification)  
- Standard logic for solving
