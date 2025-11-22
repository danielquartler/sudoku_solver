# # test_sudoku_reader.py
import unittest
import csv
import os
from typing import List, Optional
import json
from sudoku_extractor_from_image import solve_sudoku_from_image

# Input
input_csv_file = 'unit_test_data\\test_cases.csv'

class SudokuPuzzleReader:

    # Read a Sudoku puzzle from an image.
    def read_puzzle(self, image_path: str) -> List[int]:
        """ Input:  image_path: Path to the puzzle image
            Output:  List of 81 integers representing the puzzle (0 for empty cells)
        """
        board = solve_sudoku_from_image(img_path=image_path, doSolvePuzzle=False, doVisualization=False)
        return board.flatten().tolist()


# Unit tests for Sudoku puzzle reader
class TestSudokuPuzzleReader(unittest.TestCase):

    # Set up test fixtures that are used by all tests
    @classmethod
    def setUpClass(cls):
        cls.csv_file = input_csv_file
        cls.reader = SudokuPuzzleReader()
        cls.test_cases = cls.load_test_cases()

        # Statistics
        cls.total_tests = 0
        cls.passed_tests = 0
        cls.failed_tests = 0
        cls.error_tests = 0


    # Load test cases from CSV file.
    @classmethod
    def load_test_cases(cls) -> List[tuple]:
        """ Required CSV format: image_path, digit_0, digit_1, ..., digit_80
        Output:
            List of tuples: (image_path, expected_values) """
        if not os.path.exists(cls.csv_file):
            raise FileNotFoundError(f"Test cases file not found: {cls.csv_file}")

        test_cases = []
        print('\n')
        with open(cls.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            #header = next(reader, None)  # Skip header if exists

            for row_num, row in enumerate(reader, start=1):  # use start=2 when there is header
                if len(row) < 82:  # 1 path + 81 digits
                    print(f"Warning: Row {row_num} has insufficient columns ({len(row)}), skipping")
                    continue

                image_path = row[0].strip()
                grid_values = [int(val.strip()) for val in row[1:]]
                print(f"Row {row_num}: {image_path}, grid has {len(grid_values)} values")

                try:
                    # Parse the 81 digit values
                    expected_values = [int(val) for val in row[1:82]]

                    if len(expected_values) != 81:
                        print(f"Warning: Row {row_num} does not have exactly 81 digits, skipping")
                        continue

                    # Validate digits are in range 0-9
                    if not all(0 <= digit <= 9 for digit in expected_values):
                        print(f"Warning: Row {row_num} contains invalid digits (not in 0-9), skipping")
                        continue

                    test_cases.append((image_path, expected_values))

                except ValueError as e:
                    print(f"Warning: Row {row_num} has invalid digit values: {e}, skipping")
                    continue

        print(f"\nLoaded {len(test_cases)} test cases from {cls.csv_file}")
        return test_cases


    # Compare two puzzle solutions
    def compare_puzzles(self, actual: List[int], expected: List[int]) -> tuple:
        """ Output: (is_match, differences, accuracy) """
        # init
        is_match = False
        differences = []  # a dictionary contains all the data regarding the discrepancies.
        accuracy = 0.0
        cMatches = 0  # counts number of matches

        # validate input
        if len(actual) != 81 or len(expected) != 81:
            return is_match, differences, accuracy

        # for each cell:
        for i in range(81):
            if actual[i] == expected[i]:
                cMatches += 1
            else:
                row = i // 9
                col = i % 9
                differences.append({'position': i, 'row': row, 'col': col, 'expected': expected[i], 'actual': actual[i]})

        accuracy = (cMatches / 81) * 100
        is_match = (cMatches == 81)

        return is_match, differences, accuracy


    # Format sudoku puzzle board for display
    def representative_string(self, values: List[int]) -> str:
        lines = []
        for i in range(0, 81, 9):
            row = values[i:i + 9]
            row_str = ' '.join([str(d) if d != 0 else '.' for d in row])
            lines.append(row_str)
            if (i // 9 + 1) % 3 == 0 and i < 72:
                lines.append('-' * 17)
        return '\n'.join(lines)


    # Test all puzzles from the CSV file
    def test_all_puzzles(self):
        if not self.test_cases:
            self.skipTest("No test cases loaded")

        results = []

        for idx, (image_path, expected_values) in enumerate(self.test_cases, start=1):
            self.total_tests += 1

            with self.subTest(test_case=idx, image=image_path):
                # Check if image file exists
                if not os.path.exists(image_path):
                    self.error_tests += 1
                    self.fail(f"Image file not found: {image_path}")
                    results.append({
                        'test_id': idx,
                        'image_path': image_path,
                        'status': 'ERROR',
                        'reason': 'File not found'
                    })
                    continue

                try:
                    # Read puzzle
                    actual_values = self.reader.read_puzzle(image_path)

                    # Validate output
                    self.assertIsInstance(actual_values, list, f"Reader must return a list")
                    self.assertEqual(len(actual_values), 81, f"Reader must return exactly 81 values")
                    self.assertTrue(all(isinstance(v, int) for v in actual_values), f"All values must be integers")
                    self.assertTrue(all(0 <= v <= 9 for v in actual_values), f"All values must be in range 0-9")

                    # Compare puzzles
                    is_match, differences, accuracy = self.compare_puzzles(actual_values, expected_values)

                    # assemble result:
                    result = {
                        'test_id': idx,
                        'image_path': image_path,
                        'accuracy': accuracy,
                        'differences': len(differences)
                    }
                    if is_match:
                        self.passed_tests += 1
                        result['status'] = 'PASS'
                        print(f" Test {idx}/{len(self.test_cases)}: PASS - {image_path}")
                    else:
                        self.failed_tests += 1
                        result['status'] = 'FAIL'
                        result['error_details'] = differences

                        # Create detailed failure message
                        msg = [f" Test {idx}/{len(self.test_cases)}: FAIL - {image_path}"]
                        msg.append(f"\nDifferences found at {len(differences)} positions:")
                        msg.append("\nExpected grid:")
                        msg.append(self.representative_string(expected_values))
                        msg.append("\nActual grid:")
                        msg.append(self.representative_string(actual_values))
                        print('\n'.join(msg))
                        self.fail('\n'.join(msg))

                    results.append(result)

                except Exception as e:
                    self.error_tests += 1
                    result = {
                        'test_id': idx,
                        'image_path': image_path,
                        'status': 'ERROR',
                        'reason': str(e)
                    }
                    results.append(result)

                    msg = f"\n✗ Test {idx}/{len(self.test_cases)}: ERROR - {image_path}\n{str(e)}"
                    print(msg)
                    self.fail(msg)

        # Save results
        self.save_results(results)


    # Save test results to a JSON file
    def save_results(self, results: List[dict]):
        output_file = 'test_results.json'

        summary = {
            'total_tests': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'errors': self.error_tests,
            'pass_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nTest results saved to {output_file}")


    # Print summary after all tests
    @classmethod
    def tearDownClass(cls):
        if cls.total_tests > 0:
            print("\n" + "=" * 70)
            print("TEST SUMMARY")
            print("=" * 70)
            print(f"Total tests:  {cls.total_tests}")
            print(f"Passed:       {cls.passed_tests} ({cls.passed_tests / cls.total_tests * 100:.1f}%)")
            print(f"Failed:       {cls.failed_tests} ({cls.failed_tests / cls.total_tests * 100:.1f}%)")
            print(f"Errors:       {cls.error_tests} ({cls.error_tests / cls.total_tests * 100:.1f}%)")
            print("=" * 70)


# Individual test methods for specific scenarios
class TestSudokuReaderIndividual(unittest.TestCase):

    # Set up for each test
    def setUp(self):
        self.reader = SudokuPuzzleReader()

    # Test that reader returns a list
    def test_reader_returns_list(self):
        # This test requires at least one valid image
        csv_file = input_csv_file
        if not os.path.exists(csv_file):
            self.skipTest(f"Test cases file not found: {csv_file}")

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            row = next(reader, None)

            if row is None:
                self.skipTest("No test cases in CSV")

            image_path = row[0].strip()

            if not os.path.exists(image_path):
                self.skipTest(f"Image not found: {image_path}")

            try:
                result = self.reader.read_puzzle(image_path)
                self.assertIsInstance(result, list)
            except NotImplementedError:
                self.skipTest("Reader not implemented")

    # Test that reader returns exactly 81 values
    def test_reader_returns_81_values(self):
        csv_file = input_csv_file
        if not os.path.exists(csv_file):
            self.skipTest(f"Test cases file not found: {csv_file}")

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            row = next(reader, None)

            if row is None:
                self.skipTest("No test cases in CSV")

            image_path = row[0].strip()

            if not os.path.exists(image_path):
                self.skipTest(f"Image not found: {image_path}")

            try:
                result = self.reader.read_puzzle(image_path)
                self.assertEqual(len(result), 81,
                                 f"Expected 81 values, got {len(result)}")
            except NotImplementedError:
                self.skipTest("Reader not implemented")

    # Test that reader returns only valid digits (0-9)
    def test_reader_returns_valid_digits(self):
        csv_file = input_csv_file
        if not os.path.exists(csv_file):
            self.skipTest(f"Test cases file not found: {csv_file}")

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            row = next(reader, None)

            if row is None:
                self.skipTest("Line is empty (skipped test case)")
                return

            image_path = row[0].strip()
            if not os.path.exists(image_path):
                self.skipTest(f"Image not found: {image_path}")
                return

            try:
                result = self.reader.read_puzzle(image_path)
                self.assertTrue(all(isinstance(v, int) for v in result), "All values must be integers")
                self.assertTrue(all(0 <= v <= 9 for v in result), "All values must be in range 0-9")

            except NotImplementedError:
                self.skipTest("Reader not implemented")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
    print("Finished")
