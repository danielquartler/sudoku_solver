""" extract_puzzle_boundaries_from_screen.py
Sudoku puzzle boundary extraction from images.

This module provides functionality to detect and extract Sudoku grids from photographs,
applying perspective correction to produce a clean top-down view.    """
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Configuration parameters for the Sudoku extractor.
@dataclass
class ExtractorConfig:
    debug_mode: bool = False

    output_size: int = 450
    input_width: int = 480
    input_height: int = 640
    threshold_value: int = 150
    min_line_length_initial: int = 400
    line_length_reduction_factor: float = 0.9

    # Morphology kernel sizes
    open_kernel_size: int = 8
    close_kernel_size1: int = 12
    close_kernel_size2: int = 9

    # Histogram equalization percentiles
    hist_lower_percentile: float = 5.0
    hist_upper_percentile: float = 95.0

    # corner enlargement
    enlarge_ratio: float = 8  # [%]


# Return line coefficients (a, b, c) for ax + by + c = 0
def line_equation(line):
    x1, y1, x2, y2 = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, -c

def merge_lines(cur_Horizontal):
    v_lines_abc = [line_equation(line) for line in cur_Horizontal]
    nHorz = np.size(cur_Horizontal, 0)
    inds = np.arange(nHorz)
    for k0 in inds:
        for k1 in inds:
            if k0 != k1:
                dist1 = np.abs(cur_Horizontal[k0, 0] - cur_Horizontal[k1, 2])
                dist2 = np.abs(cur_Horizontal[k0, 1] - cur_Horizontal[k1, 3])
                if dist1 < 5 and dist2 < 5:
                    # if np.abs(v_lines_abc[k0][0] - v_lines_abc[k1][0]) < 25 and  np.abs(v_lines_abc[k0][1] - v_lines_abc[k1][1]) < 25:
                    if np.abs(v_lines_abc[k0][0]-v_lines_abc[k1][0]) + np.abs(v_lines_abc[k0][1]-v_lines_abc[k1][1]) < 90:
                        inds = np.hstack((range(k1), range(k1 + 1, nHorz))).astype(int)  # remove k1
                        cur_Horizontal[k0, :2] = cur_Horizontal[k1, :2]
                        cur_Horizontal = cur_Horizontal[inds, :]
                        return True, cur_Horizontal

    return False, cur_Horizontal


# Extract and rectify Sudoku puzzles from images.
class SudokuExtractor:
    """    This class processes an input image to locate the Sudoku grid boundaries,
    then applies a perspective transform to produce a clean, square output image.

    Attributes:
        config: Configuration parameters for the extraction process.
        original_image: The loaded BGR image.
        grayscale_image: Grayscale version of the input.
        warped_grayscale: Perspective-corrected grayscale output.
        warped_color: Perspective-corrected color output.
    """

    # Initialize the extractor with an image
    def __init__(self, image_path, config: Optional[ExtractorConfig] = None):
        """ Input:
            image_path: Path to the input image file.
            config: Optional configuration object. Uses defaults if not provided.
        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be loaded.
        """
        self.config = config or ExtractorConfig()
        self.image_path = Path(image_path)

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.original_image = cv2.resize(self.original_image, (self.config.input_width, self.config.input_height),interpolation=cv2.INTER_AREA)
        self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.warped_grayscale: Optional[np.ndarray] = None
        self.warped_color: Optional[np.ndarray] = None


    # Save an image for debugging purposes if debug mode is enabled
    def _save_debug_image(self, name: str, image: np.ndarray) -> None:
        if self.config.debug_mode:
            cv2.imwrite(name, image)


    # Create a clean binary mask highlighting the grid structure
    def _create_binary_mask(self, grayscale: np.ndarray) -> np.ndarray:
        """ Input:
            grayscale: Input grayscale image with equalized histogram.
        Output:
            Binary mask with noise removed and holes filled.    """

        # apply threshold
        _, binary_mask = cv2.threshold(grayscale, self.config.threshold_value, 255, cv2.THRESH_BINARY_INV)
        self._save_debug_image("debug_01_threshold_INV.png", binary_mask)

        # Remove small noise with morphological opening
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.open_kernel_size, self.config.open_kernel_size))
        mask_opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
        self._save_debug_image("debug_02_opened.png", mask_opened)
        mask_opened_INV = 255-mask_opened

        # Fill small holes with morphological closing
        close_kernel = np.ones((self.config.close_kernel_size1, self.config.close_kernel_size2), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_opened_INV, cv2.MORPH_CLOSE, close_kernel)
        self._save_debug_image("debug_03_cleaned.png", mask_cleaned)
        self.mask_cleaned = mask_cleaned  # 4 debug

        return mask_cleaned

    # to remove:
    # Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # Detect the four corners of the Sudoku grid using line detection
    def _detect_grid_corners(self, mask):
        """ Input:
            mask: Binary mask of the grid area.
        Output:
            corners: Array of shape (4, 2) containing corner coordinates. """
        lsd = cv2.createLineSegmentDetector()
        lines, _, _, _ = lsd.detect(mask)

        if lines is None or len(lines) == 0:
            raise ValueError("No lines detected in the image")

        # calculate line dimensions
        num_lines = lines.shape[0]
        line_dimensions = np.zeros((num_lines, 2))
        for i in range(num_lines):
            line = lines[i, 0]
            line_dimensions[i, 0] = abs(line[2] - line[0])  # horizontal extent
            line_dimensions[i, 1] = abs(line[3] - line[1])  # vertical extent

        # keep only long lines
        dist1 = np.sum(line_dimensions, 1)
        minDist = 50
        i2keep = []
        while len(i2keep)==0:
            i2keep = np.where(dist1 > minDist)[0]
            minDist = 0.9*minDist
        # keep only long lines
        line_dimensions = line_dimensions[i2keep, :]
        lines = lines[i2keep, :, :]
        num_lines = lines.shape[0]

        # separate to vertical and horizontal lines
        vertical, horizontal = [], []
        for idx in range(num_lines):
            if line_dimensions[idx, 0] < line_dimensions[idx, 1]:  # vertical
                vertical.append(lines[idx])
            else:
                horizontal.append(lines[idx])

        # reshape
        horizontal_lines = np.reshape(horizontal, (-1, 4))
        vertical_lines = np.reshape(vertical, (-1, 4))

        # merge horizontal lines
        nHorizontal = horizontal_lines.shape[0]
        if nHorizontal > 2:
            didMerge = True
            while didMerge:
                didMerge, horizontal_lines = merge_lines(horizontal_lines)  # merge adjacent & parallel lines

        # merge vertical lines
        nVertical = vertical_lines.shape[0]
        if nVertical > 2:
            didMerge = True
            while didMerge:
                didMerge, vertical_lines = merge_lines(vertical_lines)  # merge adjacent & parallel lines

        # calc lines length - horizontal
        nHorizontal = horizontal_lines.shape[0]
        hori_dist = np.zeros((nHorizontal, ))
        for ind in range(nHorizontal):
            line = horizontal_lines[ind, :]
            hori_dist[ind] = abs(line[2] - line[0]) + abs(line[3] - line[1])

        # calc lines length - horizontal
        nVertical = vertical_lines.shape[0]
        vert_dist = np.zeros((nVertical, ))
        for ind in range(nVertical):
            line = vertical_lines[ind, :]
            vert_dist[ind] = abs(line[2] - line[0]) + abs(line[3] - line[1])

        # keep longest lines using adjusting threshold
        horizontal_indices, vertical_indices = np.arange(nHorizontal), np.arange(nVertical)

        if len(horizontal_lines) < 2:
            raise ValueError("Could not detect enough horizontal lines for grid boundaries")
        elif len(horizontal_lines) > 2:
            # # keep 2 longest lines
            iSort = np.argsort(hori_dist)
            horizontal_lines = horizontal_lines[iSort[-2:], :]

        # use vertical lines' data
        if len(vertical_indices) > 1:
            # keep only lines adjacent to horizontal lines
            nVert = len(vertical_indices)
            nHorz = horizontal_lines.shape[0]  # =2
            vec2keep = np.zeros((nVert,))
            for k1 in range(nVert):
                vert_line = vertical_lines[k1, :]
                for k2 in range(nHorz):
                    if np.abs(vert_line[0] - horizontal_lines[k2, 0]) < 10:
                        if np.abs(vert_line[1] - horizontal_lines[k2, 1]) < 10:
                            vec2keep[k1] = 1
                    if np.abs(vert_line[0] - horizontal_lines[k2, 2]) < 10:
                        if np.abs(vert_line[1] - horizontal_lines[k2, 3]) < 10:
                            vec2keep[k1] = 1
            vertical_indices = vertical_indices[vec2keep==1]
            nVert = len(vertical_indices)

            # use vertical
            if nVert==2:
                vertical_lines = vertical_lines[vertical_indices, :]
                points = np.vstack((np.reshape(horizontal_lines, (4, 2)) , np.reshape(vertical_lines, (4, 2))))
                hull = cv2.convexHull(points)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)

                # If we have 4 points, it's likely our grid
                if len(approx) == 4:
                    corners = self._order_corners_clockwise(np.reshape(approx, (4, 2)))
                    self._save_debug_corners(corners)

                    return corners  # approx.reshape(4, 2)


        # Construct bounding corners from detected lines
        corners = self._order_corners_clockwise(np.reshape(horizontal_lines, (4, 2)))

        self._save_debug_corners(corners)

        return corners


    # Save an image for debugging purposes if debug mode is enabled
    def _save_debug_enlarged(self, corners, new_corners, center) -> None:
        if self.config.debug_mode:
            debug_image = self.original_image.copy()
            cv2.circle(debug_image, np.astype(center, int), 3, (0, 255, 0), 5)  # green
            for point in corners:
                cv2.circle(debug_image, np.astype(point, int), 3, (0, 0, 255), 3)  # red
            for point in new_corners:
                cv2.circle(debug_image, np.astype(point, int), 3, (255, 0, 0), 2)  # blue
            cv2.imwrite("debug_05_corners_enlarged.png", debug_image)


    # enlarge corners to cover all the grid (in case we missed some parts of it)
    def _enlarge_boundaries(self, corners):
        # calculate line angle (if the angle is too large, then enlarge the image more)
        x1, y1, x2, y2 = corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]
        theta_degrees = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        enlarge_ratio = np.min((np.max((self.config.enlarge_ratio, theta_degrees/2)), 15))

        # apply enlargement
        center = np.mean(corners, 0)
        dist_vec = corners - center
        enlarge_vec = dist_vec*enlarge_ratio/100
        new_corners = corners + enlarge_vec
        new_corners = np.astype(new_corners, int)

        # validate corners
        new_corners[new_corners<0] = 0
        new_corners[new_corners[:, 0]>self.config.input_width, 0] = self.config.input_width
        new_corners[new_corners[:, 1]>self.config.input_height, 1] = self.config.input_height

        # debug plots
        self._save_debug_enlarged(corners, new_corners, center)

        return new_corners


    # Save debug image showing detected corner points
    def _save_debug_corners(self, corners):
        if self.config.debug_mode:
            debug_image = self.grayscale_image.copy()
            for point in corners:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, (0, 0, 255), 3)
            cv2.imwrite("debug_04_corners.png", debug_image)

            debug_image = self.original_image.copy()
            for point in corners:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, (255, 0, 0), 3)
            cv2.imwrite("debug_04_corners_color.png", debug_image)

        if False:
            debug_image = self.grayscale_image.copy()
            corner_points = []
            for line in horizontal_lines:
                corner_points.append([line[0], line[1]])
                corner_points.append([line[2], line[3]])

            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            for point in corner_points:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, (0, 0, 255), 3)

            cv2.imwrite("debug_04_horizontal_lines.png", debug_image)


    # Order corner points in clockwise order starting from top-left.
    @staticmethod
    def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
        """ Input:  points: Array of 4 corner points.
            Output: Ordered points: [top-left, top-right, bottom-right, bottom-left].  """
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest sum
        point_sums = points.sum(axis=1)
        ordered[0] = points[np.argmin(point_sums)]
        ordered[2] = points[np.argmax(point_sums)]

        # Top-right has smallest difference, bottom-left has largest difference
        point_diffs = np.diff(points, axis=1).flatten()
        ordered[1] = points[np.argmin(point_diffs)]
        ordered[3] = points[np.argmax(point_diffs)]

        return ordered


    # Apply perspective transform to rectify the grid.
    def _apply_perspective_transform(self, corners: np.ndarray) -> None:
        """ Input: corners: Four corner points of the detected grid.  """
        source_points = self._order_corners_clockwise(corners)  # not necessary

        size = self.config.output_size
        destination_points = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

        self.warped_grayscale = cv2.warpPerspective(self.grayscale_image, transform_matrix, (size, size))
        self.warped_color = cv2.warpPerspective(self.original_image, transform_matrix, (size, size))

        self._save_debug_image("debug_06_warped_gray.png", self.warped_grayscale.copy())
        self._save_debug_image("debug_06_warped_color.png", self.warped_color.copy())


    # Apply histogram equalization to improve contrast
    def _equalize_histogram(self) -> np.ndarray:
        """ Output: Contrast-enhanced grayscale image. """
        grayscale = self.grayscale_image.copy()

        # Calculate cumulative histogram
        counts, bin_edges = np.histogram(grayscale.flatten(), bins=range(0, 270, 10))
        cumulative_sum = np.cumsum(counts)
        total_pixels = grayscale.size
        cumulative_ratio = cumulative_sum / total_pixels * 100
        cumulative_ratio = np.append(cumulative_ratio, 100)

        # Find intensity values at specified percentiles
        upper_intensity = np.interp(self.config.hist_upper_percentile, cumulative_ratio, bin_edges)
        lower_intensity = np.interp(self.config.hist_lower_percentile, cumulative_ratio, bin_edges)

        # Stretch histogram to full range
        if upper_intensity - lower_intensity > 0:
            grayscale = 255 * (grayscale - lower_intensity) / (upper_intensity - lower_intensity)

        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

        return grayscale


    # Execute the complete extraction pipeline
    def process(self) -> bool:
        """ Output:  True if extraction was successful.        """
        # Step 1: Enhance contrast
        enhanced_grayscale = self._equalize_histogram()

        # Step 2: Create binary mask
        binary_mask = self._create_binary_mask(enhanced_grayscale)

        # Step 3: Detect grid corners
        corners = self._detect_grid_corners(binary_mask)

        # Step 4: enlarge image (ensure puzzle is in the boundaries)
        corners = self._enlarge_boundaries(corners)

        # Step 5: Apply perspective correction
        self._apply_perspective_transform(corners)

        return True


# Extract a Sudoku puzzle from an image file.
def extract_puzzle(image_path="sudoku.png", debug=False):
    """ Input:
        image_path: Path to the input image containing the Sudoku puzzle.
        debug: If True, save intermediate processing images for debugging.
    Output:
        Perspective-corrected color image of the Sudoku grid,
        or None if extraction fails.    """
    try:
        config = ExtractorConfig(debug_mode=debug)
        extractor = SudokuExtractor(image_path, config)

        if extractor.process():
            cv2.imwrite("extracted_puzzle.png", extractor.warped_color)
            return extractor.warped_color
        return None

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return None
    except ValueError as e:
        print(f"Processing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    result = extract_puzzle("screenshots/sudoku_full_screenshot_1.png", debug=True)

    if result is not None:
        print("Puzzle extracted successfully")
        #cv2.imwrite("extracted_puzzle.png", result)
    else:
        print("Failed to extract puzzle")
