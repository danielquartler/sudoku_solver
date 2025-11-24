# # sudoku_extractor_from_image
import cv2
import numpy as np
from typing import List, Tuple, Optional
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import sudoko_solver

# configurations
doDebugPlots = False
doDebug_digits_classifier = False


# Merge similar lines that are close to each other
def merge_similar_lines(lines, axis='vertical', threshold=20):
    if axis == 'vertical':
        coord_idx = 0  # x coordinate
    else:
        coord_idx = 1  # y coordinate

    coords = sorted([np.mean([line[coord_idx], line[2+coord_idx]]) for line in lines])
    merged = []
    group = [coords[0]]

    for c in coords[1:]:
        if abs(c - group[-1]) < threshold:
            group.append(c)
        else:
            merged.append(int(np.mean(group)))
            group = [c]
    merged.append(int(np.mean(group)))
    return merged


# Return line coefficients (a, b, c) for ax + by + c = 0
def line_equation(line):
    x1, y1, x2, y2 = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, -c


# keep lines with similar (a,b) values
def keep_similar_a_b(lines, lines_abc):
    a_vec = [line[0] for line in lines_abc]
    b_vec = [line[1] for line in lines_abc]

    med_a = np.median(a_vec)
    med_b = np.median(b_vec)
    dst_a = np.abs(a_vec - med_a)
    dst_b = np.abs(b_vec - med_b)
    iValid = np.where(np.logical_and(dst_a<50, dst_b<50))[0]

    if len(iValid)>0:
        lines_out = [lines[k] for k in iValid]
        lines_abc_out = [lines_abc[k] for k in iValid]
    else:
        stopHere = 1
        lines_out = lines
        lines_abc_out = lines_abc
    return lines_out, lines_abc_out


# remove lines with similar c values (ax+by+c=0)
def remove_similar_c(lines, lines_abc):
    c_vec = [line[2] for line in lines_abc]

    c_vec = np.asarray(c_vec)
    i_c_srt = np.argsort(c_vec)

    c_vec_srt = c_vec[i_c_srt]
    lines = [lines[k] for k in i_c_srt]
    lines_abc = [lines_abc[k] for k in i_c_srt]

    c_dff = np.diff(c_vec_srt)
    n_c = len(c_vec_srt)

    do_proceed = True
    while do_proceed:
        if n_c<3:
            do_proceed = False
        else:
            med_dff = np.median(c_dff)
            iValid = np.where(c_dff/med_dff > 0.4)[0]
            iValid = np.hstack((0, 1+iValid))

            nValid = len(iValid)
            if nValid == n_c:
                do_proceed = False
            else:
                #c_vec_srt = c_vec_srt[iValid]
                lines = [lines[k] for k in iValid]
                lines_abc = [lines_abc[k] for k in iValid]
                c_vec_srt = [line[2] for line in lines_abc]
                c_dff = np.diff(c_vec_srt)
                n_c = len(c_vec_srt)

    return lines, lines_abc


# Return intersection point of two lines
def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    if D == 0:
        return None
    x = (L1[2] * L2[1] - L1[1] * L2[2]) / D
    y = (L1[0] * L2[2] - L1[2] * L2[0]) / D
    return int(x), int(y)


# keep only lines with porper length (remove outliers: too short/long lines)
def keep_lines_proper_length(vertical):
    nVertical = len(vertical)
    dist_vertical = np.zeros((nVertical,))
    for k in range(nVertical):
        line = vertical[k]
        dist_vertical[k] = (line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2
    med_dist = np.median(dist_vertical)
    dist_from_med = np.abs(dist_vertical - med_dist) / med_dist

    vertical_new = [line for k, line in enumerate(vertical) if dist_from_med[k] < 0.2]
    return vertical_new, np.sqrt(med_dist)


# sort lines
def sort_lines(lines, isVertical):
    if isVertical:
        x_vals = [line[0] for line in lines]
    else:
        x_vals = [line[1] for line in lines]
    i_sorted = np.argsort(x_vals)
    line_sorted = [lines[k] for k in i_sorted]
    return line_sorted


# Calculate perpendicular distance from point to line
def distance_point_to_line(x: float, y: float, a: float, b: float, c: float) -> float:
    return abs(a * x + b * y + c)


# Find intersection point of two lines
def line_intersection(abc_line1, abc_line2):
    a1, b1, c1 = abc_line1
    a2, b2, c2 = abc_line2

    # calc dist
    dist = a1 * b2 - a2 * b1
    if abs(dist) < 1e-6:  # Lines are parallel
        return None

    x = (b1 * c2 - b2 * c1) / dist
    y = (a2 * c1 - a1 * c2) / dist
    return [x, y]


# Estimate the spacing between grid lines (should be uniform for Sudoku)
def estimate_grid_spacing(lines: List[Tuple[float, float, float]], is_vertical: bool = True) -> float:
    if len(lines) < 2:
        return 0

    # Get positions
    if is_vertical:
        positions = [-c / a if abs(a) > 0.5 else 0 for a, b, c in lines]
    else:
        positions = [-c / b if abs(b) > 0.5 else 0 for a, b, c in lines]

    positions = sorted(positions)

    # Calculate differences
    diffs = np.diff(positions)

    # Use median to be robust to outliers
    return np.median(diffs) if len(diffs) > 0 else 0


# Find the outer corners of the Sudoku grid
def find_sudoku_corners(lines, img):
    img_shape = img.shape
    if lines is None:
        raise ValueError("No lines detected.")

    lines = [l[0] for l in lines]

    # Separate vertical & horizontal lines
    vertical, horizontal = [], []
    for x1, y1, x2, y2 in lines:
        if abs(x2 - x1) < abs(y2 - y1):  # vertical
            vertical.append([x1, y1, x2, y2])
        else:
            horizontal.append([x1, y1, x2, y2])

    if not vertical or not horizontal:
        raise ValueError("Not enough vertical or horizontal lines found.")

    # DQ:
    if doDebugPlots:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in vertical:
            cv2.line(img_color, line[:2], line[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_vertical_2.png", img_color)

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in horizontal:
            cv2.line(img_color, line[:2], line[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_horizontal_2.png", img_color)

    # # keep only lines with porper length (return sorted lines)
    vertical_new, vertical_distance = keep_lines_proper_length(vertical)
    horizontal_new, horizontal_distance = keep_lines_proper_length(horizontal)

    # order lines
    vertical_new = sort_lines(vertical_new, True)
    horizontal_new = sort_lines(horizontal_new, False)

    # convert to abc
    v_lines_abc = [ line_equation(line) for line in vertical_new ]
    h_lines_abc = [ line_equation(line) for line in horizontal_new ]

    # keep similar a,b
    vertical_new, v_lines_abc = keep_similar_a_b(vertical_new, v_lines_abc)
    horizontal_new, h_lines_abc = keep_similar_a_b(horizontal_new, h_lines_abc)

    vertical_new, v_lines_abc = remove_similar_c(vertical_new, v_lines_abc)  # , vertical_distance
    horizontal_new, h_lines_abc = remove_similar_c(horizontal_new, h_lines_abc)

    # Estimate grid spacing
    v_spacing = estimate_grid_spacing(v_lines_abc, is_vertical=True)  # should be vertical_distance/9
    h_spacing = estimate_grid_spacing(h_lines_abc, is_vertical=False) # should be horizontal_distance/9

    if doDebugPlots:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in vertical_new:
            cv2.line(img_color, line[:2], line[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_vertical_3.png", img_color)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for line in horizontal_new:
            cv2.line(img_color, line[:2], line[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_horizontal_3.png", img_color)

    # find boundaries
    # horizontal
    option_bottom = np.asarray(horizontal_new[0])
    if img_shape[0]-option_bottom[2] < 40 and img_shape[1]-option_bottom[3] < 40:
        print(f"Succeeded retrieving bottom line {option_bottom}")
    else:
        option_bottom = []

    option_top = np.asarray(horizontal_new[-1])
    if img_shape[0]-option_top[2] < 40 and option_top[0] < 40 and option_top[1] < 40:
        print(f"Succeeded retrieving top line {option_top}")
    else:
        option_top = []

    if len(option_bottom)==0:
        if len(option_top) == 0:
            stopHere = 1  # I have a problem
            return None
        else:
            option_bottom = option_top.copy()
            option_bottom[1] += int(9 * h_spacing)
            option_bottom[3] += int(9 * h_spacing)
            option_bottom[1] = min(option_bottom[1], img_shape[0])
            option_bottom[3] = min(option_bottom[3], img_shape[1])
    else:
        if len(option_top) == 0:
            option_top = option_top.copy()
            option_top[1] -= int(9 * h_spacing)
            option_top[3] -= int(9 * h_spacing)
            option_top[1] = max(0, option_top[1])
            option_top[3] = max(0, option_top[3])
        #else:
            # all is perfect

    if doDebugPlots:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img_color, option_bottom[:2], option_bottom[2:], (255, 0, 0), 3)
        cv2.line(img_color, option_top[:2], option_top[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_horizontal_boundaries.png", img_color)


    # # # # # vertical
    # v_spacing vertical_new
    option_right = np.asarray(vertical_new[0])
    if img_shape[0]-option_right[0] < 40 and img_shape[1]-option_right[1] < 40:
        print(f"Succeeded retrieving right line {option_right}")
    else:
        option_right = []

    option_left = np.asarray(vertical_new[-1])
    if img_shape[0]-option_left[1] < 40 and option_left[0] < 40 and option_left[2] < 40:
        print(f"Succeeded retrieving left line {option_left}")
    else:
        option_left = []

    if len(option_right)==0:
        if len(option_left) == 0:
            stopHere = 1  # I have a problem
        else:
            option_right = np.asarray(vertical_new[0])
            # which line's index is it?
            dLR = option_right - option_left
            nML = 9 - np.round(dLR[2]/v_spacing)
            # update line
            option_right[0] += int(nML * v_spacing)
            option_right[2] += int(nML * v_spacing)
            option_right[0] = min(option_right[0], img_shape[0])
            option_right[2] = min(option_right[2], img_shape[1])
    else:
        if len(option_left) == 0:
            option_left = option_right.copy()
            option_left[0] -= int(9 * v_spacing)
            option_left[2] -= int(9 * v_spacing)
            option_left[0] = max(0, option_left[1])
            option_left[2] = max(0, option_left[3])
        #else:
            # all is perfect

    if doDebugPlots:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img_color, option_right[:2], option_right[2:], (255, 0, 0), 3)
        cv2.line(img_color, option_left[:2], option_left[2:], (255, 0, 0), 3)
        cv2.imwrite("thresh_based_verical_boundaries.png", img_color)

    # 4 point line presentation to [a,b,c] presentation
    abc_top = line_equation(option_top)
    abc_bottom = line_equation(option_bottom)
    abc_left = line_equation(option_left)
    abc_right = line_equation(option_right)

    # line intersections:
    intersections = np.zeros((4, 2))
    intersections[0, :] = line_intersection( abc_top, abc_left )
    intersections[1, :] = line_intersection( abc_top, abc_right )
    intersections[2, :] = line_intersection( abc_bottom, abc_right )
    intersections[3, :] = line_intersection( abc_bottom, abc_left )
    intersections = np.round(-intersections)
    intersections = np.astype(intersections, "int32")

    if doDebugPlots:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img_color, option_right[:2], option_right[2:], (255, 0, 0), 3)
        cv2.line(img_color, option_left[:2], option_left[2:], (255, 0, 0), 3)
        cv2.line(img_color, option_bottom[:2], option_bottom[2:], (255, 0, 0), 3)
        cv2.line(img_color, option_top[:2], option_top[2:], (255, 0, 0), 3)
        for corner in intersections:
            cv2.circle(img_color, corner, 3, (0, 0, 255), 3)
        cv2.imwrite("thresh_boundaries0.png", img_color)



    if False:
        #  Merge similar lines (remove duplicates)
        merged_x = merge_similar_lines(vertical, 'vertical', threshold=25)
        merged_y = merge_similar_lines(horizontal, 'horizontal', threshold=25)

        # Build line equations
        vlines = [line_equation([x, 0, x, img_shape[0]]) for x in merged_x]
        hlines = [line_equation([0, y, img_shape[1], y]) for y in merged_y]

        # Compute intersections
        intersections = []
        for vl in vlines:
            for hl in hlines:
                p = intersection(vl, hl)
                if p is not None:
                    x, y = p
                    if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                        intersections.append(p)

        intersections = np.array(intersections)
        if len(intersections) < 4:
            raise ValueError("Not enough intersection points found.")

    if False:
        # Find convex hull & approximate to 4 corners
        hull = cv2.convexHull(intersections)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

    return intersections.reshape(-1, 2)


# extract sudoku board
class SudokuExtractor:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Could not load image from {img_path}")
        self.img_perspective = []  # for debug
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.grid = None
        self.cells = []
        self.boundingRect = []
        self.cells_fullness_list = []
        self.has_digit_list = []
        model_path = "digit_classifier.keras"
        self.model = load_model(model_path)


    # Apply preprocessing to enhance grid detection
    def preprocess(self) -> np.ndarray:
        self.gray = cv2.resize(self.gray, (450, 450), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)

        if True:
            # adaptive thresholding computes a different threshold for each small region in the image.
            # This is very useful when:  lighting is uneven (some parts bright, others dark).
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 3)
        else:
            # global fixed threshold value
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological Operations (Erosion/Dilation): post-processing operations applied on the binary (thresh) image to clean up small noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        if doDebugPlots:
            cv2.imwrite("opened.png", opened)
        return opened


    # Find the largest rectangular contour (the Sudoku grid)
    def find_grid(self, thresh: np.ndarray) -> Optional[np.ndarray]:
        # input: thresh: a binary image (output of thresholding)
        if doDebugPlots:
            cv2.imwrite("thresh.png", thresh)

        if False:  # option 1
            img_shape = thresh.shape

            # Line Segment Detector (LSD)
            lsd = cv2.createLineSegmentDetector()
            lines, widths, precisions, nfas = lsd.detect(thresh)  # self.gray

            nLines = lines.shape[0]
            line_len = np.zeros((nLines, 2))
            for kL in range(nLines):
                curLine = lines[kL]
                line_len[kL, 0] = np.abs(curLine[0, 2] - curLine[0, 0])
                line_len[kL, 1] = np.abs(curLine[0, 3] - curLine[0, 1])
            i_hor = np.where(line_len[:, 0] > img_shape[1] / 2)[0]
            i_ver = np.where(line_len[:, 1] > img_shape[0] / 2)[0]

            line_horizontal = lines[i_hor]
            line_vertical = lines[i_ver]

            # get bounding rectangle
            approx = np.zeros((4, 1, 2), dtype="int32")
            approx[0, 0, 0] = int(min(line_horizontal[:, 0, 0]))
            approx[0, 0, 1] = int(min(line_horizontal[:, 0, 1]))
            approx[2, 0, 0] = int(max(line_horizontal[:, 0, 0]))
            approx[1, 0, 1] = int(max(line_horizontal[:, 0, 1]))

            approx[1, 0, 0] = approx[0, 0, 0]
            approx[3, 0, 1] = approx[0, 0, 1]
            approx[3, 0, 0] = approx[2, 0, 0]
            approx[2, 0, 1] = approx[1, 0, 1]
            # approx = [[[10  3]],, [[10 385]],, [[389 385]],, [[389   3]]]
            # approx = [[[10  4]],, [[12 386]],, [[392 384]],, [[392   4]]].  shape=[4,1,2]

            return approx.reshape(4, 2)

        else:  # option 2
            # findContours:remember, object to be found should be white and background should be black.
            # RETR_EXTERNAL= Retrieve only the outermost contours (ignores nested/inside contours).
            # cv2.CHAIN_APPROX_SIMPLE: tells OpenCV how to store the contour points.  Stores only endpoints.
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Find the largest contour
            largest = max(contours, key=cv2.contourArea)

            # Approximate to a polygon
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            # If we have 4 points, it's likely our grid
            if len(approx) == 4:
                return approx.reshape(4, 2)

            # Otherwise, get bounding rectangle
            #x, y, w, h = cv2.boundingRect(largest)

            #thresh is 640x480 and not 400x400  I have to fix that

            # canny
            edges = cv2.Canny(thresh, 100, 200)  # img, TH1, TH2
            if doDebugPlots:
                cv2.imwrite("thresh_Canny.png", edges)
            # Hough: (image, distance resolution (typical 1), Angle resolution in radians, threshold, minLineLength [pixels], The maximum allowed gap between two points on the same line for them to be connected into a single line segment)
            lines4 = None
            min_line_len = 400
            while lines4 is None:
                #print(min_line_len)
                lines4 = cv2.HoughLinesP(edges, 2, np.pi / 300, 200, minLineLength=min_line_len, maxLineGap=int(min_line_len/10))
                min_line_len = int(min_line_len*0.7)
            pts4 = np.concatenate([l.reshape(2, 2) for l in lines4])
            nPts4 = len(pts4)
            if doDebugPlots:
                img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.polylines(img_color, np.reshape(pts4, (nPts4, 1, 2)), True, (255, 0, 0), 25)
                cv2.imwrite("thresh_based_HoughLinesP4.png", img_color)

            corners4 = find_sudoku_corners(lines4, edges)
            if corners4 is None:
                # Find convex hull & approximate to 4 corners
                hull = cv2.convexHull(np.reshape(lines4, (-1, 2)))
                hull = cv2.convexHull(lines4)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)

                debug_image = self.img.copy()
                corners4 = np.reshape(hull, (-1, 2))
                for point in corners4:
                    cv2.circle(debug_image, np.astype(point, int), 3, (0, 0, 255), 2)  # blue
                cv2.imwrite("thresh_with_hull.png", debug_image)


                debug_image = self.img.copy()
                corners4 = np.reshape(approx, (-1, 2))
                for point in corners4:
                    cv2.circle(debug_image, np.astype(point, int), 3, (255, 0, 0), 2)  # blue
                cv2.imwrite("thresh_with_bad_approx.png", debug_image)


                xxx= 1

        return corners4


    # Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


    # Apply perspective transform to get a top-down view of the grid
    def warp_grid(self, corners: np.ndarray, size: int = 450) -> np.ndarray:
        rect = self.order_points(corners)
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        # print(f"getPerspectiveTransform = {M}")
        self.grid = cv2.warpPerspective(self.gray, M, (size, size))
        if doDebugPlots:
            cv2.imwrite("warpPerspective.png", self.grid)
            img2 = cv2.resize(self.img, (450, 450), interpolation=cv2.INTER_AREA)
            self.img_perspective = cv2.warpPerspective(img2, M, (size, size))
            cv2.imwrite("warpPerspective_color.png", self.img_perspective)
        # self.grid.shape=(450,450).    self.img.shape=(299,345,3)


    # clean boundaries (color boundaries)
    def clean_boundaries(self, new_color= 255):
        if doDebugPlots:
            cv2.imwrite("warpPerspective_before_boundaries_fix.png", self.grid)
        gray_TH = 170
        #new_color = 255 # 0=black
        for k in range(0, 6):
            # Left border:
            gray_scale_p50 = np.percentile(self.grid[:, k], 50)
            if gray_scale_p50 < gray_TH:
                self.grid[:, k] = new_color

            # Right border:
            gray_scale_p50 = np.percentile(self.grid[:, -k], 50)
            if gray_scale_p50 < gray_TH:
                self.grid[:, -k] = new_color

            # Up border:
            gray_scale_p50 = np.percentile(self.grid[k, :], 50)
            if gray_scale_p50 < gray_TH:
                self.grid[k, :] = new_color

            # Bottom border:
            gray_scale_p50 = np.percentile(self.grid[-k, :], 50)
            if gray_scale_p50 < gray_TH:
                self.grid[-k, :] = new_color
        if doDebugPlots:
            cv2.imwrite("warpPerspective_after_boundaries_fix.png", self.grid)


    # calculate gray histogram
    def calc_gray_hist(self):
        [counts, gray_bins] = np.histogram(self.grid, bins=range(0, 270, 10))
        cum_sum = np.cumsum(counts)
        rat_cum = cum_sum / 2025  # 202,500 = 450*450
        rat_cum = np.append(rat_cum, 100)
        return rat_cum, gray_bins


    # calc grayness values in 3 different percentiles
    def calc_gray_prctls(self):
        rat_cum, gray_bins = self.calc_gray_hist()

        # y = np.interp(x, xp, fp)
        gray_p50 = np.interp(50, rat_cum, gray_bins)  # I wish: gray_p50 => 235
        gray_p25 = np.interp(25, rat_cum, gray_bins)  # I wish: gray_p25 => 100
        gray_p02 = np.interp(3.5, rat_cum, gray_bins)  # I wish: gray_p02 => 0

        # y = a x sqrt(X)+bX+c  <=>   A·[a, b, c] = y where A:
        A = np.array([
            [np.sqrt(gray_p02), gray_p02, 1],
            [np.sqrt(gray_p25), gray_p25, 1],
            [np.sqrt(gray_p50), gray_p50, 1]
        ])

        return gray_p02, gray_p25, gray_p50, A


    # adjust gray colors
    def histogram_equalization(self):
        thresh = cv2.adaptiveThreshold(self.grid, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 8)
        if doDebugPlots:
            cv2.imwrite(r"warpPerspective_myHist_thresh_mean_19_8.png", thresh)

        self.grid = thresh


    # Extract individual cells from the warped grid.  self.grid => self.cells
    def extract_cells(self) -> List[np.ndarray]:
        # parameter
        prctl_val = 40
        gray_TH = 220

        # init
        warped = self.grid
        cells = []
        h, w = warped.shape[:2]
        cell_h, cell_w = h // 9, w // 9

        # for each cell:
        for i in range(9):
            for j in range(9):
                # get current cell
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = warped[y1:y2, x1:x2]

                # remove cell's external lines
                iTop_border = np.where(np.percentile(cell[:9, :], prctl_val, axis=1) < gray_TH)[0]
                if len(iTop_border) == 0:
                    y0 = 0
                else:
                    y0 = np.max(iTop_border)

                iBottom_border = np.where(np.percentile(cell[-8:, :], prctl_val, axis=1) < 200)[0]
                if len(iBottom_border) == 0:
                    yF = cell_h - 8
                else:
                    yF = cell_h - 8 + np.min(iBottom_border)

                iLeft_border = np.where( np.percentile(cell[:, :9], prctl_val, axis=0) < gray_TH )[0]
                if len(iLeft_border)==0:
                    x0 = 0
                else:
                    x0 = np.max(iLeft_border) + 1

                iRight_border = np.where(np.percentile(cell[:, -9:], prctl_val, axis=0) < gray_TH)[0]
                if len(iRight_border) == 0:
                    xF = cell_w - 9
                else:
                    xF = cell_w - 9 +np.min(iRight_border)

                if doDebugPlots:
                    cv2.imwrite("cell_" + str(i) +"_"+ str(j) +".png", cell[y0:yF, x0:xF])
                    print(f"cell ({i},{j}) borders are [{y0}:{yF},{x0}:{xF}]")
                # remove borders
                cell = cell[y0:yF, x0:xF]

                # add cell
                cells.append(cell)

        self.cells = cells


    # Preprocess a cell to extract the digit's pixels out of the cell
    def preprocess_cell(self, cell: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell

        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove borders (grid lines)
        h, w = thresh.shape
        border = int(h * 0.1)
        thresh[:border, :] = 0
        thresh[-border:, :] = 0
        thresh[:, :border] = 0
        thresh[:, -border:] = 0

        return thresh


    # Check if a cell contains a digit based on white pixel ratio
    def cells_fullness(self, threshold: float = 0.025):
        for idx, cell in enumerate(self.cells):
            ratio = np.mean(cell[1:-1, 1:-1] > 220)
            self.cells_fullness_list.append(ratio)
            #print(f"{idx}) ratio= {ratio:.4f}")


    # Check if a cell contains a digit based on white pixel ratio
    def has_digit(self, idx, threshold: float = 0.97) -> bool:
        ratio = self.cells_fullness_list[idx]
        return 0.5 < ratio and ratio < threshold


    # Extract and isolate the digits from all cells
    def extract_digit_images(self, fullness_TH):
        digit_images = []
        h_list = []  # all digits should have similar height
        larget_list = []

        # for each cell
        for idx, cell in enumerate(self.cells):
            digit_images.append(None)
            if self.has_digit(idx, fullness_TH):
                # make cell ready for "contour"
                if False:  # option 1
                    _, cell = cv2.threshold(cell, 50, 255, cv2.THRESH_BINARY)
                else:
                    cell = 255 - cell
                    cell[cell<50] = 0
                    _, cell = cv2.threshold(cell, 50, 255, cv2.THRESH_BINARY)
                # Find the largest connected component (the digit)
                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get the largest contour
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)

                    self.has_digit_list.append(idx)
                    larget_list.append([x, y, w, h])
                    h_list.append(h)

        # extract the digits out of the cells with digits (fine tuning)
        h_med = np.median(h_list)
        # i_bad = np.where(abs(h_med-h_list)>3)[0]

        # for each cell with digits
        for innder_idx, idx in enumerate(self.has_digit_list):
            cell = 255 - self.cells[idx]
            x, y, w, h = larget_list[innder_idx]

            # [counts, gray_bins] = np.histogram(cell, bins=range(0, 270, 10))  # for debug

            if np.abs(h_med-h)>3:  # we have a problem, we didn't calculate it right before
                if h < h_med :
                    cell = cell[1:-1, 1:-1]
                    _, thresh = cv2.threshold(cell, 26, 255, cv2.THRESH_BINARY)  # we should use smaller TH
                else:
                    _, thresh = cv2.threshold(cell, 120, 255, cv2.THRESH_BINARY)  # we should use larger TH
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get the largest contour
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)

                    # validate again
                    if h < h_med:
                        cell = cell[1:-1, 1:-1]
                        _, thresh = cv2.threshold(cell, 16, 255, cv2.THRESH_BINARY)  # we should use smaller TH
                    else:
                        _, thresh = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY)  # we should use larger TH
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Get the largest contour
                        largest = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest)

            # Extract digit from cell
            digit = cell[y:y + h, x:x + w]

            # Resize to a standard size (28x28 for typical digit recognition)
            #digit = 255 - digit
            # option 1
            digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            # option 2 (keep original size)
            #x0 = (28 - h) // 2
            #y0 = (28 - w) // 2
            #digit_resized = np.zeros((28, 28))
            #digit_resized[x0:x0 + h, y0:y0 + w] = digit

            if doDebugPlots:
                print(f"idx={idx}) row={idx // 9} , col={idx % 9}")
                cv2.imwrite("digit_resized_" + str(idx) + ".png", digit_resized)

                _, thresh = cv2.threshold(cell, 50, 255, cv2.THRESH_BINARY)
                cv2.imwrite("digit_thresh_" + str(idx) + ".png", thresh)

                # consider drawing original cell and the digit area:
                img_color = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the red rectangle
                cv2.imwrite("largest_contour_box_" + str(idx) + ".png", img_color)  # Save the result

            # add current digit
            digit_images[idx] = digit_resized

        return digit_images


    # extract the digit from the images' cells
    def image_to_digit(self, digit_images):
        if doDebug_digits_classifier:
            print("image_to_digit():")
        digits = np.zeros((9, 9), dtype=int)
        for idx, digit_image in enumerate(digit_images):
            if digit_image is not None:
                row, col = idx // 9, idx % 9
                digit_image = digit_image.astype("float32") / 255.0
                predictions = self.model.predict(digit_image.reshape(1, 28, 28, 1), verbose=0)
                digits[row, col] = np.argmax(predictions)
                if doDebug_digits_classifier:
                    print([idx, [np.round(v, 2) for v in predictions], digits[row, col]])  # may consider to do something if we have an ambiguity

                prediction_score = np.max(predictions)
                if prediction_score < 0.84:
                    digits[row, col] = 0


        if doDebug_digits_classifier:
            print("Finished digits extraction")
            print(digits.astype(int))
            print("\n1 line representation:")
            sBoard_1line = ""
            for d in np.reshape(digits, (81, 1)):
                sBoard_1line += str(d[0]) +", "
            print(sBoard_1line[:-2])
        return digits


    # Main processing pipeline
    def process(self) -> List[Optional[np.ndarray]]:
        # Preprocess
        thresh = self.preprocess()

        # Find grid
        corners = self.find_grid(thresh)
        if corners is None:
            raise ValueError("Could not find Sudoku grid in image")
        self.boundingRect = corners

        # Warp to top-down view (new image in the size of 450x450). Generates self.grid
        self.warp_grid(corners)

        # adjust colors
        self.histogram_equalization()

        # clean boundaries
        self.clean_boundaries()

        # Extract 81 cells (self.cells)
        self.extract_cells()

        # determine fullness TH
        self.cells_fullness()  # generates self.cells_fullness_list  # when=1 means empty cell. val(digit)~=0.7
        fullness_TH = 0.97  # 0.97 * np.percentile(self.cells_fullness_list, 40)  # should be ~0.97

        # Extract digits from cells
        digit_images = self.extract_digit_images(fullness_TH)
        digits = self.image_to_digit(digit_images)
        return digits


    # Visualization: print the solved digits on top of the given image.
    def generate_solved_board_image(self, board0, board_solution):
        original_size = self.img.shape
        image_size = self.gray.shape
        #true_board_size = [ self.boundingRect[2][0] - self.boundingRect[0][0], self.boundingRect[2][1] - self.boundingRect[0][1]]
        true_board_size = [np.max(self.boundingRect[:, 0]) - np.min(self.boundingRect[:, 0]), np.max(self.boundingRect[:, 1]) - np.min(self.boundingRect[:, 1])]

        # adjust attributes to original size
        ratio_0, ratio_1 = original_size[0] / image_size[0] , original_size[1] / image_size[1]
        true_board_size[0] = int(true_board_size[0] * ratio_0)
        true_board_size[1] = int(true_board_size[1] * ratio_1)

        boundingRect = self.boundingRect.copy()
        boundingRect[:, 0] = boundingRect[:, 0] * ratio_0
        boundingRect[:, 1] = boundingRect[:, 1] * ratio_1

        # create a new image
        img = Image.open(self.img_path)
        draw = ImageDraw.Draw(img)

        # Define font and size
        font_size = int( (true_board_size[0])/12)
        font = ImageFont.truetype("arial.ttf", font_size)

        # write solutions
        net_true_size = true_board_size.copy()
        net_true_size[0] -= boundingRect[0][1]
        net_true_size[1] -= boundingRect[0][0]
        dX, dY = net_true_size[0]/9, net_true_size[1]/9

        margin_L = dY/5 + boundingRect[0][0] + dY/4
        margin_U = dX/6 + boundingRect[0][1]

        # for each cell:
        for k0 in range(3):
            for k1 in range(3):
                for k2 in range(3):
                    for k3 in range(3):
                        if board0[k0][k1][k2][k3]==0:  # was empty = digits[k0 * 3 + k2][k1 * 3 + k3]
                            position = np.round([dY*(k1*3+k3)+margin_L, dX*(k0*3+k2)+margin_U]).astype(int)  # (x, y) coordinates from the top-left corner
                            draw.text(position, str(board_solution[k0][k1][k2][k3][0]), fill="black", font=font)

        # Show or save the result
        img.show()
        img.save("output.png")


# fill the board object with the digits from "digits" list.
def digits_to_board(digits):
    Dim = 3
    #board0 = np.reshape(digits, (3,3,3,3))
    board0 = [[[[0 for _ in range(Dim)] for _ in range(Dim)] for _ in range(Dim)] for _ in range(Dim)]
    for k0 in range(Dim):
        for k1 in range(Dim):
            for k2 in range(Dim):
                for k3 in range(Dim):
                    board0[k0][k1][k2][k3] = digits[k0 * 3 + k2][k1 * 3 + k3]
    return board0


# main
def solve_sudoku_from_image(img_path="sudoku.png", doSolvePuzzle=True, doVisualization=True):
    try:
        # # Reader
        extractor = SudokuExtractor(img_path)
        digits = extractor.process()

        # # Solver
        if doSolvePuzzle:
            board0 = digits_to_board(digits)  # 9x9 => 3x3x3x3
            board = sudoko_solver.init_board_options(board0, 3)
            board_solution = sudoko_solver.solve_board(board, 3, 0)
        else:
            board_solution = digits  # returns initial digits extracted from the image.

        # # Visualization
        if doVisualization:
            sudoko_solver.print_board(board_solution, 3)
            extractor.generate_solved_board_image(board0, board_solution)

    except Exception as e:
        print(f"Error: {e}")
        board_solution = []
    return board_solution


if __name__ == "__main__":
    #solve_sudoku_from_image("unit_test_data\sudoku7.png")
    #solve_sudoku_from_image("screenshots//sudoku_full_screenshot_3.png")
    #solve_sudoku_from_image("sudoku.png")
    #solve_sudoku_from_image("extracted_puzzle.png")
    solve_sudoku_from_image("unit_test_data\sudoku4.png", False, False)

    #solve_sudoku_from_image()
