import cv2
import numpy as np
from typing import List, Tuple, Optional
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sudoko_solver

doDebugPlots = False

class SudokuExtractor:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Could not load image from {img_path}")
        self.img_perspective = []
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
        self.gray = cv2.resize(self.gray, (400, 400), interpolation=cv2.INTER_AREA)
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

            # consider
            #[vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Find the largest contour
            largest = max(contours, key=cv2.contourArea)

            # Approximate to a polygon
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            # If we have 4 points, it's likely our grid
            if len(approx) == 4:
                return approx.reshape(4, 2)

            # Otherwise, get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest)

        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


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
        self.grid = cv2.warpPerspective(self.gray, M, (size, size))
        self.img_perspective = cv2.warpPerspective(self.img, M, (size, size))


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
        for idx, cell in enumerate(self.cells):
            if not self.has_digit(idx, fullness_TH):
                digit_images.append(None)
            else:
                # Find the largest connected component (the digit)
                cell = 255 - cell
                cell[cell<50] = 0
                contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    digit_images.append(None)

                else:
                    # Get the largest contour
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)

                    # Extract digit with padding
                    digit = cell[y:y + h, x:x + w]

                    # Resize to a standard size (28x28 for typical digit recognition)
                    # option 1
                    digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                    # option 2 (keep original size)
                    #x0 = (28 - h) // 2
                    #y0 = (28 - w) // 2
                    #digit_resized = np.zeros((28, 28))
                    #digit_resized[x0:x0 + h, y0:y0 + w] = digit

                    if doDebugPlots:
                        cv2.imwrite("digit_resized_" + str(idx) + ".png", digit_resized)

                    # add current digit
                    digit_images.append(digit_resized)
                    self.has_digit_list.append(idx)

        return digit_images


    # extract the digit from the images' cells
    def image_to_digit(self, digit_images):
        print("image_to_digit():")
        digits = np.zeros((9, 9), dtype=int)
        for idx, digit_image in enumerate(digit_images):
            if digit_image is not None:
                row, col = idx // 9, idx % 9
                digit_image = digit_image.astype("float32") / 255.0
                predictions = self.model.predict(digit_image.reshape(1, 28, 28, 1), verbose=0)
                print([np.round(v, 2) for v in predictions])  # may consider to do something if we have an ambiguity
                digits[row, col] = np.argmax(predictions)
        print("Finished digits extraction")
        print(digits.astype(int))
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
        true_board_size = [ self.boundingRect[2][0] - self.boundingRect[1][0] , self.boundingRect[1][1] - self.boundingRect[0][1]]

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
        font = ImageFont.truetype("arial.ttf", font_size)  # You can use any .ttf font

        # write solutions
        net_true_size = true_board_size.copy()
        net_true_size[0] -= boundingRect[0][1]
        net_true_size[1] -= boundingRect[0][0]
        dX, dY = net_true_size[0]/9, net_true_size[1]/9

        margin_L = dY/5 + boundingRect[0][0] + dY/4
        margin_U = dX/8 + boundingRect[0][1]

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
def main():
    # Example usage
    img_path = "sudoku.png"  # Replace with your image path

    try:
        # # Reader
        extractor = SudokuExtractor(img_path)
        digits = extractor.process()

        # # Solver
        board0 = digits_to_board(digits)  # 9x9 => 3x3x3x3
        board = sudoko_solver.init_board_options(board0, 3)
        board_solution = sudoko_solver.solve_board(board, 3, 0)

        # # Visualization
        sudoko_solver.print_board(board_solution, 3)
        extractor.generate_solved_board_image(board0, board_solution)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
