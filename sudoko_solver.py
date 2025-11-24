# solve sudoko
import numpy as np
import copy


def load_board( board_num ):
    solution = []
    if board_num==0:
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,4]], [[0,4],[0,1]]],  [ [[2,0],[4,0]] , [[4,0],[1,2]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2
    elif board_num==1:
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,0]], [[0,4],[0,1]]],  [ [[2,0],[4,0]] , [[4,0],[1,0]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2
    elif board_num==2:
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,0]], [[0,0],[0,1]]],  [ [[2,0],[4,0]] , [[4,0],[0,0]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2
    elif board_num==3:    # requires guessing
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,0]], [[0,0],[0,1]]],  [ [[0,0],[4,0]] , [[4,0],[0,0]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2
    elif board_num==4:    # contains a mistake
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,0]], [[0,0],[0,1]]],  [ [[0,0],[4,0]] , [[4,1],[0,0]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2
    elif board_num==5:    # requires many guesses
        #                  Up Left        Up Right          Down Left      Down Right
        board    = [ [  [[1,0],[0,0]], [[0,0],[0,1]]],  [ [[0,0],[0,0]] , [[4,0],[0,0]] ] ]  # UL  UR  DL  DR
        solution = [ [  [[1,2],[3,4]], [[3,4],[2,1]]],  [ [[2,1],[4,3]] , [[4,3],[1,2]] ] ]  # UL  UR  DL  DR
        Dim = 2

    elif board_num==6:    # pic 1.png   note:  no need for any guessing.
        #                    Up Left                     Up Middle                Up Right                       Middle Left                 Middle Middle             Middle Right                      Down Left                  Down Middle                Down Right
        board    = [  [  [[9,1,3],[6,0,7],[0,5,0]], [[0,0,0],[0,0,0],[0,8,0]],  [[5,0,0],[0,2,4],[0,7,0]]  ],[  [[0,7,9],[0,0,2],[0,0,0]], [[0,0,0],[0,9,0],[0,0,4]], [[0,0,0],[0,4,3],[0,9,0]]  ],[   [[0,4,0],[7,0,6],[0,0,1]], [[0,0,1],[0,0,9],[0,0,6]], [[9,0,0],[0,0,5],[4,0,7]]  ]    ]
        solution = [  [  [[9,1,3],[6,8,7],[2,5,4]], [[4,2,7],[9,1,5],[6,8,3]],  [[5,8,6],[3,2,4],[1,7,9]]  ],[  [[4,7,9],[1,6,2],[5,3,8]], [[1,3,2],[5,9,8],[7,6,4]], [[6,5,8],[7,4,3],[2,9,1]]  ],[   [[3,4,5],[7,2,6],[8,9,1]], [[8,7,1],[3,4,9],[2,5,6]], [[9,6,2],[8,1,5],[4,3,7]]  ]    ]
        Dim = 3

    elif board_num==7:    # pic 11.png   note:  no need for any guessing.
        #                    Up Left                     Up Middle                Up Right                       Middle Left                 Middle Middle             Middle Right                      Down Left                  Down Middle                Down Right
        board = [
            [[[0, 9, 0], [1, 0, 6], [0, 4, 0]], [[0, 0, 0], [0, 0, 0], [0, 5, 0]], [[0, 5, 0], [3, 0, 7], [0, 1, 0]]],
            [[[0, 0, 0], [0, 0, 7], [0, 0, 0]], [[0, 4, 0], [8, 0, 5], [0, 3, 0]], [[0, 0, 0], [6, 0, 0], [0, 0, 0]]],
            [[[0, 1, 0], [6, 0, 4], [0, 8, 0]], [[0, 8, 0], [0, 0, 0], [0, 0, 0]], [[0, 6, 0], [9, 0, 8], [0, 2, 0]]]]
        # board    = [  [  [[0,9,0],[1,0,6],[0,4,0]], [[0,0,0],[0,0,0],[0,5,0]],  [[0,5,0],[3,0,7],[0,1,0]]  ],[  [[0,0,0],[0,0,7],[0,0,0]], [[0,4,0],[8,0,5],[0,3,0]], [[0,0,0],[6,0,0],[0,0,0]]  ],[   [[0,1,0],[6,0,4],[0,8,0]], [[0,8,0],[0,0,0],[0,0,0]], [[0,6,0],[9,0,8],[0,2,0]]  ]    ]
        # added:  board[0][0][1][1] = 5   ;  board[0][2][2][2]= 9
        # board[2][0][2][0] == 5:
        Dim = 3

    elif board_num==8:    # pic 11.png   note:  added some more cells.
        #                    Up Left                     Up Middle                Up Right                       Middle Left                 Middle Middle             Middle Right                      Down Left                  Down Middle                Down Right
        board    = [  [  [[0,9,0],[1,5,6],[0,4,0]], [[0,0,0],[0,0,0],[0,5,0]],  [[0,5,0],[3,0,7],[0,1,9]]  ],[  [[0,0,0],[0,0,7],[0,0,0]], [[0,4,0],[8,0,5],[0,3,0]], [[0,0,0],[6,0,0],[0,0,0]]  ],[   [[0,1,0],[6,0,4],[0,8,0]], [[0,8,0],[0,0,0],[0,0,0]], [[0,6,0],[9,0,8],[0,2,0]]  ]    ]
        Dim = 3

    return solution, board, Dim

# each cell may store values 1 to Dim (all options).  The board true values enforce certain values.
def init_board_options(board, Dim):
    # init board (fill with all optional values)
    #board_options = [ [ [ [np.arange(1, Dim*Dim+1)]*Dim ]*Dim ]*Dim ]*Dim  # not good
    board_options = [[[[0 for _ in range(Dim)] for _ in range(Dim)] for _ in range(Dim)] for _ in range(Dim)]

    # insert board's initial values
    for k1 in range(Dim):
        for k2 in range(Dim):
            for k3 in range(Dim):
                for k4 in range(Dim):
                    if board[k1][k2][k3][k4] > 0:
                        board_options[k1][k2][k3][k4] = np.asarray([ board[k1][k2][k3][k4] ])
                    else :
                        board_options[k1][k2][k3][k4] = np.arange(1, Dim*Dim+1)

    print("initial board:")
    print_board(board_options, Dim)

    # update board
    board_new, n_values_add, was_mistake = update_board(board_options, Dim)
    return board_new

def solve_board(board, Dim, nGuesses):
    MAX_DEPTH = 5
    n_cells_not_done, list_not_done = calc_perms(board, Dim)
    if n_cells_not_done < 17:
        MAX_DEPTH += 1
    if n_cells_not_done < 10:
        MAX_DEPTH += 1
        if n_cells_not_done < 5:
            MAX_DEPTH += 1

    do_proceed = n_cells_not_done > 0
    nGuesses += 1
    if nGuesses>MAX_DEPTH :
        return board

    if do_proceed:
        arr_not_done = np.asarray(list_not_done)
        iOrd = np.argsort( arr_not_done[:,-1] )
        arr_not_done = arr_not_done[iOrd,:]
        # kVec1, kVec2, kVec3, kVec4, vec_not_done = zip(*list_not_done)
        board_prev= copy.deepcopy(board)
        n_cells_to_try = n_cells_not_done

        # Try all options
        for kC in range(n_cells_to_try):  # kC = cell_to_look
            was_mistake = False
            if n_cells_not_done > 0 and not was_mistake and nGuesses<MAX_DEPTH:
                l1,l2,l3,l4 = arr_not_done[kC,0], arr_not_done[kC,1], arr_not_done[kC,2], arr_not_done[kC,3]
                Queue_small = board_prev[l1][l2][l3][l4]
                for guess in Queue_small:
                    # set guess
                    board = copy.deepcopy(board_prev) # note: .copy() is not a soft copy. We need deep copy!
                    board[l1][l2][l3][l4] = [int(guess)]
                    if nGuesses==1:
                        print([nGuesses,kC,n_cells_not_done, int(l1),int(l2),int(l3),int(l4),int(guess)])

                    if len(board[0][2][0][0]) == 1 and board[0][2][0][0] == 2:
                        if len(board[2][1][1][2]) == 1 and board[2][1][1][2] == 1:
                            StopHere = 1

                    # try solving
                    board, n_values_add, was_mistake = update_board(board, Dim)
                    if was_mistake:
                        pass # we need to try a different guess
                    else :
                        n_cells_not_done, list_not_done = calc_perms(board, Dim)

                        if n_cells_not_done<14:
                            print([nGuesses, kC, n_cells_not_done, int(l1), int(l2), int(l3), int(l4), int(guess)])
                            print_board(board, Dim)
                        if n_cells_not_done==0:  # if finished
                            return board
                        elif nGuesses+1==MAX_DEPTH:
                            pass # we don't allow another guess
                        else: # requires another guess
                            board = solve_board(board, Dim, nGuesses)
                            board, n_values_add, was_mistake = update_board(board, Dim)
                            if was_mistake:
                                pass  # we need to try a different guess
                            else:
                                n_cells_not_done, list_not_done = calc_perms(board, Dim)
                                if n_cells_not_done == 0:  # if finished
                                    return board

        n_cells_not_done, list_not_done = calc_perms(board, Dim)
        do_proceed = n_cells_not_done > 0

    return board   # solve_board()


# Calculates the number of possible solutions.
def calc_perms(board, Dim):
    n_cells_not_done = 0
    list_not_done = []

    for k1 in range(Dim):  # rows
        for k2 in range(Dim):  # columns
            for k3 in range(Dim):  # inside a square, rows:
                for k4 in range(Dim):  # inside a square, cols:
                    cur_len = len(board[k1][k2][k3][k4])
                    if cur_len > 1:
                        n_cells_not_done += 1
                        list_not_done.append([k1, k2, k3, k4, cur_len])
    return n_cells_not_done, list_not_done


# update board (Keep updating the board until no more candidates can be eliminated through logic)
def update_board(board, Dim):
    n_values_add = 1  # init
    total_values_added = 0
    was_mistake = False
    while n_values_add > 0 and not was_mistake:
        board, n_values_add, was_mistake = update_board_1step(board, Dim)
        total_values_added += n_values_add
    return board, total_values_added, was_mistake


# update board (for each cell, removing all disqualified digits)
def update_board_1step(board, Dim):
    # init
    Vector = np.arange(1, Dim * Dim + 1)
    Bins = np.arange(1, Dim * Dim + 2)
    n_values_add = 0
    was_mistake = False

    # update board
    for k1 in range(Dim):   # rows
        for k2 in range(Dim):  # columns
            curSquare_elements = np.asarray( [item[0] for row in board[k1][k2] for item in row if len(item) == 1] )
            if len(curSquare_elements) != len(np.unique(curSquare_elements)):
                return board, n_values_add, True

            # another solving method: [[[2, 4, 8], [5], [2, 4, 6]], [[3], [4, 8], [7]], [[2, 8], [1], [2, 6, 9]]]
            # Rule: a digit is optional at only 1 cell in a square (such as 9 in our case).
            flattened = [val for row in board[k1][k2] for item in row for val in item]
            hist, bin_edges = np.histogram( flattened, bins=Bins )
            ele_unique = 1+ np.where(hist == 1)[0]
            new_unique = np.setdiff1d( ele_unique, curSquare_elements)  # [9] in our example
            if len(new_unique)>0:  # there is a new element we may fill
                for ele in new_unique:
                    do_break = False
                    # for every cell in current square:
                    for k3 in range(Dim):  # inside a square, rows:
                        for k4 in range(Dim):  # inside a square, cols:
                            if len(board[k1][k2][k3][k4])>1 and np.isin(ele, board[k1][k2][k3][k4]):  # only if cell isn't determined
                                board[k1][k2][k3][k4] = [int(ele)]
                                do_break = True
                                break
                        if do_break:
                            break
                curSquare_elements = np.append(curSquare_elements, new_unique)
                n_values_add = n_values_add + len(new_unique)

            # example 2: [ [[2, 4, 8], [5], [2, 4, 6]] , [[3], [4, 8], [7]] , [[2, 8], [1], [9]]]

            # for every cell in current square:
            for k3 in range(Dim):  # inside a square, rows:
                for k4 in range(Dim):  # inside a square, cols:

                    if len(board[k1][k2][k3][k4]) > 1:  # only if cell isn't determined
                        # remove elements appears in the same square
                        board[k1][k2][k3][k4] = np.setdiff1d( board[k1][k2][k3][k4] , curSquare_elements)

                        if len(board[k1][k2][k3][k4]) > 1:  # only if cell isn't determined
                            cur_row_elements = get_row_elements(board, Dim, k1, k3)
                            board[k1][k2][k3][k4] = np.setdiff1d(board[k1][k2][k3][k4], cur_row_elements)

                        if len(board[k1][k2][k3][k4]) > 1:  # only if cell isn't determined
                            cur_col_elements = get_col_elements(board, Dim, k2, k4)
                            board[k1][k2][k3][k4] = np.setdiff1d(board[k1][k2][k3][k4], cur_col_elements)

                        if len(board[k1][k2][k3][k4]) == 1: # cell wasn't determined but now it is (after the update)
                            n_values_add = n_values_add+1
                            curSquare_elements = np.append(curSquare_elements, board[k1][k2][k3][k4]) # this digit is determined

                    elif len(board[k1][k2][k3][k4]) == 1: # finished.  Ensure there was no error.
                        was_mistake = ensure_cell(board, Dim, k1, k2, k3, k4)
                        if was_mistake:
                            return board, n_values_add, was_mistake
                    elif len(board[k1][k2][k3][k4]) < 1: # there was a mistake
                        was_mistake = True
                        return board, n_values_add, was_mistake



            # complicated rule: if in a square, a digit must be in certain line, then we may remove it from "options list" in the rest of the line.
            curSqr_opt = np.setdiff1d( Vector, curSquare_elements)
            if len(curSqr_opt)>0: # not finished
                rows_opt = [curSqr_opt,curSqr_opt,curSqr_opt]  # all row starts with all possible digits
                for k3 in range(Dim):  # inside a square, rows:
                    for k4 in range(Dim):  # inside a square, cols:
                        rows_opt[k3] = np.setdiff1d( rows_opt[k3],board[k1][k2][k3][k4] )
                #rows_noWay = rows_opt.copy() # the remaining are the digits which can't appear in lines. Example:
                # rows_noWay = [ [1], [1, 4, 5], []]  # this means "1" can't appear in top line no middle line.
                # we may conduct that "1" must appear in bottom line and there for we can't exclude it from this line in other squares.
                flattened = []
                hist = np.zeros((Dim*Dim+1,))
                for k3 in range(Dim):
                    for val in rows_opt[k3]:
                        flattened.append(val)
                        hist[val] += 1
                ele_unique = np.where(hist == 2)[0]
                if len(ele_unique)>0:
                    for val in ele_unique:
                        for k3 in range(Dim):
                            if not np.isin(val, rows_opt[k3]): # then val must be in this line in Square.
                                # we may remove val from other lines
                                for l2 in range(Dim): # go to other square in that square's row
                                    if l2!=k2: # if not current square
                                        for l4 in range(Dim): # all columns
                                            if len(board[k1][l2][k3][l4]) > 1:  # only if cell isn't determined
                                                # debug
                                                if np.isin(val,board[k1][l2][k3][l4]):
                                                    #print([ board[k1][l2][k3][l4], int(val), [k1,l2,k3,l4] ])
                                                    really_removing = 1
                                                # must:
                                                board[k1][l2][k3][l4] = np.setdiff1d(board[k1][l2][k3][l4],val)
                # similar, for columns:
                cols_opt = [curSqr_opt,curSqr_opt,curSqr_opt]  # all row starts with all possible digits
                for k4 in range(Dim):  # inside a square, cols:
                    for k3 in range(Dim):  # inside a square, rows:
                        cols_opt[k4] = np.setdiff1d( cols_opt[k4],board[k1][k2][k3][k4] )
                hist = np.zeros((Dim*Dim+1,))
                for k4 in range(Dim):
                    for val in cols_opt[k4]:
                        hist[val] += 1
                ele_unique = np.where(hist == 2)[0]
                if len(ele_unique)>0:
                    for val in ele_unique:
                        for k4 in range(Dim):
                            if not np.isin(val, cols_opt[k4]): # then val must be in this column in Square.
                                # we may remove val from other lines
                                for l1 in range(Dim): # go to other square in that square's column
                                    if l1!=k1: # if not current square
                                        for l3 in range(Dim): # all rows
                                            if len(board[l1][k2][l3][k4]) > 1:  # only if cell isn't determined
                                                # debug
                                                if np.isin(val,board[l1][k2][l3][k4]):
                                                    really_removing = 1
                                                # must:
                                                board[l1][k2][l3][k4] = np.setdiff1d(board[l1][k2][l3][k4],val)




    return board, n_values_add, was_mistake


# get all the valid elements in certain cell position (k1,k2,k3,k4)
def get_row_elements(board, Dim, k1, k3):
    # init
    cur_row_elements = []

    # update board
    for k2 in range(Dim):
        for k4 in range(Dim):
            if len( board[k1][k2][k3][k4] )==1:
                cur_row_elements.append( board[k1][k2][k3][k4][0] )
    return cur_row_elements

# get all the valid elements in certain cell position (k1,k2,k3,k4)
def get_col_elements(board, Dim, k2, k4):
    # init
    cur_col_elements = []

    # update board
    for k1 in range(Dim):
        for k3 in range(Dim):
            if len( board[k1][k2][k3][k4] )==1:
                cur_col_elements.append( board[k1][k2][k3][k4][0] )
    return cur_col_elements


# Determine if any mistakes occurred while solving the puzzle
def ensure_cell(board, Dim, k1, k2, k3, k4):
    # init
    other_col_elements = []
    other_row_elements = []

    # search for a mistake
    for l1 in range(Dim):
        for l3 in range(Dim):
            if l1==k1 and l3==k3:
                pass
            elif len(board[l1][k2][l3][k4]) == 1:
                other_col_elements.append(board[k1][k2][k3][k4][0])  # optional
                if board[k1][k2][k3][k4] == board[l1][k2][l3][k4]:
                    return True  # indicates that an error was detected in the solving attempt
    for l2 in range(Dim):
        for l4 in range(Dim):
            if l2==k2 and l4==k4:
                pass
            elif len(board[k1][l2][k3][l4]) == 1:
                other_row_elements.append(board[k1][k2][k3][k4][0])
                if board[k1][k2][k3][k4] == board[k1][l2][k3][l4]:
                    return True  #  indicates that an error was detected in the solving attempt
    return False  # there is no sign for an error while trying to solve the puzzle


# debug function:  enables to understand the matrix locations. Such as debug_cell_id(1,1,1,1)= 8+4+2+1= 15
def debug_cell_id( k1,k2,k3,k4 ):
    debug_mat = [  #  option 1
        [  # Up
            [[0, 1], [2, 3]], # Left
            [[4, 5], [6, 7]]  # Right
        ],
        [  # Down
            [[8,   9], [10, 11]],  # Left
            [[12, 13], [14, 15]]  # Right
        ]
    ]
    return debug_mat[k1][k2][k3][k4]


# prints the board
def print_board(board, Dim):
    # update board
    if Dim==2:
        print('################')  # 16 symbols
    elif Dim==3:
        print('###############################')  # 31 symbols

    for k1 in range(Dim):   # rows

        for k3 in range(Dim):  # inside a square, rows:
            sCurLine = '#'
            for k2 in range(Dim):  # columns

                for k4 in range(Dim):  # inside a square, cols:
                    if len(board[k1][k2][k3][k4])==1:
                        sCurLine += str( board[k1][k2][k3][k4] )
                    else : # empty
                        sCurLine += '[ ]'
                sCurLine += '#'

            print(sCurLine)
        if k1 < Dim:
            if Dim == 2:
                print('---------------')
            elif Dim == 3:
                print(' -----------------------------')


    if Dim == 2:
        print('################')  # 16 symbols
    elif Dim == 3:
        print('###############################')  # 31 symbols
    print(' ')


if __name__ == '__main__':
    solution, board0, Dim = load_board(7)

    board = init_board_options(board0, Dim)  # each cell may store values 1:Dim.  The board values removes many options.
    board, n_values_add, was_mistake = update_board(board, Dim)
    board_solution = solve_board(board, Dim, 0)

    print("solved board:")
    print_board(board_solution, Dim)
    # optional - plot_board using pygame.
    print('Finished')
