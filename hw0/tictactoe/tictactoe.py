"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    count_x = sum(row.count(X) for row in board)
    count_o = sum(row.count(O) for row in board)
    return X if count_x == count_o else O


def actions(board):
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    if action not in actions(board):
        raise Exception("Invalid action")

    new_board = [row[:] for row in board]  
    new_board[action[0]][action[1]] = player(board)
    return new_board



def winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not None:
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    return None



def terminal(board):
    if winner(board) is not None:
        return True
    if all(cell != EMPTY for row in board for cell in row):
        return True
    return False


def utility(board):
    winner_player = winner(board)
    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    else:
        return 0
   



def minimax(board):
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        value, move = max_value(board)
    else:
        value, move = min_value(board)

    return move

def max_value(board):
    if terminal(board):
        return utility(board), None

    v = -math.inf
    best_move = None
    for action in actions(board):
        new_value, _ = min_value(result(board, action))
        if new_value > v:
            v = new_value
            best_move = action
            if v == 1:  # Early termination if winning move is found
                break
    return v, best_move

def min_value(board):
    if terminal(board):
        return utility(board), None

    v = math.inf
    best_move = None
    for action in actions(board):
        new_value, _ = max_value(result(board, action))
        if new_value < v:
            v = new_value
            best_move = action
            if v == -1:  # Early termination if winning move is found
                break
    return v, best_move


"""
initial_state() -> returns starting state of the board as 3x3 grid with all cells to empty. 

player(board) -> determines whose turn it is by counring number of X and O on the board. if counts equal, it is X's turn, otherwise O's.

actions(board) -> counts set of all possible moves, each move is represented as tuple, i row index, j column. 

result(board,action) -> returns new board state after applying given action. 

winner(board) -> checks for winner by examining rows, columns and diagonals. 

terminal(board) -> game is over if there is winner or if all cells are filled that's a tie. 

utility(board) -> computes utility of terminal board state, returns 1 if x has won, -1 id o has won, 0 if tie. 

minimax(board) -> used to determine optinal move for current player, uses recursion, return optinal tuple. 

max_value(board) -> explores all possible moves, chooses highest utility. min_value explores all possible moves and chooses the one with the lowest utility. 
"""