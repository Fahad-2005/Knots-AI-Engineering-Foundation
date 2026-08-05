import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O


def actions(board):
    moves = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == EMPTY:
                moves.append((r, c))
    return moves


def result(board, action):
    r, c = action
    if board[r][c] is not EMPTY:
        raise ValueError("Cell already occupied")
    
    # Deepcopy keeps minimax branches completely isolated
    new_board = copy.deepcopy(board)
    new_board[r][c] = player(board)
    return new_board


def winner(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None


def terminal(board):
    return winner(board) is not None or len(actions(board)) == 0


def utility(board):
    w = winner(board)
    if w == X:
        return 1
    elif w == O:
        return -1
    return 0


def minimax(board):
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        _, move = max_value(board, -float("inf"), float("inf"))
    else:
        _, move = min_value(board, -float("inf"), float("inf"))

    return move


def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = -float("inf")
    best_move = None

    for action in actions(board):
        min_val, _ = min_value(result(board, action), alpha, beta)
        if min_val > v:
            v = min_val
            best_move = action
        alpha = max(alpha, v)
        if beta <= alpha:
            break

    return v, best_move


def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = float("inf")
    best_move = None

    for action in actions(board):
        max_val, _ = max_value(result(board, action), alpha, beta)
        if max_val < v:
            v = max_val
            best_move = action
        beta = min(beta, v)
        if beta <= alpha:
            break

    return v, best_move