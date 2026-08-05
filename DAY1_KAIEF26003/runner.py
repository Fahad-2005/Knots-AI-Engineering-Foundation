from tictactoe import initial_state, player, result, terminal, winner, minimax

def render(board):
    print("\n  0 1 2")
    for idx, row in enumerate(board):
        cells = [c if c is not None else " " for c in row]
        print(f"{idx} " + "|".join(cells))
    print()

def main():
    board = initial_state()
    print("AI plays X, you play O.")

    while not terminal(board):
        render(board)
        
        if player(board) == "O":
            try:
                move_str = input("Your turn (row,col): ")
                r, c = map(int, move_str.split(","))
                board = result(board, (r, c))
            except Exception:
                print("Invalid input! Enter coordinates as row,col (e.g. 0,1).")
                continue
        else:
            print("AI thinking...")
            move = minimax(board)
            board = result(board, move)

    render(board)
    game_winner = winner(board)
    if game_winner:
        print(f"Winner: {game_winner}")
    else:
        print("Draw!")

if __name__ == "__main__":
    main()