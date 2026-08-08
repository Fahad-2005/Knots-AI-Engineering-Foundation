from collections import deque


class NQueensCSP:
    def __init__(self, n=8):
        self.n = n
        self.variables = list(range(n))  # Col indices 0..N-1
        self.domains = {col: list(range(n)) for col in self.variables}  # Rows 0..N-1
        self.neighbors = {col: [c for c in self.variables if c != col] for col in self.variables}

    def is_valid_pair(self, col1, row1, col2, row2):
        """Returns True if two queens do not attack each other."""
        if row1 == row2:
            return False  # Same row
        if abs(row1 - row2) == abs(col1 - col2):
            return False  # Same diagonal
        return True

    def revise(self, col1, col2):
        """Removes values from col1's domain that have no support in col2."""
        revised = False
        for row1 in self.domains[col1][:]:
            if not any(self.is_valid_pair(col1, row1, col2, row2) for row2 in self.domains[col2]):
                self.domains[col1].remove(row1)
                revised = True
        return revised

    def ac3(self):
        """Applies AC-3 algorithm to enforce arc consistency."""
        queue = deque([(c1, c2) for c1 in self.variables for c2 in self.neighbors[c1]])

        while queue:
            col1, col2 = queue.popleft()
            if self.revise(col1, col2):
                if not self.domains[col1]:
                    return False  # Domain wiped out
                for nbr in self.neighbors[col1]:
                    if nbr != col2:
                        queue.append((nbr, col1))
        return True


def select_unassigned_mrv(assignment, csp):
    """MRV Heuristic: Selects the unassigned variable with the smallest domain."""
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda var: len(csp.domains[var]))


def is_consistent(var, row, assignment, csp):
    """Checks if placing queen at (var, row) violates any assigned constraints."""
    for assigned_var, assigned_row in assignment.items():
        if not csp.is_valid_pair(var, row, assigned_var, assigned_row):
            return False
    return True


def backtrack(assignment, csp):
    """Backtracking search to find a complete consistent assignment."""
    if len(assignment) == len(csp.variables):
        return assignment

    var = select_unassigned_mrv(assignment, csp)

    for row in csp.domains[var]:
        if is_consistent(var, row, assignment, csp):
            assignment[var] = row
            result = backtrack(assignment, csp)
            if result is not None:
                return result
            del assignment[var]  # Undo choice

    return None


def print_board(solution, n):
    print(f"\nSolution for {n}-Queens:")
    print("-" * (2 * n + 1))
    for r in range(n):
        row_str = "| "
        for c in range(n):
            row_str += "Q " if solution[c] == r else ". "
        print(row_str + "|")
    print("-" * (2 * n + 1))


def main():
    n = 8
    csp = NQueensCSP(n)

    # Step 1: Preprocess with AC-3
    if not csp.ac3():
        print("No solution possible after AC-3 arc consistency check.")
        return

    # Step 2: Backtracking with MRV
    solution = backtrack({}, csp)

    if solution:
        print_board(solution, n)
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()