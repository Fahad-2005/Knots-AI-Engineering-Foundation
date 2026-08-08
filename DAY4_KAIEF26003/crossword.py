from collections import deque


class Variable:
    def __init__(self, r, c, direction, length):
        self.r = r
        self.c = c
        self.direction = direction  # 'across' or 'down'
        self.length = length
        self.cells = []
        for i in range(length):
            if direction == "across":
                self.cells.append((r, c + i))
            else:
                self.cells.append((r + i, c))

    def __hash__(self):
        return hash((self.r, self.c, self.direction, self.length))

    def __eq__(self, other):
        return (self.r, self.c, self.direction, self.length) == (
            other.r, other.c, other.direction, other.length
        )

    def __repr__(self):
        return f"Var({self.r},{self.c},{self.direction},{self.length})"


class Crossword:
    def __init__(self, structure, words):
        self.height = len(structure)
        self.width = max(len(row) for row in structure)
        self.structure = structure
        self.words = set(words)
        self.variables = self._find_variables()
        self.overlaps = self._find_overlaps()
        self.domains = {
            var: [w for w in self.words if len(w) == var.length]
            for var in self.variables
        }

    def _find_variables(self):
        vars_set = set()
        # Find horizontal words
        for r in range(self.height):
            c = 0
            while c < self.width:
                if self.structure[r][c] and (c == 0 or not self.structure[r][c - 1]):
                    length = 0
                    while c + length < self.width and self.structure[r][c + length]:
                        length += 1
                    if length > 1:
                        vars_set.add(Variable(r, c, "across", length))
                    c += length
                else:
                    c += 1

        # Find vertical words
        for c in range(self.width):
            r = 0
            while r < self.height:
                if self.structure[r][c] and (r == 0 or not self.structure[r - 1][c]):
                    length = 0
                    while r + length < self.height and self.structure[r + length][c]:
                        length += 1
                    if length > 1:
                        vars_set.add(Variable(r, c, "down", length))
                    r += length
                else:
                    r += 1
        return vars_set

    def _find_overlaps(self):
        overlaps = {}
        for v1 in self.variables:
            for v2 in self.variables:
                if v1 == v2:
                    continue
                intersection = set(v1.cells).intersection(set(v2.cells))
                if intersection:
                    cell = list(intersection)[0]
                    idx1 = v1.cells.index(cell)
                    idx2 = v2.cells.index(cell)
                    overlaps[v1, v2] = (idx1, idx2)
                else:
                    overlaps[v1, v2] = None
        return overlaps


def revise(crossword, v1, v2):
    overlap = crossword.overlaps.get((v1, v2))
    if not overlap:
        return False

    i, j = overlap
    revised = False
    for w1 in crossword.domains[v1][:]:
        if not any(w1[i] == w2[j] for w2 in crossword.domains[v2] if w1 != w2):
            crossword.domains[v1].remove(w1)
            revised = True
    return revised


def ac3(crossword):
    queue = deque([
        (v1, v2) for v1 in crossword.variables
        for v2 in crossword.variables if crossword.overlaps.get((v1, v2))
    ])

    while queue:
        v1, v2 = queue.popleft()
        if revise(crossword, v1, v2):
            if not crossword.domains[v1]:
                return False
            for v_nbr in crossword.variables:
                if v_nbr != v1 and crossword.overlaps.get((v_nbr, v1)):
                    queue.append((v_nbr, v1))
    return True


def select_unassigned_mrv(assignment, crossword):
    unassigned = [v for v in crossword.variables if v not in assignment]
    return min(unassigned, key=lambda var: len(crossword.domains[var]))


def is_consistent(var, word, assignment, crossword):
    if word in assignment.values():
        return False  # Words must be unique

    for other_var, other_word in assignment.items():
        overlap = crossword.overlaps.get((var, other_var))
        if overlap:
            i, j = overlap
            if word[i] != other_word[j]:
                return False
    return True


def backtrack(assignment, crossword):
    if len(assignment) == len(crossword.variables):
        return assignment

    var = select_unassigned_mrv(assignment, crossword)

    for word in crossword.domains[var]:
        if is_consistent(var, word, assignment, crossword):
            assignment[var] = word
            result = backtrack(assignment, crossword)
            if result is not None:
                return result
            del assignment[var]

    return None


def print_crossword(solution, crossword):
    grid = [
        ["#" if not crossword.structure[r][c] else " " for c in range(crossword.width)]
        for r in range(crossword.height)
    ]

    for var, word in solution.items():
        for idx, (r, c) in enumerate(var.cells):
            grid[r][c] = word[idx]

    print("\nSolved Crossword Grid:")
    for row in grid:
        print(" ".join(row))


def main():
    # Grid structure: True = open cell, False = wall (#)
    structure = [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ]

    vocabulary = ["CAT", "DOG", "COD", "TAG", "SUN", "RED", "BAT", "LOG"]

    crossword = Crossword(structure, vocabulary)

    if ac3(crossword):
        solution = backtrack({}, crossword)
        if solution:
            print_crossword(solution, crossword)
        else:
            print("No solution found.")
    else:
        print("No solution possible (failed AC-3).")


if __name__ == "__main__":
    main()