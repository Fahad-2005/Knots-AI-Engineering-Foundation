from logic import Symbol, And, Or, Not, Implication, Biconditional, model_check

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Structural rules: Everyone is either a Knight or a Knave, but not both
base_rules = And(
    Or(AKnight, AKnave), Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave), Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave), Not(And(CKnight, CKnave))
)

# Puzzle 0: A says "I am both a knight and a knave."
knowledge0 = And(
    base_rules,
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1: A says "We are both knaves." B says nothing.
knowledge1 = And(
    base_rules,
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2: A says "We are the same kind." B says "We are of different kinds."
knowledge2 = And(
    base_rules,
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)))
)

# Puzzle 3:
# A says "I am a knight" or "I am a knave" (unclear which).
# B says "A said 'I am a knave'".
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    base_rules,
    Biconditional(AKnight, Or(AKnight, AKnave)),
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    Biconditional(BKnight, CKnave),
    Biconditional(CKnight, AKnight)
)


def solve():
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]

    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]

    for name, knowledge in puzzles:
        print(f"\n--- {name} ---")
        for symbol in symbols:
            if model_check(knowledge, symbol):
                print(f"  {symbol.name}")


if __name__ == "__main__":
    solve()