from logic import Symbol, And, Or, Not, Implication, model_check

# Suspects
mustard = Symbol("Col. Mustard")
plum = Symbol("Prof. Plum")
scarlet = Symbol("Ms. Scarlet")

# Weapons
knife = Symbol("Knife")
candlestick = Symbol("Candlestick")
revolver = Symbol("Revolver")

# Rooms
ballroom = Symbol("Ballroom")
kitchen = Symbol("Kitchen")
library = Symbol("Library")

all_symbols = [mustard, plum, scarlet, knife, candlestick, revolver, ballroom, kitchen, library]

# Clue Knowledge Base
knowledge = And(
    # At least one card in envelope per category
    Or(mustard, plum, scarlet),
    Or(knife, candlestick, revolver),
    Or(ballroom, kitchen, library),

    # Hands / Revealed innocents (Cards we hold cannot be in the envelope)
    Not(mustard),
    Not(knife),
    Not(revolver),
    Not(ballroom),
    Not(kitchen)
)


def deduce():
    print("Deductions from Clue Knowledge Base:")
    print("-" * 35)
    for symbol in all_symbols:
        if model_check(knowledge, symbol):
            print(f"  [CONFIRMED IN ENVELOPE]: {symbol.name}")
        elif model_check(knowledge, Not(symbol)):
            print(f"  [INNOCENT / CLEARED]:    {symbol.name}")


if __name__ == "__main__":
    deduce()