class Symbol:
    def __init__(self, name):
        self.name = name

    def evaluate(self, model):
        try:
            return bool(model[self.name])
        except KeyError:
            raise Exception(f"variable '{self.name}' not in model")

    def symbols(self):
        return {self.name}

    def __repr__(self):
        return self.name


class Not:
    def __init__(self, operand):
        self.operand = operand

    def evaluate(self, model):
        return not self.operand.evaluate(model)

    def symbols(self):
        return self.operand.symbols()


class And:
    def __init__(self, *conjuncts):
        self.conjuncts = list(conjuncts)

    def evaluate(self, model):
        return all(c.evaluate(model) for c in self.conjuncts)

    def symbols(self):
        return set().union(*[c.symbols() for c in self.conjuncts])


class Or:
    def __init__(self, *disjuncts):
        self.disjuncts = list(disjuncts)

    def evaluate(self, model):
        return any(d.evaluate(model) for d in self.disjuncts)

    def symbols(self):
        return set().union(*[d.symbols() for d in self.disjuncts])


class Implication:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, model):
        return not self.antecedent.evaluate(model) or self.consequent.evaluate(model)

    def symbols(self):
        return self.antecedent.symbols() | self.consequent.symbols()


class Biconditional:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, model):
        return self.left.evaluate(model) == self.right.evaluate(model)

    def symbols(self):
        return self.left.symbols() | self.right.symbols()


def model_check(knowledge, query):
    symbols = list(knowledge.symbols() | query.symbols())
    return check_all(knowledge, query, symbols, {})


def check_all(knowledge, query, symbols, model):
    if not symbols:
        if knowledge.evaluate(model):
            return query.evaluate(model)
        return True

    remaining = symbols[1:]
    sym = symbols[0]

    return (
        check_all(knowledge, query, remaining, {**model, sym: True}) and
        check_all(knowledge, query, remaining, {**model, sym: False})
    )