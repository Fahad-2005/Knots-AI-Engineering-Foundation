# Topological variable ordering
VARIABLES = ["fire", "smoke", "alarm"]

# Conditional Probability Tables (CPTs)
# P(Fire)
CPT_FIRE = {True: 0.01, False: 0.99}

# P(Smoke | Fire)
CPT_SMOKE = {
    True: {True: 0.90, False: 0.10},   # P(Smoke=True|Fire=True), P(Smoke=False|Fire=True)
    False: {True: 0.05, False: 0.95}   # P(Smoke=True|Fire=False), P(Smoke=False|Fire=False)
}

# P(Alarm | Fire, Smoke)
CPT_ALARM = {
    (True, True):   {True: 0.99, False: 0.01},
    (True, False):  {True: 0.85, False: 0.15},
    (False, True):  {True: 0.50, False: 0.50},
    (False, False): {True: 0.001, False: 0.999}
}


def node_probability(var, value, assignment):
    """Evaluates local conditional probability P(var=value | parents)."""
    if var == "fire":
        return CPT_FIRE[value]
    elif var == "smoke":
        parent_val = assignment["fire"]
        return CPT_SMOKE[parent_val][value]
    elif var == "alarm":
        parent_vals = (assignment["fire"], assignment["smoke"])
        return CPT_ALARM[parent_vals][value]


def joint_probability(assignment):
    """Computes joint probability of a complete assignment of all variables."""
    prob = 1.0
    for var in VARIABLES:
        prob *= node_probability(var, assignment[var], assignment)
    return prob


def enumerate_all(vars_left, assignment):
    """Recursively sums joint probabilities over all unassigned hidden variables."""
    if not vars_left:
        return joint_probability(assignment)

    first = vars_left[0]
    rest = vars_left[1:]

    if first in assignment:
        return enumerate_all(rest, assignment)
    else:
        total = 0.0
        for val in [True, False]:
            new_assignment = assignment.copy()
            new_assignment[first] = val
            total += enumerate_all(rest, new_assignment)
        return total


def query(query_var, evidence):
    """Computes exact posterior probability P(query_var | evidence) via enumeration."""
    distribution = {}

    for val in [True, False]:
        assignment = evidence.copy()
        assignment[query_var] = val
        distribution[val] = enumerate_all(VARIABLES, assignment)

    # Normalize distribution so probabilities sum to 1.0
    total = sum(distribution.values())
    return {k: v / total for k, v in distribution.items()}


def main():
    print("--- Bayesian Network Exact Inference ---")

    # Scenario 1: P(Fire | Alarm = True)
    p_fire_given_alarm = query("fire", {"alarm": True})
    print("\nScenario 1: Alarm is ringing")
    print(f"  P(Fire = True  | Alarm = True) = {p_fire_given_alarm[True]:.4f}")
    print(f"  P(Fire = False | Alarm = True) = {p_fire_given_alarm[False]:.4f}")

    # Scenario 2: P(Fire | Alarm = True, Smoke = True)
    p_fire_given_alarm_smoke = query("fire", {"alarm": True, "smoke": True})
    print("\nScenario 2: Alarm is ringing AND Smoke is present")
    print(f"  P(Fire = True  | Alarm, Smoke) = {p_fire_given_alarm_smoke[True]:.4f}")
    print(f"  P(Fire = False | Alarm, Smoke) = {p_fire_given_alarm_smoke[False]:.4f}")


if __name__ == "__main__":
    main()