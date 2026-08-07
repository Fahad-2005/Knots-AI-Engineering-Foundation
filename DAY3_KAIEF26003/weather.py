import random

# Transition probabilities: T[current_state][next_state]
TRANSITIONS = {
    "sun": {"sun": 0.7, "cloud": 0.2, "rain": 0.1},
    "cloud": {"sun": 0.3, "cloud": 0.4, "rain": 0.3},
    "rain": {"sun": 0.2, "cloud": 0.3, "rain": 0.5},
}


def sample_next_state(current_state, transitions):
    """Samples the next state according to the transition probabilities of current_state."""
    dist = transitions[current_state]
    r = random.random()
    cumulative = 0.0
    for state, prob in dist.items():
        cumulative += prob
        if r <= cumulative:
            return state
    return current_state


def simulate_weather(start_state, steps, transitions):
    """Simulates a sequence of weather states over N steps."""
    current = start_state
    history = [current]
    for _ in range(steps):
        current = sample_next_state(current, transitions)
        history.append(current)
    return history


def compute_steady_state(transitions, iterations=1000):
    """Calculates stationary probability distribution via power iteration."""
    states = list(transitions.keys())
    # Start with a uniform distribution across states
    dist = {s: 1.0 / len(states) for s in states}

    for _ in range(iterations):
        new_dist = {s: 0.0 for s in states}
        for source in states:
            for target in states:
                new_dist[target] += dist[source] * transitions[source][target]
        dist = new_dist

    return dist


def main():
    print("--- 10-Day Weather Simulation ---")
    simulation = simulate_weather("sun", 10, TRANSITIONS)
    print(" -> ".join(simulation))

    print("\n--- Long-Run Steady-State Distribution ---")
    steady = compute_steady_state(TRANSITIONS)
    for state, prob in steady.items():
        print(f"  {state.capitalize():<6}: {prob:.4f} ({prob * 100:.2f}%)")


if __name__ == "__main__":
    main()