#!/usr/bin/env python3
"""
main.py
-------
Core simulation engine (Random / RL / ML) + metrics + headless mode.
No Tkinter code lives here – everything UI-related is in simulation.py.
"""

import numpy as np

class WirelessSimulation:
    def __init__(self, num_agents: int, num_bands: int, num_time_steps: int, algorithm: str = 'Random'):
        self.num_agents = num_agents
        self.num_bands = num_bands
        self.num_time_steps = num_time_steps
        self.algorithm = algorithm

        # Success counters
        self.success_counts = [0] * num_agents

        # Text grid for terminal output
        self.grid_texts = [['' for _ in range(num_time_steps)] for _ in range(num_bands)]

        # RL / ML internal states
        self.rl_states = None
        self.ml_states = None
        if algorithm == 'Reinforcement Learning':
            self.rl_states = [{'Q': np.ones(num_bands) * 0.5,
                               'alpha': 0.05,
                               'epsilon': 0.5} for _ in range(num_agents)]
        elif algorithm == 'Machine Learning':
            self.ml_states = [{'successes': np.zeros(num_bands),
                               'failures': np.zeros(num_bands)} for _ in range(num_agents)]

    # --------------------------------------------------------------------- #
    # RL helpers
    # --------------------------------------------------------------------- #
    def rl_approach(self, agent: int) -> int:
        state = self.rl_states[agent]
        if np.random.rand() < state['epsilon']:
            return np.random.randint(self.num_bands)
        return int(np.argmax(state['Q']))

    def rl_update(self, agent: int, action: int, reward: float) -> None:
        state = self.rl_states[agent]
        state['Q'][action] += state['alpha'] * (reward - state['Q'][action])
        state['epsilon'] *= 0.995
        state['epsilon'] = max(0.01, state['epsilon'])

    # --------------------------------------------------------------------- #
    # ML (Thompson-sampling) helpers
    # --------------------------------------------------------------------- #
    def ml_approach(self, agent: int) -> int:
        state = self.ml_states[agent]
        samples = np.random.beta(1 + state['successes'], 1 + state['failures'])
        return int(np.argmax(samples))

    def ml_update(self, agent: int, action: int, reward: float) -> None:
        state = self.ml_states[agent]
        if reward > 0:
            state['successes'][action] += 1
        else:
            state['failures'][action] += 1

    # --------------------------------------------------------------------- #
    # Decision making
    # --------------------------------------------------------------------- #
    def get_transmissions(self, t: int):
        decisions = []
        for a in range(self.num_agents):
            if self.algorithm == 'Random':
                band = np.random.randint(self.num_bands)
            elif self.algorithm == 'Reinforcement Learning':
                band = self.rl_approach(a)
            elif self.algorithm == 'Machine Learning':
                band = self.ml_approach(a)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            decisions.append((True, band))
        return decisions

    # --------------------------------------------------------------------- #
    # One simulation step
    # --------------------------------------------------------------------- #
    def step(self, t: int) -> None:
        decisions = self.get_transmissions(t)
        band_usage = [[] for _ in range(self.num_bands)]

        for a, (transmit, band) in enumerate(decisions):
            if transmit and band is not None:
                band_usage[band].append(a)

        # Resolve collisions / successes
        for b, users in enumerate(band_usage):
            if len(users) == 1:
                agent = users[0]
                self.success_counts[agent] += 1
                if self.algorithm == 'Reinforcement Learning':
                    self.rl_update(agent, b, 1)
                elif self.algorithm == 'Machine Learning':
                    self.ml_update(agent, b, 1)
            elif len(users) > 1:
                for agent in users:
                    if self.algorithm == 'Reinforcement Learning':
                        self.rl_update(agent, b, 0)
                    elif self.algorithm == 'Machine Learning':
                        self.ml_update(agent, b, 0)

            # Fill terminal-grid text
            if len(users) == 0:
                txt = ""
            elif len(users) == 1:
                txt = f"{users[0] + 1}"
            else:
                txt = ",".join(str(a + 1) for a in sorted(users))
                if len(txt) > 12:
                    txt = txt[:12] + "..."
            self.grid_texts[b][t] = txt

    # --------------------------------------------------------------------- #
    # Terminal output
    # --------------------------------------------------------------------- #
    def print_grid(self) -> None:
        print("\nTransmission Grid (Y: Bands, X: Time Steps)")
        line = "-" * (13 + self.num_time_steps * 14)
        print(line)
        header = "Band \\ Time | " + " | ".join([f"t={i+1:2d}" for i in range(self.num_time_steps)])
        print(header)
        print(line)
        for b in range(self.num_bands):
            row = f"B{b+1:8} | " + " | ".join([f"{self.grid_texts[b][t]:<12}" for t in range(self.num_time_steps)])
            print(row)
        print(line)

    # --------------------------------------------------------------------- #
    # Metrics
    # --------------------------------------------------------------------- #
    def compute_metrics(self):
        sent = np.array(self.success_counts, dtype=float)
        total_sent = sent.sum()
        total_slots = self.num_time_steps * self.num_bands
        overall = total_sent / total_slots if total_slots else 0.0
        per_agent = sent / total_slots if total_slots else np.zeros(self.num_agents)

        sum_th = sent.sum()
        sum_sq = (sent ** 2).sum()
        fairness = (sum_th ** 2) / (self.num_agents * sum_sq) if sum_sq else 0.0

        return overall, fairness, per_agent

    # --------------------------------------------------------------------- #
    # Headless execution (used by UI when visualisation is disabled)
    # --------------------------------------------------------------------- #
    def run_headless(self) -> tuple:
        for t in range(self.num_time_steps):
            self.step(t)
        self.print_grid()
        return self.compute_metrics()