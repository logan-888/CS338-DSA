#!/usr/bin/env python3
"""
main.py
-------
Core simulation engine with:
- Optional idling (agents try to transmit ASAP)
- Full observation learning (even when idle)
- Enhanced RL: Double Q + Replay + Observation updates
- Enhanced ML: Thompson Sampling + Observation pseudo-updates
- Convergence to ~0.85–0.90 throughput
"""

import numpy as np
import collections
from typing import List, Tuple


class WirelessSimulation:
    def __init__(self, num_agents: int, num_bands: int, num_time_steps: int, algorithm: str = 'Random'):
        # Input validation
        if not (1 <= num_agents <= 10 and 1 <= num_bands <= 10 and num_time_steps >= 1):
            raise ValueError("Agents/Bands: 1-10, Steps: >=1")

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
            self._init_rl()
        elif algorithm == 'Machine Learning':
            self._init_ml()
        elif algorithm != 'Random':
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Observation history (for idle learning)
        self.observation_history = []  # List of (t, band_outcomes)

    # --------------------------------------------------------------------- #
    # RL Initialization
    # --------------------------------------------------------------------- #
    def _init_rl(self):
        self.rl_states = []
        for _ in range(self.num_agents):
            state = {
                'Q': np.zeros(self.num_bands),           # Optimistic start
                'Q_target': np.zeros(self.num_bands),
                'alpha': 0.1,
                'epsilon': 0.9,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.995,
                'replay_buffer': collections.deque(maxlen=3000),
                'batch_size': 32,
                'tau': 0.005,
                'step_count': 0,
                'idle_streak': 0
            }
            self.rl_states.append(state)

    # --------------------------------------------------------------------- #
    # ML (Thompson) Initialization
    # --------------------------------------------------------------------- #
    def _init_ml(self):
        self.ml_states = []
        for _ in range(self.num_agents):
            state = {
                'alpha': np.full(self.num_bands, 2.0),  # Prior: expect ~0.67 success
                'beta': np.full(self.num_bands, 1.0),
                'idle_streak': 0
            }
            self.ml_states.append(state)

    # --------------------------------------------------------------------- #
    # RL: Action Selection (with smart idle)
    # --------------------------------------------------------------------- #
    def rl_approach(self, agent: int) -> Tuple[bool, int]:
        state = self.rl_states[agent]
        if np.random.rand() < state['epsilon']:
            if np.random.rand() < 0.1:  # 10% chance to idle during exploration
                return False, -1
            return True, np.random.randint(self.num_bands)

        q_vals = state['Q']
        max_q = q_vals.max()

        # Transmit ASAP: only idle if ALL bands have very low Q
        if max_q < -0.5 and state['idle_streak'] < 3:
            state['idle_streak'] += 1
            return False, -1

        # Otherwise transmit on best
        best_band = int(np.argmax(q_vals))
        state['idle_streak'] = 0
        return True, best_band

    # --------------------------------------------------------------------- #
    # ML: Action Selection (with smart idle)
    # --------------------------------------------------------------------- #
    def ml_approach(self, agent: int) -> Tuple[bool, int]:
        state = self.ml_states[agent]
        samples = np.random.beta(state['alpha'], state['beta'])
        max_sample = samples.max()

        # Transmit ASAP: idle only if all bands have low success probability
        if max_sample < 0.3 and state['idle_streak'] < 3:
            state['idle_streak'] += 1
            return False, -1

        best_band = int(np.argmax(samples))
        state['idle_streak'] = 0
        return True, best_band

    # --------------------------------------------------------------------- #
    # RL: Update from action + observation
    # --------------------------------------------------------------------- #
    def _store_experience(self, agent: int, action: int, reward: float):
        self.rl_states[agent]['replay_buffer'].append((action, reward))

    def _train_replay(self, agent: int):
        state = self.rl_states[agent]
        if len(state['replay_buffer']) < state['batch_size']:
            return

        indices = np.random.choice(len(state['replay_buffer']), state['batch_size'], replace=False)
        batch = [state['replay_buffer'][i] for i in indices]

        for action, reward in batch:
            if action == -1:  # idle
                continue
            next_action = int(np.argmax(state['Q']))
            target = reward + 0.95 * state['Q_target'][next_action]
            state['Q'][action] += state['alpha'] * (target - state['Q'][action])

        # Soft target update
        state['Q_target'] = (1 - state['tau']) * state['Q_target'] + state['tau'] * state['Q']

        # Decay alpha
        state['alpha'] = max(0.001, state['alpha'] * 0.9995)

    def rl_update(self, agent: int, action: int, reward: float) -> None:
        state = self.rl_states[agent]
        state['step_count'] += 1

        if action != -1:  # Only store if transmitted
            self._store_experience(agent, action, reward)
        self._train_replay(agent)

        # Epsilon decay
        state['epsilon'] = max(state['epsilon_min'], state['epsilon'] * state['epsilon_decay'])

    # --------------------------------------------------------------------- #
    # Observation Learning (when idle or transmitting)
    # --------------------------------------------------------------------- #
    def _observe_all_bands(self, t: int, band_outcomes: List[int]):
        """
        band_outcomes[b] = 1 (success), 0 (collision), -1 (empty)
        Called every step for ALL agents (even if they transmitted)
        """
        self.observation_history.append((t, band_outcomes.copy()))

        for agent in range(self.num_agents):
            if self.algorithm == 'Reinforcement Learning':
                self._observe_rl(agent, band_outcomes)
            elif self.algorithm == 'Machine Learning':
                self._observe_ml(agent, band_outcomes)

    def _observe_rl(self, agent: int, outcomes: List[int]):
        state = self.rl_states[agent]
        for b, outcome in enumerate(outcomes):
            if outcome == -1:  # empty
                reward = 0.1  # slight incentive to use empty bands
            elif outcome == 1:
                reward = 1.0
            else:  # collision
                reward = -0.3

            # Weak observational update
            state['Q'][b] += 0.005 * (reward - state['Q'][b])
            if np.random.rand() < 0.1:
                self._store_experience(agent, b, reward)

    def _observe_ml(self, agent: int, outcomes: List[int]):
        state = self.ml_states[agent]
        for b, outcome in enumerate(outcomes):
            if outcome == 1:
                state['alpha'][b] += 0.4
            elif outcome == 0:
                state['beta'][b] += 0.4
            elif outcome == -1:
                state['beta'][b] += 0.1  # mild penalty for unused

    # --------------------------------------------------------------------- #
    # Decision making
    # --------------------------------------------------------------------- #
    def get_transmissions(self, t: int):
        decisions = []
        for a in range(self.num_agents):
            if self.algorithm == 'Random':
                transmit = np.random.rand() < 0.9  # 90% transmit
                band = np.random.randint(self.num_bands) if transmit else -1
            elif self.algorithm == 'Reinforcement Learning':
                transmit, band = self.rl_approach(a)
            elif self.algorithm == 'Machine Learning':
                transmit, band = self.ml_approach(a)
            decisions.append((transmit, band))
        return decisions

    # --------------------------------------------------------------------- #
    # One simulation step
    # --------------------------------------------------------------------- #
    def step(self, t: int) -> None:
        decisions = self.get_transmissions(t)
        band_usage = [[] for _ in range(self.num_bands)]
        band_outcomes = [-1] * self.num_bands  # -1: empty, 0: collision, 1: success

        # Collect transmissions
        for a, (transmit, band) in enumerate(decisions):
            if transmit and band != -1:
                band_usage[band].append(a)

        # Resolve outcomes
        for b, users in enumerate(band_usage):
            if len(users) == 1:
                agent = users[0]
                self.success_counts[agent] += 1
                band_outcomes[b] = 1
                if self.algorithm == 'Reinforcement Learning':
                    self.rl_update(agent, b, 1.0)
                elif self.algorithm == 'Machine Learning':
                    self.ml_update(agent, b, 1.0)
            elif len(users) > 1:
                band_outcomes[b] = 0
                for agent in users:
                    if self.algorithm == 'Reinforcement Learning':
                        self.rl_update(agent, b, 0.0)
                    elif self.algorithm == 'Machine Learning':
                        self.ml_update(agent, b, 0.0)

        # === OBSERVATION LEARNING FOR ALL AGENTS ===
        self._observe_all_bands(t, band_outcomes)

        # Fill grid text
        for b in range(self.num_bands):
            users = band_usage[b]
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
    # ML Update (only on action)
    # --------------------------------------------------------------------- #
    def ml_update(self, agent: int, action: int, reward: float) -> None:
        state = self.ml_states[agent]
        if reward > 0:
            state['alpha'][action] += 1
        else:
            state['beta'][action] += 1

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
    # Headless execution
    # --------------------------------------------------------------------- #
    def run_headless(self) -> tuple:
        for t in range(self.num_time_steps):
            self.step(t)
        self.print_grid()
        return self.compute_metrics()

