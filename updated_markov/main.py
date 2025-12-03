#!/usr/bin/env python3
"""
main.py
-------
Core simulation engine for dynamic wireless channel allocation.

The simulation models multiple agents competing for limited frequency bands
over discrete time steps. It implements various control algorithms:
* Random
* Reinforcement Learning (Tabular SARSA)
* Machine Learning (Thompson Sampling)
* Advanced RL (Deep Q-Network with experience replay and time encoding)

Key Features:
* Markov chain for agent transmit/idle decision persistence.
* Observation learning from band outcomes (success/collision/empty).
* Throughput calculation: successful / attempted transmissions.
"""

import os
import collections
import random
from typing import List, Tuple, Optional

# Set environment variables to control threading behavior for libraries like NumPy/Torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def jains_fairness(throughputs: List[float]) -> float:
    """
    Calculates Jain's fairness index for a list of throughput values.

    Args:
        throughputs: A list of non-negative throughput values.

    Returns:
        The Jain's fairness index (0.0 to 1.0).
    """
    if not throughputs:
        return 1.0
    num_agents = len(throughputs)
    sum_x = sum(throughputs)
    sum_x2 = sum(x * x for x in throughputs)
    if sum_x == 0:
        return 1.0
    return (sum_x ** 2) / (num_agents * sum_x2)


class WirelessSimulation:
    """
    Manages the state and execution logic for the wireless channel allocation simulation.
    """
    def __init__(self, num_agents: int, num_bands: int, num_time_steps: int, algorithm: str = 'Random'):
        """
        Initializes the simulation environment and algorithm specific states.

        Args:
            num_agents: The number of transmitting agents.
            num_bands: The number of available frequency bands.
            num_time_steps: The total number of steps to run.
            algorithm: The channel selection strategy ('Random', 'Reinforcement Learning', etc.).
        """
        if not (1 <= num_agents and 1 <= num_bands and num_time_steps >= 1):
            raise ValueError("Agents/Bands must be >= 1, Steps must be >= 1")

        self.num_agents = num_agents
        self.num_bands = num_bands
        self.num_time_steps = num_time_steps
        self.algorithm = algorithm

        # Metrics
        self.success_counts = [0] * num_agents
        self.attempt_counts = [0] * num_agents
        self.step_throughputs: List[float] = []

        # Markov chain state: last action (True: transmitted, False: idled)
        self.last_actions = [True] * num_agents

        # Grid for visual output (stores text representation of band usage)
        self.grid_texts = [['' for _ in range(num_time_steps)] for _ in range(num_bands)]

        # Algorithm-specific state containers
        self.rl_states: Optional[List[dict]] = None
        self.ml_states: Optional[List[dict]] = None
        self.advanced_rl_states: Optional[List[dict]] = None
        self.observation_history: List[Tuple[int, List[int]]] = []

        # Initialize specific algorithm logic
        if algorithm == 'Reinforcement Learning':
            self._init_rl()
        elif algorithm == 'Machine Learning':
            self._init_ml()
        elif algorithm == 'Advanced RL':
            self._init_advanced_rl()
        elif algorithm != 'Random':
            raise ValueError(
                f"Unknown algorithm: {algorithm}. Must be 'Random', 'Reinforcement Learning', 'Machine Learning', or 'Advanced RL'.")

    # ---------------------------------------------------------------------
    # Algorithm Initialization
    # ---------------------------------------------------------------------

    def _init_rl(self):
        """Initializes tabular SARSA state for each agent."""
        self.rl_states = []
        for _ in range(self.num_agents):
            state = {
                'Q': np.zeros(self.num_bands),
                'alpha': 0.5,  # Learning rate
                'gamma': 0.95,  # Discount factor
                'epsilon': 1.0,  # Exploration rate
                'epsilon_min': 0.05,
                'epsilon_decay': 0.999,
                'last_action': -1,  # Previous band selected (or -1 if idle)
                'last_reward': 0.0,
                'step_count': 0
            }
            self.rl_states.append(state)

    def _init_ml(self):
        """Initializes Beta distribution priors for Thompson Sampling."""
        self.ml_states = []
        for _ in range(self.num_agents):
            state = {
                'alpha': np.full(self.num_bands, 1.0),  # Success count prior
                'beta': np.full(self.num_bands, 1.0)  # Failure/Collision count prior
            }
            self.ml_states.append(state)

    def _init_advanced_rl(self):
        """Initializes Deep Q-Network (DQN) architecture and state."""
        # --- Deep Q-Network Model ---
        class QNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        # --- DQN State Setup ---
        self.advanced_rl_states = []
        # State: 3 steps of band outcomes (3 * num_bands) + Time encoding (2)
        input_size = self.num_bands * 3 + 2
        # Action: num_bands (select band) + 1 (idle)
        output_size = self.num_bands + 1
        hidden_size = 128

        # Initial state history (3 steps of normalized empty outcome)
        init_obs = [0.0] * self.num_bands
        init_obs_tensor = torch.tensor(init_obs, dtype=torch.float32)

        for _ in range(self.num_agents):
            policy_net = QNet(input_size, hidden_size, output_size)
            target_net = QNet(input_size, hidden_size, output_size)
            target_net.load_state_dict(policy_net.state_dict())
            optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
            # Experience Replay Buffer
            replay_buffer = collections.deque(maxlen=10000)
            # History of last 3 band outcome observations
            history = collections.deque(maxlen=3)
            for _ in range(3):
                history.append(init_obs_tensor.clone())

            state = {
                'policy_net': policy_net,
                'target_net': target_net,
                'optimizer': optimizer,
                'replay': replay_buffer,
                'history': history,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.999,
                'gamma': 0.99,
                'batch_size': 32,
                'tau': 0.005,  # Soft update parameter for target network
                'step_count': 0
            }
            self.advanced_rl_states.append(state)

    # ---------------------------------------------------------------------
    # Band Selection Logic
    # ---------------------------------------------------------------------

    def rl_select_band(self, agent: int) -> int:
        """Selects a band for a transmitting RL agent using epsilon-greedy on Q-values."""
        state = self.rl_states[agent]
        if np.random.rand() < state['epsilon']:
            return np.random.randint(self.num_bands)
        q_vals = state['Q']
        return int(np.argmax(q_vals))

    def ml_select_band(self, agent: int) -> int:
        """Selects a band for a transmitting ML agent using Thompson Sampling."""
        state = self.ml_states[agent]
        samples = np.random.beta(state['alpha'], state['beta'])
        return int(np.argmax(samples))

    def advanced_rl_select_action(self, agent: int, t: int) -> int:
        """
        Selects an action (band 0..N-1 or Idle N) for a DQN agent.

        Args:
            agent: The agent index.
            t: The current time step for time encoding.

        Returns:
            The selected action (band index or num_bands for idle).
        """
        s_dict = self.advanced_rl_states[agent]

        # Time encoding
        period = self.num_agents
        shift = agent
        time_enc = [
            np.sin(2 * np.pi * (t + shift) / period),
            np.cos(2 * np.pi * (t + shift) / period)
        ]
        time_tensor = torch.tensor(time_enc, dtype=torch.float32)

        # Concatenate history and time encoding to form the full state
        history_t = [h.clone() for h in s_dict['history']]
        hist_flat = torch.cat(history_t)
        full_state = torch.cat([hist_flat, time_tensor])

        if np.random.rand() < s_dict['epsilon']:
            return np.random.randint(self.num_bands + 1)  # 0 to num_bands (idle)

        with torch.no_grad():
            q_values = s_dict['policy_net'](full_state)
        return int(q_values.argmax().item())

    # ---------------------------------------------------------------------
    # Learning and Update Methods
    # ---------------------------------------------------------------------

    def rl_update(self, agent: int, action: int, reward: float) -> None:
        """Stores the current action/reward for the next step's SARSA update."""
        state = self.rl_states[agent]
        state['last_action'] = action
        state['last_reward'] = reward

    def ml_update(self, agent: int, action: int, reward: float) -> None:
        """Updates Beta distribution priors based on success or failure."""
        state = self.ml_states[agent]
        if reward > 0:
            state['alpha'][action] += 1
        else:
            state['beta'][action] += 1

    def _train_advanced_rl(self, agent: int):
        """Performs one training step for the DQN using experience replay."""
        s_dict = self.advanced_rl_states[agent]
        if len(s_dict['replay']) < s_dict['batch_size']:
            return

        batch = random.sample(s_dict['replay'], s_dict['batch_size'])
        # Unpack batch: (state, action, reward, next_state)
        states, acts, rews, next_s = zip(*batch)
        states_t = torch.stack(states)
        acts_t = torch.tensor(acts).long()
        rews_t = torch.tensor(rews, dtype=torch.float32)
        next_s_t = torch.stack(next_s)

        # Compute Q(s_t, a)
        curr_q = s_dict['policy_net'](states_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

        # Compute max Q(s_{t+1}, a) using the target network
        with torch.no_grad():
            next_q_max = s_dict['target_net'](next_s_t).max(1)[0]
            # Compute target Q-value
            target_q = rews_t + s_dict['gamma'] * next_q_max

        # Compute loss and perform optimization step
        loss = F.mse_loss(curr_q, target_q)
        s_dict['optimizer'].zero_grad()
        loss.backward()
        s_dict['optimizer'].step()

        # Soft update of the target network
        tau = s_dict['tau']
        target_params = s_dict['target_net'].state_dict()
        policy_params = s_dict['policy_net'].state_dict()
        for key in policy_params:
            target_params[key] = policy_params[key] * tau + target_params[key] * (1 - tau)
        s_dict['target_net'].load_state_dict(target_params)

        # Decay epsilon
        s_dict['step_count'] += 1
        s_dict['epsilon'] = max(s_dict['epsilon_min'], s_dict['epsilon'] * s_dict['epsilon_decay'])

    # ---------------------------------------------------------------------
    # Observation Learning (Passive updates for all agents)
    # ---------------------------------------------------------------------

    def _observe_all_bands(self, t: int, band_outcomes: List[int]):
        """
        Updates agent states based on general band outcomes, regardless of the agent's action.

        Args:
            t: Current time step.
            band_outcomes: List where -1=empty, 0=collision, 1=success for each band.
        """
        self.observation_history.append((t, band_outcomes.copy()))

        for agent in range(self.num_agents):
            if self.algorithm == 'Reinforcement Learning':
                self._observe_rl(agent, band_outcomes)
            elif self.algorithm == 'Machine Learning':
                self._observe_ml(agent, band_outcomes)
            elif self.algorithm == 'Advanced RL':
                self._observe_advanced_rl(agent, band_outcomes)

    def _observe_rl(self, agent: int, outcomes: List[int]):
        """Weakly updates Q-values based on observed band success/failure."""
        state = self.rl_states[agent]
        obs_lr = state['alpha'] * 0.1
        for band_idx, outcome in enumerate(outcomes):
            if outcome == 1:
                reward = 1.0  # Observed success
            elif outcome == 0:
                reward = -0.3  # Observed collision
            else:
                reward = 0.1  # Observed empty (slight positive incentive)

            # Weak update: Q(b) = Q(b) + obs_lr * (reward - Q(b))
            state['Q'][band_idx] += obs_lr * (reward - state['Q'][band_idx])

    def _observe_ml(self, agent: int, outcomes: List[int]):
        """Weakly updates Thompson Sampling priors based on band observations."""
        state = self.ml_states[agent]
        obs_weight = 0.1
        for band_idx, outcome in enumerate(outcomes):
            if outcome == 1:
                state['alpha'][band_idx] += obs_weight
            elif outcome == 0:
                state['beta'][band_idx] += obs_weight
            elif outcome == -1:
                state['beta'][band_idx] += obs_weight * 0.1

    def _observe_advanced_rl(self, agent: int, outcomes: List[int]):
        """Pushes the current observation into the DQN agent's history buffer."""
        # Normalize outcomes to [0, 1]: -1 (empty) -> 0, 0 (collision) -> 0.5, 1 (success) -> 1
        obs_norm = [(o + 1) / 2.0 for o in outcomes]
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        self.advanced_rl_states[agent]['history'].append(obs_tensor)

    # ---------------------------------------------------------------------
    # Step Execution
    # ---------------------------------------------------------------------

    def get_transmissions(self, t: int) -> Tuple[List[Tuple[bool, int]], Optional[List[torch.Tensor]]]:
        """
        Determines the transmission decision (transmit/idle) and band choice for all agents.

        Returns:
            A tuple of:
            1. List of (do_transmit: bool, band_index: int or -1) for each agent.
            2. List of full state tensors (for Advanced RL only), or None.
        """
        decisions: List[Tuple[bool, int]] = []
        full_states: Optional[List[torch.Tensor]] = None
        if self.algorithm == 'Advanced RL':
            full_states = [None] * self.num_agents

        for agent_idx in range(self.num_agents):
            do_transmit = False
            band = -1

            if self.algorithm == 'Random':
                # Markov Chain: 80% persistence
                last_transmit = self.last_actions[agent_idx]
                p_transmit = 0.8 if last_transmit else 0.2
                do_transmit = np.random.rand() < p_transmit
                self.last_actions[agent_idx] = do_transmit
                band = np.random.randint(self.num_bands) if do_transmit else -1

            elif self.algorithm == 'Advanced RL':
                action = self.advanced_rl_select_action(agent_idx, t)
                do_transmit = action < self.num_bands
                band = action if do_transmit else -1
                self.last_actions[agent_idx] = do_transmit

                # Compute and store the full state used for this decision (s_t)
                state_dict = self.advanced_rl_states[agent_idx]
                period = self.num_agents
                shift = agent_idx
                time_enc = [
                    np.sin(2 * np.pi * (t + shift) / period),
                    np.cos(2 * np.pi * (t + shift) / period)
                ]
                time_tensor = torch.tensor(time_enc, dtype=torch.float32)
                history_t = [h.clone() for h in state_dict['history']]
                hist_flat = torch.cat(history_t)
                full_state = torch.cat([hist_flat, time_tensor])
                full_states[agent_idx] = full_state

            else:  # RL, ML: use Markov for transmit, algo for band selection
                last_transmit = self.last_actions[agent_idx]
                p_transmit = 0.8 if last_transmit else 0.2
                do_transmit = np.random.rand() < p_transmit
                self.last_actions[agent_idx] = do_transmit

                if do_transmit:
                    if self.algorithm == 'Reinforcement Learning':
                        band = self.rl_select_band(agent_idx)
                    elif self.algorithm == 'Machine Learning':
                        band = self.ml_select_band(agent_idx)
                # If do_transmit is False, band remains -1 (idle)

            decisions.append((do_transmit, band))

        return decisions, full_states

    def step(self, t: int) -> None:
        """
        Executes one time step of the simulation: agent decisions, conflict resolution,
        reward calculation, and algorithm updates.

        Args:
            t: The current time step index.
        """
        decisions, full_states = self.get_transmissions(t)
        band_usage: List[List[int]] = [[] for _ in range(self.num_bands)]
        # -1: empty, 0: collision, 1: success
        band_outcomes: List[int] = [-1] * self.num_bands

        transmitting_agents: dict[int, int] = {}
        num_attempts_this_step = 0

        # Collect transmissions and count attempts
        for agent_idx, (transmit, band) in enumerate(decisions):
            if transmit:
                self.attempt_counts[agent_idx] += 1
                num_attempts_this_step += 1
                if band != -1:
                    band_usage[band].append(agent_idx)
                    transmitting_agents[agent_idx] = band

        # Resolve outcomes and compute rewards for transmitting agents
        step_successful_transmissions = 0
        rewards: dict[int, float] = {}

        for band_idx, users in enumerate(band_usage):
            if len(users) == 1:
                agent = users[0]
                self.success_counts[agent] += 1
                step_successful_transmissions += 1
                band_outcomes[band_idx] = 1
                rewards[agent] = 1.0  # Success reward
            elif len(users) > 1:
                band_outcomes[band_idx] = 0
                for agent in users:
                    rewards[agent] = 0.0  # Collision reward

        # === UPDATE ALGORITHMS (On Transmitting Action) ===
        for agent in transmitting_agents:
            band = transmitting_agents[agent]
            reward = rewards[agent]
            if self.algorithm == 'Reinforcement Learning':
                # SARSA: stores (s, a, r) to be used for Q(s', a') in the next step
                self.rl_update(agent, band, reward)
            elif self.algorithm == 'Machine Learning':
                self.ml_update(agent, band, reward)

        # === DELAYED RL UPDATE (SARSA) ===
        if self.algorithm == 'Reinforcement Learning':
            for agent_idx in range(self.num_agents):
                state = self.rl_states[agent_idx]

                if state['last_action'] >= 0:
                    last_action = state['last_action']
                    last_reward = state['last_reward']

                    # The next action is the band selected in THIS step, or -1 if idle
                    next_action = decisions[agent_idx][1] if decisions[agent_idx][0] else -1

                    # Q'(s', a'): If next action is a band, use Q-value. Otherwise, 0 (idle penalty)
                    Q_prime_a_prime = state['Q'][next_action] if next_action >= 0 else 0.0

                    # SARSA Update: Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
                    td_target = last_reward + state['gamma'] * Q_prime_a_prime
                    td_error = td_target - state['Q'][last_action]
                    state['Q'][last_action] += state['alpha'] * td_error

                # Prepare state for NEXT step's update
                state['last_action'] = transmitting_agents.get(agent_idx, -1)
                state['last_reward'] = rewards.get(agent_idx, 0.0)

                # Decay epsilon and alpha
                state['epsilon'] = max(state['epsilon_min'], state['epsilon'] * state['epsilon_decay'])
                state['alpha'] = max(0.01, state['alpha'] * 0.9995)

        # Compute step throughput
        step_th = step_successful_transmissions / num_attempts_this_step if num_attempts_this_step > 0 else 0.0
        self.step_throughputs.append(step_th)

        # === OBSERVATION LEARNING FOR ALL AGENTS ===
        self._observe_all_bands(t, band_outcomes)

        # Advanced RL: Store experience and train
        if self.algorithm == 'Advanced RL' and full_states:
            period = self.num_agents
            # Observation used for next state history
            obs_norm_list = [(o + 1) / 2.0 for o in band_outcomes]
            obs_norm = torch.tensor(obs_norm_list, dtype=torch.float32)

            for agent_idx in range(self.num_agents):
                state_dict = self.advanced_rl_states[agent_idx]
                # Action is the selected band (0..N-1) or N for idle
                action = self.num_bands if not decisions[agent_idx][0] else decisions[agent_idx][1]

                # Improved Reward Structure (includes idle reward)
                if decisions[agent_idx][0]:
                    band = decisions[agent_idx][1]
                    users = band_usage[band]
                    reward = 1.0 if len(users) == 1 else -1.0
                else:
                    reward = 0.1  # Small positive reward for idling

                # Compute full next state (s_{t+1})
                shift = agent_idx
                current_history_list = list(state_dict['history'])[1:] + [obs_norm.clone()]
                next_hist_flat = torch.cat(current_history_list)
                next_time_enc = [
                    np.sin(2 * np.pi * ((t + 1) + shift) / period),
                    np.cos(2 * np.pi * ((t + 1) + shift) / period)
                ]
                next_time_tensor = torch.tensor(next_time_enc, dtype=torch.float32)
                full_next_state = torch.cat([next_hist_flat, next_time_tensor])

                # Store experience: (s_t, a_t, r_t, s_{t+1})
                state_dict['replay'].append((full_states[agent_idx], action, reward, full_next_state))

                # Train the network
                self._train_advanced_rl(agent_idx)

        # Fill grid text for UI visualization
        for band_idx in range(self.num_bands):
            users = band_usage[band_idx]
            if len(users) == 0:
                text = ""
            elif len(users) == 1:
                text = f"{users[0] + 1}"
            else:
                text = ",".join(str(a + 1) for a in sorted(users))
                if len(text) > 12:
                    text = text[:12] + "..."
            self.grid_texts[band_idx][t] = text

    # ---------------------------------------------------------------------
    # Output and Metrics
    # ---------------------------------------------------------------------

    def print_grid(self) -> None:
        """Prints the transmission grid to the console."""
        print("\nTransmission Grid (Y: Bands, X: Time Steps)")
        line_width = 13 + self.num_time_steps * 14
        line = "-" * line_width
        print(line)
        header = "Band \\ Time | " + " | ".join([f"t={i + 1:2d}" for i in range(self.num_time_steps)])
        print(header)
        print(line)
        for band_idx in range(self.num_bands):
            row = f"B{band_idx + 1:8} | " + " | ".join([f"{self.grid_texts[band_idx][t]:<12}" for t in range(self.num_time_steps)])
            print(row)
        print(line)

    def compute_metrics(self) -> Tuple[float, float, List[float]]:
        """Calculates overall throughput, fairness, and per-agent throughputs."""
        total_successes = sum(self.success_counts)
        total_attempts = sum(self.attempt_counts)
        overall_throughput = total_successes / total_attempts if total_attempts > 0 else 0.0

        per_agent_throughput = [
            self.success_counts[a] / self.attempt_counts[a] if self.attempt_counts[a] > 0 else 0.0
            for a in range(self.num_agents)
        ]

        fairness = jains_fairness(per_agent_throughput)

        return overall_throughput, fairness, per_agent_throughput

    def run_headless(self) -> Tuple[float, float, List[float]]:
        """Runs the entire simulation without UI and returns final metrics."""
        for t in range(self.num_time_steps):
            self.step(t)
        self.print_grid()
        return self.compute_metrics()