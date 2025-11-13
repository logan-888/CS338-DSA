#!/usr/bin/env python3
"""
main.py
-------
Core simulation engine with:
- Markov chain for transmit/idle decisions (80% persistence)
- Observation learning from band outcomes (success/collision/empty, anonymized)
- Reinforcement Learning (tabular Q-learning with replay)
- Machine Learning (Thompson Sampling)
- Advanced RL (DQN with torch, state from observations + time encoding for rotation)
- Spiking Neural Network (LIF neurons with torch for band selection)
- Throughput as successful / attempted transmissions
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Double-down
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import collections
import random
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def jains_fairness(throughputs):
    if not throughputs:
        return 1.0
    n = len(throughputs)
    sum_x = sum(throughputs)
    sum_x2 = sum(x * x for x in throughputs)
    if sum_x == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


class WirelessSimulation:
    def __init__(self, num_agents: int, num_bands: int, num_time_steps: int, algorithm: str = 'Random'):
        # Input validation
        if not (1 <= num_agents <= 10 and 1 <= num_bands <= 10 and num_time_steps >= 1):
            raise ValueError("Agents/Bands: 1-10, Steps: >=1")

        self.num_agents = num_agents
        self.num_bands = num_bands
        self.num_time_steps = num_time_steps
        self.algorithm = algorithm

        # Success and attempt counters
        self.success_counts = [0] * num_agents
        self.attempt_counts = [0] * num_agents

        # Per-step throughputs for graphing
        self.step_throughputs = []

        # Markov chain state: last action for each agent (True: transmitted, False: idled)
        self.last_actions = [True] * num_agents  # Start eager to transmit

        # Text grid for terminal output
        self.grid_texts = [['' for _ in range(num_time_steps)] for _ in range(num_bands)]

        # Algorithm states
        self.rl_states = None
        self.ml_states = None
        self.advanced_rl_states = None
        self.snn_states = None
        if algorithm == 'Reinforcement Learning':
            self._init_rl()
        elif algorithm == 'Machine Learning':
            self._init_ml()
        elif algorithm == 'Advanced RL':
            self._init_advanced_rl()
        elif algorithm == 'Spiking Neural Network':
            self._init_snn()
        elif algorithm != 'Random':
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Observation history (for idle learning)
        self.observation_history = []  # List of (t, band_outcomes)

    # --------------------------------------------------------------------- #
    # Basic RL Initialization
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
                'step_count': 0
            }
            self.rl_states.append(state)

    # --------------------------------------------------------------------- #
    # Basic ML (Thompson) Initialization
    # --------------------------------------------------------------------- #
    def _init_ml(self):
        self.ml_states = []
        for _ in range(self.num_agents):
            state = {
                'alpha': np.full(self.num_bands, 2.0),  # Prior: expect ~0.67 success
                'beta': np.full(self.num_bands, 1.0)
            }
            self.ml_states.append(state)

    # --------------------------------------------------------------------- #
    # Advanced RL Initialization
    # --------------------------------------------------------------------- #
    def _init_advanced_rl(self):
        class QNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)

        self.advanced_rl_states = []
        input_size = self.num_bands * 3 + 2  # 3-step history of outcomes + sin/cos time
        output_size = self.num_bands + 1  # bands + idle
        hidden_size = 64
        init_obs = [0.0] * self.num_bands  # Normalized empty
        init_obs_tensor = torch.tensor(init_obs, dtype=torch.float32)
        for agent in range(self.num_agents):
            policy_net = QNet(input_size, hidden_size, output_size)
            target_net = QNet(input_size, hidden_size, output_size)
            target_net.load_state_dict(policy_net.state_dict())
            optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
            replay_buffer = collections.deque(maxlen=10000)
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
                'epsilon_decay': 0.995,
                'gamma': 0.99,
                'batch_size': 32,
                'step_count': 0
            }
            self.advanced_rl_states.append(state)

    # --------------------------------------------------------------------- #
    # SNN Initialization
    # --------------------------------------------------------------------- #
    def _init_snn(self):
        self.snn_states = []
        for _ in range(self.num_agents):
            state = {
                'V': torch.zeros(self.num_bands, dtype=torch.float32),
                'I': torch.full((self.num_bands,), 0.5, dtype=torch.float32),
                'avg_obs_reward': torch.zeros(self.num_bands, dtype=torch.float32),
                'tau': 5.0,
                'dt': 1.0,
                'thresh': 1.0,
                'T_sim': 30,
                'noise_std': 0.05
            }
            self.snn_states.append(state)

    # --------------------------------------------------------------------- #
    # Band Selection Methods
    # --------------------------------------------------------------------- #
    def rl_select_band(self, agent: int) -> int:
        state = self.rl_states[agent]
        if np.random.rand() < state['epsilon']:
            return np.random.randint(self.num_bands)
        q_vals = state['Q']
        return int(np.argmax(q_vals))

    def ml_select_band(self, agent: int) -> int:
        state = self.ml_states[agent]
        samples = np.random.beta(state['alpha'], state['beta'])
        return int(np.argmax(samples))

    def advanced_rl_select_action(self, agent: int, t: int) -> int:
        s_dict = self.advanced_rl_states[agent]
        # Starvation check
        att = self.attempt_counts[agent]
        succ = self.success_counts[agent]
        th = succ / max(1, att)
        if th < 0.1 and np.random.rand() < 0.5:
            return np.random.randint(self.num_bands + 1)
        period = self.num_agents
        shift = agent
        time_enc = [
            np.sin(2 * np.pi * (t + shift) / period),
            np.cos(2 * np.pi * (t + shift) / period)
        ]
        time_tensor = torch.tensor(time_enc, dtype=torch.float32)
        history_t = [h.clone() for h in s_dict['history']]
        hist_flat = torch.cat(history_t)
        full_s = torch.cat([hist_flat, time_tensor])

        if np.random.rand() < s_dict['epsilon']:
            return np.random.randint(self.num_bands + 1)  # 0 to num_bands (idle)

        with torch.no_grad():
            q_values = s_dict['policy_net'](full_s)
        return int(q_values.argmax().item())

    def snn_select_band(self, agent: int) -> int:
        s_dict = self.snn_states[agent]
        # Starvation check
        att = self.attempt_counts[agent]
        succ = self.success_counts[agent]
        th = succ / max(1, att)
        if th < 0.1 and np.random.rand() < 0.4:
            return np.random.randint(self.num_bands)
        # Blend I with average observed rewards (history influence)
        target = s_dict['avg_obs_reward']
        blend_lr = 0.02
        s_dict['I'] += blend_lr * (target - s_dict['I'])
        s_dict['I'] = torch.clamp(s_dict['I'], 0, 1)
        V = s_dict['V'].clone()
        spikes = torch.zeros(self.num_bands, dtype=torch.float32)
        for _ in range(s_dict['T_sim']):
            I_eff = s_dict['I'] + torch.randn(self.num_bands, dtype=torch.float32) * s_dict['noise_std']
            dV = (-V + I_eff) / s_dict['tau'] * s_dict['dt']
            V += dV
            spike = (V >= s_dict['thresh']).float()
            spikes += spike
            V = V * (1 - spike)  # reset to 0
        best_band = spikes.argmax().item()
        s_dict['V'] = V
        return best_band

    # --------------------------------------------------------------------- #
    # Basic RL Updates
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

        self._store_experience(agent, action, reward)
        self._train_replay(agent)

        # Epsilon decay
        state['epsilon'] = max(state['epsilon_min'], state['epsilon'] * state['epsilon_decay'])

    # --------------------------------------------------------------------- #
    # Advanced RL Training
    # --------------------------------------------------------------------- #
    def _train_advanced_rl(self, agent: int):
        s_dict = self.advanced_rl_states[agent]
        if len(s_dict['replay']) < s_dict['batch_size']:
            return

        batch = random.sample(s_dict['replay'], s_dict['batch_size'])
        states, acts, rews, next_s = zip(*batch)
        states_t = torch.stack(states)
        acts_t = torch.tensor(acts)
        rews_t = torch.tensor(rews, dtype=torch.float32)
        next_s_t = torch.stack(next_s)
        curr_q = s_dict['policy_net'](states_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_max = s_dict['target_net'](next_s_t).max(1)[0]
            target_q = rews_t + s_dict['gamma'] * next_q_max
        loss = F.mse_loss(curr_q, target_q)
        s_dict['optimizer'].zero_grad()
        loss.backward()
        s_dict['optimizer'].step()
        if s_dict['step_count'] % 100 == 0:
            s_dict['target_net'].load_state_dict(s_dict['policy_net'].state_dict())
        s_dict['step_count'] += 1
        s_dict['epsilon'] = max(s_dict['epsilon_min'], s_dict['epsilon'] * s_dict['epsilon_decay'])

    # --------------------------------------------------------------------- #
    # SNN Updates
    # --------------------------------------------------------------------- #
    def snn_update(self, agent: int, action: int, reward: float):
        s_dict = self.snn_states[agent]
        lr = 0.02
        if reward > 0:
            s_dict['I'][action] = torch.clamp(s_dict['I'][action] + lr * (1.0 - s_dict['I'][action]), 0, 1)
        else:
            s_dict['I'][action] = torch.clamp(s_dict['I'][action] - lr * s_dict['I'][action], 0, 1)

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
            elif self.algorithm == 'Advanced RL':
                self._observe_advanced_rl(agent, band_outcomes)
            elif self.algorithm == 'Spiking Neural Network':
                self._observe_snn(agent, band_outcomes)

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

    def _observe_advanced_rl(self, agent: int, outcomes: List[int]):
        obs_norm = [(o + 1) / 2.0 for o in outcomes]
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        self.advanced_rl_states[agent]['history'].append(obs_tensor)

    def _observe_snn(self, agent: int, outcomes: List[int]):
        s_dict = self.snn_states[agent]
        lr_obs = 0.01
        decay = 0.99
        for b, o in enumerate(outcomes):
            if o == 1:
                obs_r = 1.0
                s_dict['I'][b] += lr_obs * (1.0 - s_dict['I'][b])
            elif o == 0:
                obs_r = 0.0
                s_dict['I'][b] -= lr_obs * s_dict['I'][b]
            else:
                obs_r = 0.5
                s_dict['I'][b] += lr_obs * (0.5 - s_dict['I'][b])
            # Moving average for history
            s_dict['avg_obs_reward'][b] = decay * s_dict['avg_obs_reward'][b] + (1 - decay) * obs_r

    # --------------------------------------------------------------------- #
    # Decision making with Markov chain
    # --------------------------------------------------------------------- #
    def get_transmissions(self, t: int) -> Tuple[List[Tuple[bool, int]], Optional[List[torch.Tensor]]]:
        decisions = []
        full_states: Optional[List[torch.Tensor]] = None
        if self.algorithm == 'Advanced RL':
            full_states = [None] * self.num_agents

        for a in range(self.num_agents):
            if self.algorithm == 'Random':
                last_transmit = self.last_actions[a]
                p_transmit = 0.8 if last_transmit else 0.2
                do_transmit = np.random.rand() < p_transmit
                self.last_actions[a] = do_transmit
                band = np.random.randint(self.num_bands) if do_transmit else -1
            elif self.algorithm == 'Advanced RL':
                action = self.advanced_rl_select_action(a, t)
                do_transmit = action < self.num_bands
                band = action if do_transmit else -1
                # Compute full_s for this agent
                s_dict = self.advanced_rl_states[a]
                period = self.num_agents
                shift = a
                time_enc = [
                    np.sin(2 * np.pi * (t + shift) / period),
                    np.cos(2 * np.pi * (t + shift) / period)
                ]
                time_tensor = torch.tensor(time_enc, dtype=torch.float32)
                history_t = [h.clone() for h in s_dict['history']]
                hist_flat = torch.cat(history_t)
                full_s = torch.cat([hist_flat, time_tensor])
                full_states[a] = full_s
            else:  # RL, ML, SNN: use Markov for transmit, algo for band
                last_transmit = self.last_actions[a]
                p_transmit = 0.8 if last_transmit else 0.2
                do_transmit = np.random.rand() < p_transmit
                self.last_actions[a] = do_transmit
                if not do_transmit:
                    band = -1
                else:
                    if self.algorithm == 'Reinforcement Learning':
                        band = self.rl_select_band(a)
                    elif self.algorithm == 'Machine Learning':
                        band = self.ml_select_band(a)
                    elif self.algorithm == 'Spiking Neural Network':
                        band = self.snn_select_band(a)
            decisions.append((do_transmit, band))

        if self.algorithm == 'Advanced RL':
            return decisions, full_states
        return decisions, None

    # --------------------------------------------------------------------- #
    # One simulation step
    # --------------------------------------------------------------------- #
    def step(self, t: int) -> None:
        decisions, full_states = self.get_transmissions(t)
        band_usage = [[] for _ in range(self.num_bands)]
        band_outcomes = [-1] * self.num_bands  # -1: empty, 0: collision, 1: success

        # Collect transmissions and count attempts
        num_attempts_this_step = 0
        for a, (transmit, band) in enumerate(decisions):
            if transmit:
                self.attempt_counts[a] += 1
                num_attempts_this_step += 1
            if transmit and band != -1:
                band_usage[band].append(a)

        # Resolve outcomes
        step_succ = 0
        for b, users in enumerate(band_usage):
            if len(users) == 1:
                agent = users[0]
                self.success_counts[agent] += 1
                step_succ += 1
                band_outcomes[b] = 1
                reward = 1.0
                if self.algorithm == 'Reinforcement Learning':
                    self.rl_update(agent, b, reward)
                elif self.algorithm == 'Machine Learning':
                    self.ml_update(agent, b, reward)
                elif self.algorithm == 'Spiking Neural Network':
                    self.snn_update(agent, b, reward)
            elif len(users) > 1:
                band_outcomes[b] = 0
                reward = 0.0
                for agent in users:
                    if self.algorithm == 'Reinforcement Learning':
                        self.rl_update(agent, b, reward)
                    elif self.algorithm == 'Machine Learning':
                        self.ml_update(agent, b, reward)
                    elif self.algorithm == 'Spiking Neural Network':
                        self.snn_update(agent, b, reward)

        # Compute step throughput
        step_th = step_succ / num_attempts_this_step if num_attempts_this_step > 0 else 0.0
        self.step_throughputs.append(step_th)

        # === OBSERVATION LEARNING FOR ALL AGENTS ===
        self._observe_all_bands(t, band_outcomes)

        # Advanced RL: Update all agents (including idlers)
        if self.algorithm == 'Advanced RL':
            period = self.num_agents
            obs_norm_list = [(o + 1) / 2.0 for o in band_outcomes]
            obs_norm = torch.tensor(obs_norm_list, dtype=torch.float32)
            for a in range(self.num_agents):
                s_dict = self.advanced_rl_states[a]
                action = self.num_bands if not decisions[a][0] else decisions[a][1]
                # Reward
                if decisions[a][0]:
                    b = decisions[a][1]
                    users = band_usage[b]
                    reward = 1.0 if len(users) == 1 else 0.0
                else:
                    reward = 0.0
                # Next full state
                shift = a
                current_history = s_dict['history']
                next_history_list = list(current_history)[1:] + [obs_norm.clone()]
                next_hist_flat = torch.cat(next_history_list)
                next_time_enc = [
                    np.sin(2 * np.pi * ((t + 1) + shift) / period),
                    np.cos(2 * np.pi * ((t + 1) + shift) / period)
                ]
                next_time_tensor = torch.tensor(next_time_enc, dtype=torch.float32)
                full_next_s = torch.cat([next_hist_flat, next_time_tensor])
                # Store experience
                s_dict['replay'].append((full_states[a], action, reward, full_next_s))
                # Train
                self._train_advanced_rl(a)

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
        total_succ = sum(self.success_counts)
        total_att = sum(self.attempt_counts)
        overall = total_succ / total_att if total_att > 0 else 0.0
        per_agent = [self.success_counts[a] / self.attempt_counts[a] if self.attempt_counts[a] > 0 else 0.0 for a in range(self.num_agents)]

        fairness = jains_fairness(per_agent)

        return overall, fairness, per_agent

    # --------------------------------------------------------------------- #
    # Headless execution
    # --------------------------------------------------------------------- #
    def run_headless(self) -> tuple:
        for t in range(self.num_time_steps):
            self.step(t)
        self.print_grid()
        return self.compute_metrics()