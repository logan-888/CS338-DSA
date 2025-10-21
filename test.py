import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np


class QLearner:
    def __init__(self, num_actions):
        self.Q = np.zeros(num_actions)
        self.alpha = 0.1
        self.epsilon = 0.1
        self.num_actions = num_actions

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])


class UCBLearner:
    def __init__(self, num_actions):
        self.counts = np.zeros(num_actions)
        self.values = np.zeros(num_actions)
        self.num_actions = num_actions

    def choose_action(self):
        total = max(np.sum(self.counts), 1)
        ucb = self.values + np.sqrt(2 * np.log(total) / (self.counts + 1e-10))
        return np.argmax(ucb)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] = ((n - 1) * self.values[action] + reward) / n


class ParamWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wireless Channel Allocation Simulation")
        self.geometry("400x450")
        self.configure(bg='#f0f0f0')

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Title
        title_label = tk.Label(self, text="Dynamic Wireless Channel Allocation Simulation", font=("Arial", 16, "bold"),
                               bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)

        # Description
        desc_label = tk.Label(self,
                              text="Configure the simulation parameters below. Each agent has a configurable number of packets to transmit across available bands.",
                              bg='#f0f0f0', fg='#666', wraplength=350, justify='center')
        desc_label.pack(pady=5)

        # Variables
        self.num_agents_var = tk.IntVar(value=3)
        self.num_bands_var = tk.IntVar(value=2)
        self.num_steps_var = tk.IntVar(value=20)
        self.packets_var = tk.IntVar(value=10)
        self.alg_var = tk.StringVar(value='Random')
        self.vis_var = tk.BooleanVar(value=True)

        # Frames for inputs
        input_frame = tk.LabelFrame(self, text="Parameters", font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#333',
                                    padx=10, pady=10)
        input_frame.pack(pady=10, padx=20, fill='x')

        tk.Label(input_frame, text="Number of Agents (1-10):", bg='#f0f0f0', font=("Arial", 10)).grid(row=0, column=0,
                                                                                                      sticky='w',
                                                                                                      padx=5, pady=5)
        agent_entry = tk.Entry(input_frame, textvariable=self.num_agents_var, width=10, font=("Arial", 10))
        agent_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Label(input_frame, text="Number of Bands (1-10):", bg='#f0f0f0', font=("Arial", 10)).grid(row=1, column=0,
                                                                                                     sticky='w', padx=5,
                                                                                                     pady=5)
        band_entry = tk.Entry(input_frame, textvariable=self.num_bands_var, width=10, font=("Arial", 10))
        band_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Label(input_frame, text="Number of Time Steps:", bg='#f0f0f0', font=("Arial", 10)).grid(row=2, column=0,
                                                                                                   sticky='w', padx=5,
                                                                                                   pady=5)
        steps_entry = tk.Entry(input_frame, textvariable=self.num_steps_var, width=10, font=("Arial", 10))
        steps_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        tk.Label(input_frame, text="Packets per Agent (1-50):", bg='#f0f0f0', font=("Arial", 10)).grid(row=3, column=0,
                                                                                                       sticky='w',
                                                                                                       padx=5, pady=5)
        packets_entry = tk.Entry(input_frame, textvariable=self.packets_var, width=10, font=("Arial", 10))
        packets_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        # Algorithm selection
        alg_frame = tk.Frame(input_frame, bg='#f0f0f0')
        alg_frame.grid(row=4, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        tk.Label(alg_frame, text="Algorithm:", bg='#f0f0f0', font=("Arial", 10)).pack(side='left')
        alg_combo = ttk.Combobox(alg_frame, textvariable=self.alg_var, values=['Random', 'Reinforcement Learning', 'Machine Learning'],
                                 state='readonly', width=20)
        alg_combo.pack(side='left', padx=(5, 0))

        # Visualization checkbox
        vis_check = tk.Checkbutton(input_frame, text="Enable Visualization (Grid Display)", variable=self.vis_var,
                                   bg='#f0f0f0', selectcolor='#e3f2fd', font=("Arial", 10))
        vis_check.grid(row=5, column=0, columnspan=2, sticky='w', padx=5, pady=10)

        # Start button
        start_btn = tk.Button(self, text="Start Simulation", command=self.start_sim,
                              bg='#4CAF50', fg='white', font=("Arial", 12, "bold"), relief='raised', bd=2,
                              cursor='hand2')
        start_btn.pack(pady=20)

    def start_sim(self):
        try:
            na = self.num_agents_var.get()
            nb = self.num_bands_var.get()
            ns = self.num_steps_var.get()
            pp = self.packets_var.get()
            alg = self.alg_var.get()
            vis = self.vis_var.get()

            if na < 1 or na > 10 or nb < 1 or nb > 10 or ns < 1 or pp < 1 or pp > 50:
                raise ValueError("Parameters out of range. Agents/Bands: 1-10, Packets: 1-50, Steps: >=1")

            total_packets = pp * na
            total_slots = ns * nb
            if total_packets > total_slots:
                raise ValueError(
                    "Validation failed: total slots must be sufficient for all packets (time_steps * num_bands >= packets_per_agent * num_agents)")

            self.destroy()
            sim = WirelessSimulation(na, nb, ns, alg, packets_per_agent=pp)
            if vis:
                sim.visualize()
            else:
                sim.run_headless()
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))


class WirelessSimulation:
    def __init__(self, num_agents, num_bands, num_time_steps, algorithm='Random', packets_per_agent=10):
        self.num_agents = num_agents
        self.num_bands = num_bands
        self.num_time_steps = num_time_steps
        self.algorithm = algorithm
        self.packets_per_agent = packets_per_agent

        # State: pending packets for each agent (list of packet numbers 1-pp)
        self.pending_packets = [list(range(1, packets_per_agent + 1)) for _ in range(num_agents)]

        # Grid texts for terminal print
        self.grid_texts = [['' for _ in range(num_time_steps)] for _ in range(num_bands)]

        # Progress tracking
        self.progress = 0

        # Learners for RL and ML
        self.agents_learners = [None] * num_agents
        if algorithm == 'Reinforcement Learning':
            for a in range(num_agents):
                self.agents_learners[a] = QLearner(num_bands + 1)  # actions 0..num_bands-1: bands, num_bands: idle
        elif algorithm == 'Machine Learning':
            for a in range(num_agents):
                self.agents_learners[a] = UCBLearner(num_bands + 1)

    def get_transmissions(self, t):
        """
        Get transmission decisions for each agent at time t.
        Returns list of (transmit: bool, band: int or None, packet: int or None)
        """
        decisions = []
        for a in range(self.num_agents):
            if len(self.pending_packets[a]) == 0:
                decisions.append((False, None, None))
                continue

            if self.algorithm == 'Random':
                # Randomly decide to transmit or idle
                transmit = np.random.choice([True, False], p=[0.7, 0.3])  # Bias towards transmitting
                if transmit:
                    band = np.random.randint(0, self.num_bands)
                else:
                    band = None
                packet = self.pending_packets[a][0] if transmit else None
                decisions.append((transmit, band, packet))
            elif self.algorithm in ['Reinforcement Learning', 'Machine Learning']:
                learner = self.agents_learners[a]
                if self.algorithm == 'Reinforcement Learning':
                    action = learner.choose_action()
                else:
                    action = learner.choose_action()
                if action == self.num_bands:
                    transmit = False
                    band = None
                    packet = None
                else:
                    transmit = True
                    band = action
                    packet = self.pending_packets[a][0]
                decisions.append((transmit, band, packet))
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return decisions

    def step(self, t):
        """
        Perform one time step.
        """
        decisions = self.get_transmissions(t)
        band_usage = [[] for _ in range(self.num_bands)]  # list of (agent, packet)

        for a, (transmit, band, packet) in enumerate(decisions):
            if transmit and band is not None and packet is not None:
                band_usage[band].append((a, packet))

        # Process outcomes
        for b in range(self.num_bands):
            users = band_usage[b]
            if len(users) == 1:
                agent, pack = users[0]
                # Successful transmission
                del self.pending_packets[agent][0]
                if self.algorithm in ['Reinforcement Learning', 'Machine Learning']:
                    learner = self.agents_learners[agent]
                    action = b
                    reward = 1
                    learner.update(action, reward)
            elif len(users) > 1:
                # Collision
                for agent, pack in users:
                    if self.algorithm in ['Reinforcement Learning', 'Machine Learning']:
                        learner = self.agents_learners[agent]
                        action = b
                        reward = 0
                        learner.update(action, reward)

        # Process idles
        for a, (transmit, band, packet) in enumerate(decisions):
            if not transmit and len(self.pending_packets[a]) > 0:  # Only update if had packets and chose idle
                if self.algorithm in ['Reinforcement Learning', 'Machine Learning']:
                    learner = self.agents_learners[a]
                    action = self.num_bands
                    reward = 0
                    learner.update(action, reward)

        # Update grid text and labels if in visualize mode
        for b in range(self.num_bands):
            users = band_usage[b]
            text = ""
            if len(users) == 0:
                text = ""
            elif len(users) == 1:
                agent, pack = users[0]
                text = f"{agent + 1}.{pack}"
            else:
                # Collision
                user_texts = [f"{a + 1}.{p}" for a, p in sorted(users, key=lambda x: x[0])]
                text = ",".join(user_texts)
                if len(text) > 12:  # Truncate if too long
                    text = text[:12] + "..."

            self.grid_texts[b][t] = text

            # Update label if exists
            if hasattr(self, 'labels'):
                label = self.labels[b][t]
                if len(users) == 0:
                    label.config(text="", bg="white")
                elif len(users) == 1:
                    label.config(text=text, bg="#4CAF50", fg="white")
                else:
                    label.config(text=text, bg="#F44336", fg="white")

    def print_grid(self):
        """
        Print the grid neatly in the terminal.
        """
        print("\nTransmission Grid (Y: Bands, X: Time Steps)")
        print("-" * (13 + self.num_time_steps * 14))
        header = "Band \\ Time | " + " | ".join([f"t={i + 1:2d}" for i in range(self.num_time_steps)])
        print(header)
        print("-" * (13 + self.num_time_steps * 14))
        for b in range(self.num_bands):
            row = f"B{b + 1:8} | " + " | ".join([f"{self.grid_texts[b][t]:<12}" for t in range(self.num_time_steps)])
            print(row)
        print("-" * (13 + self.num_time_steps * 14))

    def compute_metrics(self):
        """
        Compute performance metrics.
        """
        agent_sent = np.array([self.packets_per_agent - len(p) for p in self.pending_packets])
        total_sent = np.sum(agent_sent)
        total_slots = self.num_time_steps * self.num_bands
        overall_throughput = total_sent / total_slots if total_slots > 0 else 0
        per_agent_throughput = agent_sent / total_slots if total_slots > 0 else np.zeros(self.num_agents)

        sum_th = np.sum(agent_sent)
        sum_sq = np.sum(agent_sent ** 2)
        fairness = (sum_th ** 2) / (self.num_agents * sum_sq) if sum_sq > 0 else 0

        unsent = self.packets_per_agent * self.num_agents - total_sent

        return overall_throughput, fairness, unsent, per_agent_throughput

    def run_headless(self):
        """
        Run simulation without visualization and display metrics.
        """
        self.current_step = 0
        for t in range(self.num_time_steps):
            self.step(t)
            self.current_step += 1

        self.print_grid()

        overall_throughput, fairness, unsent, per_agent_throughput = self.compute_metrics()

        root = tk.Tk()
        root.title(f"Simulation Metrics - {self.algorithm}")
        root.geometry("500x400")
        root.configure(bg='#f8f9fa')

        title = tk.Label(root, text="Performance Metrics", font=("Arial", 14, "bold"), bg='#f8f9fa')
        title.pack(pady=10)

        text_frame = tk.Frame(root, bg='#f8f9fa')
        text_frame.pack(fill='both', expand=True, padx=20, pady=10)

        metrics_text = tk.Text(text_frame, wrap='word', font=("Courier", 11), bg='white', relief='flat', padx=10,
                               pady=10)
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=metrics_text.yview)
        metrics_text.config(yscrollcommand=scrollbar.set)

        metrics_content = f"SIMULATION COMPLETE!\n\nOverall Throughput: {overall_throughput:.3f}\nJain's Fairness Index: {fairness:.3f}\nTotal Unsent Packets: {unsent}\n\nThroughput Per Agent:\n"
        for a in range(self.num_agents):
            metrics_content += f"Agent {a + 1}: {per_agent_throughput[a]:.3f}\n"

        metrics_text.insert(1.0, metrics_content)
        metrics_text.config(state='disabled')

        metrics_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        root.mainloop()

    def visualize(self):
        """
        Launch the Tkinter visualization and run the simulation dynamically.
        """
        self.root = tk.Tk()
        self.root.title(f"Dynamic Wireless Channel Allocation Simulation - {self.algorithm}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')

        # Main frame
        main_frame = tk.Frame(self.root, bg='#f8f9fa')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title_frame = tk.Frame(main_frame, bg='#f8f9fa')
        title_frame.pack(fill='x', pady=(0, 10))
        tk.Label(title_frame,
                 text=f"Simulation: {self.algorithm} | Agents: {self.num_agents} | Bands: {self.num_bands} | Time Steps: {self.num_time_steps} | Packets/Agent: {self.packets_per_agent}",
                 font=("Arial", 16, "bold"), bg='#f8f9fa', fg='#333').pack()

        # Grid frame (no scrolling, no expand)
        grid_frame = tk.LabelFrame(main_frame, text="Transmission Grid (Bands Y / Time X)", font=("Arial", 12, "bold"),
                                   bg='#f8f9fa', fg='#333', padx=5, pady=5)
        grid_frame.pack(fill='x', pady=(0, 10))

        # Header
        tk.Label(grid_frame, text="Bands (Y) \\ Time Steps (X)", font=("Arial", 12, "bold"), bg='#f8f9fa',
                 fg='#333').grid(row=0, column=0, columnspan=self.num_time_steps + 1, pady=5, sticky='ew')

        # Band labels on left
        for b in range(self.num_bands):
            tk.Label(grid_frame, text=f"B{b + 1}", width=10, relief="solid", borderwidth=1, bg='#2196F3', fg='white',
                     font=("Arial", 10, "bold")).grid(row=b + 1, column=0, sticky='nsew', padx=1)

        # Time labels on top, starting from t=1
        time_labels = list(range(1, self.num_time_steps + 1))
        for idx, t_val in enumerate(time_labels):
            tk.Label(grid_frame, text=f"t={t_val}", width=10, relief="solid", borderwidth=1, bg='#FF9800', fg='white',
                     font=("Arial", 10, "bold")).grid(row=0, column=idx + 1, sticky='nsew', padx=1)

        # Create grid labels
        self.labels = [[None for _ in range(self.num_time_steps)] for _ in range(self.num_bands)]
        for b in range(self.num_bands):
            for t_idx in range(self.num_time_steps):
                label = tk.Label(grid_frame, text="", width=10, height=2, relief="solid", borderwidth=1,
                                 font=("Arial", 9), justify='center', bg='white', fg='#333')
                label.grid(row=b + 1, column=t_idx + 1, sticky='nsew', padx=1, pady=1)
                self.labels[b][t_idx] = label

        # Configure grid weights
        for col in range(self.num_time_steps + 1):
            grid_frame.grid_columnconfigure(col, weight=1)
        for row in range(self.num_bands + 1):
            grid_frame.grid_rowconfigure(row, weight=1)

        # Progress bar
        self.progress_frame = tk.Frame(main_frame, bg='#f8f9fa')
        self.progress_frame.pack(fill='x', pady=(0, 10))
        tk.Label(self.progress_frame, text="Progress:", font=("Arial", 10, "bold"), bg='#f8f9fa', fg='#333').pack(
            side='left')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                            maximum=self.num_time_steps, length=300)
        self.progress_bar.pack(side='left', padx=10)
        self.step_label = tk.Label(self.progress_frame, text=f"Step 0/{self.num_time_steps}", bg='#f8f9fa',
                                   font=("Arial", 10), fg='#333')
        self.step_label.pack(side='left')

        # Metrics display area
        metrics_frame = tk.LabelFrame(main_frame, text="Performance Metrics", font=("Arial", 12, "bold"), bg='#f8f9fa',
                                      fg='#333', padx=10, pady=10)
        metrics_frame.pack(fill='x', pady=(0, 10))

        self.metrics_text = tk.Text(metrics_frame, height=8, width=80, font=("Courier", 10), state='disabled',
                                    bg='white', relief='flat')
        scrollbar_metrics = tk.Scrollbar(metrics_frame, orient='vertical', command=self.metrics_text.yview)
        self.metrics_text.config(yscrollcommand=scrollbar_metrics.set)
        self.metrics_text.pack(side='left', fill='both', expand=True)
        scrollbar_metrics.pack(side='right', fill='y')

        # Start the simulation
        self.current_step = 0
        self.restart_btn = None
        self.update_metrics()
        self.update_step()

        self.root.mainloop()

    def update_metrics(self):
        """
        Update metrics display.
        """
        overall_throughput, fairness, unsent, per_agent = self.compute_metrics()
        if self.current_step == 0:
            title = "Simulation Starting..."
        else:
            title = f"After Step {self.current_step}"
        metrics_content = f"{title}:\n\nThroughput (overall): {overall_throughput:.3f}\nFairness Index: {fairness:.3f}\nUnsent Packets: {unsent}\n\nPer-Agent Throughput:\n"
        for a in range(self.num_agents):
            sent = self.packets_per_agent - len(self.pending_packets[a])
            metrics_content += f"Agent {a + 1}: {sent / (self.num_time_steps * self.num_bands):.3f}\n"
        self.metrics_text.config(state='normal')
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_content)
        self.metrics_text.config(state='disabled')

    def update_step(self):
        """
        Update one step and schedule the next.
        """
        if self.current_step < self.num_time_steps:
            self.step(self.current_step)
            self.progress_var.set(self.current_step + 1)
            self.step_label.config(text=f"Step {self.current_step + 1}/{self.num_time_steps}")
            self.update_metrics()
            self.current_step += 1
            delay = 200  # ms between steps
            self.root.after(delay, self.update_step)
        else:
            # Simulation complete
            self.update_metrics()
            self.print_grid()
            self.step_label.config(text="Simulation Complete")
            if self.restart_btn is None:
                self.restart_btn = tk.Button(self.progress_frame, text="Restart Simulation", command=self.restart,
                                             bg='#4CAF50', fg='white', font=("Arial", 10, "bold"), relief='raised',
                                             bd=2, cursor='hand2')
                self.restart_btn.pack(side='left', padx=10)

    def restart(self):
        """
        Restart the simulation by returning to parameter menu.
        """
        self.root.destroy()
        ParamWindow().mainloop()


# Example usage
if __name__ == "__main__":
    ParamWindow().mainloop()