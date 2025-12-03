#!/usr/bin/env python3
"""
simulation.py
-------------
Tkinter UI code for the wireless channel allocation simulation.

Contains:
* ParamWindow: Parameter input and configuration.
* Visualiser: Live grid display, progress bar, and metrics during simulation.
* HeadlessResult: Post-simulation metric display when visualization is disabled.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import only the core engine – no UI code lives in main.py
from main import WirelessSimulation


def jains_fairness(throughputs: List[float]) -> float:
    """
    Calculates Jain's fairness index for a list of throughput values.

    The index ranges from 0 to 1, where 1 indicates perfect fairness.

    Args:
        throughputs: A list of non-negative throughput values.

    Returns:
        The Jain's fairness index. Returns 1.0 if the list is empty or all sums are zero.
    """
    if not throughputs:
        return 1.0
    n = len(throughputs)
    sum_x = sum(throughputs)
    sum_x2 = sum(x * x for x in throughputs)
    if sum_x == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


class ParamWindow(tk.Tk):
    """
    Tkinter window for setting initial simulation parameters.
    """
    def __init__(self):
        super().__init__()
        self.title("Wireless Channel Allocation Simulation")
        self.geometry("700x450")
        self.configure(bg='#f0f0f0')

        # Apply a consistent style
        style = ttk.Style()
        style.theme_use('clam')

        # Title
        tk.Label(self,
                 text="Dynamic Wireless Channel Allocation Simulation",
                 font=("Arial", 16, "bold"),
                 bg='#f0f0f0', fg='#333').pack(pady=10)

        # Description
        tk.Label(self,
                 text=("Configure the simulation parameters below. "
                       "Each agent has a configurable number of packets to transmit "
                       "across available bands."),
                 bg='#f0f0f0', fg='#666', wraplength=350, justify='center').pack(pady=5)

        # Control variables for inputs
        self.num_agents_var = tk.IntVar(value=4)
        self.num_bands_var = tk.IntVar(value=3)
        self.num_steps_var = tk.IntVar(value=10000)
        self.alg_var = tk.StringVar(value='Advanced RL')
        self.vis_var = tk.BooleanVar(value=False)

        # Input frame
        input_frame = tk.LabelFrame(self, text="Parameters", font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#333', padx=10, pady=10)
        input_frame.pack(pady=10, padx=20, fill='x')

        # ---- Number of Agents -------------------------------------------
        tk.Label(input_frame, text="Number of Agents:", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_agents_var, width=10,
                 font=("Arial", 10)).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # ---- Number of Bands --------------------------------------------
        tk.Label(input_frame, text="Number of Bands:", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_bands_var, width=10,
                 font=("Arial", 10)).grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # ---- Number of Time Steps ---------------------------------------
        tk.Label(input_frame, text="Number of Time Steps:", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_steps_var, width=10,
                 font=("Arial", 10)).grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # ---- Algorithm Selection ----------------------------------------
        alg_frame = tk.Frame(input_frame, bg='#f0f0f0')
        alg_frame.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        tk.Label(alg_frame, text="Algorithm:", bg='#f0f0f0',
                 font=("Arial", 10)).pack(side='left')
        ttk.Combobox(alg_frame, textvariable=self.alg_var,
                     values=['Random', 'Reinforcement Learning', 'Machine Learning', 'Advanced RL'],
                     state='readonly', width=20).pack(side='left', padx=(5, 0))

        # ---- Visualisation Checkbox -------------------------------------
        tk.Checkbutton(input_frame, text="Enable Visualization (Grid Display)",
                       variable=self.vis_var, bg='#f0f0f0', selectcolor='#e3f2fd',
                       font=("Arial", 10)).grid(row=4, column=0, columnspan=2,
                                                sticky='w', padx=5, pady=10)

        # ---- Start Button ------------------------------------------------
        tk.Button(self, text="Start Simulation", command=self._start_sim,
                  bg='#4CAF50', fg='white', font=("Arial", 12, "bold"),
                  relief='raised', bd=2, cursor='hand2').pack(pady=20)

    def _start_sim(self):
        """
        Validates input parameters, destroys the parameter window, and starts
        either the visual or headless simulation.
        """
        try:
            num_agents = self.num_agents_var.get()
            num_bands = self.num_bands_var.get()
            num_steps = self.num_steps_var.get()
            algorithm = self.alg_var.get()
            visualize = self.vis_var.get()

            # Input validation
            if not (num_agents >= 1 and num_bands >= 1 and num_steps >= 1):
                raise ValueError("Number of Agents, Bands, and Steps must be >= 1.")

            self.destroy()
            simulation = WirelessSimulation(num_agents, num_bands, num_steps, algorithm)
            if visualize:
                Visualiser(simulation).run()
            else:
                HeadlessResult(simulation).run()
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))


class HeadlessResult:
    """
    Displays final metrics and a performance graph after a simulation run
    without live visualization.
    """
    def __init__(self, sim: WirelessSimulation):
        self.sim = sim
        self.root: Optional[tk.Tk] = None
        self.agent_step_th: Optional[List[List[Optional[float]]]] = None
        self.agent_step_collisions: Optional[List[List[int]]] = None
        self.band_step_idle: Optional[List[List[int]]] = None
        self.restart_btn: Optional[tk.Button] = None

    def run(self):
        """
        Executes the simulation steps and displays the result window.
        """
        # Initialize data collection lists
        self.agent_step_th = [[] for _ in range(self.sim.num_agents)]
        self.agent_step_collisions = [[] for _ in range(self.sim.num_agents)]
        self.band_step_idle = [[] for _ in range(self.sim.num_bands)]

        # Run simulation steps
        for step in range(self.sim.num_time_steps):
            prev_succ = list(self.sim.success_counts)
            prev_att = list(self.sim.attempt_counts)
            self.sim.step(step)
            curr_succ = self.sim.success_counts
            curr_att = self.sim.attempt_counts

            # Collect per-agent throughput and collisions for the current step
            for agent_idx in range(self.sim.num_agents):
                successful_transmissions = curr_succ[agent_idx] - prev_succ[agent_idx]
                attempts = curr_att[agent_idx] - prev_att[agent_idx]
                throughput = successful_transmissions / attempts if attempts > 0 else None
                self.agent_step_th[agent_idx].append(throughput)
                collisions = attempts - successful_transmissions
                self.agent_step_collisions[agent_idx].append(collisions)

            # Collect per-band idle slots for the current step
            for band_idx in range(self.sim.num_bands):
                grid_text = self.sim.grid_texts[band_idx][step]
                has_users = bool(grid_text and any(c.isdigit() for c in grid_text))
                idle = 0 if has_users else 1
                self.band_step_idle[band_idx].append(idle)

        # Final metrics calculation
        total_successes = sum(self.sim.success_counts)
        total_attempts = sum(self.sim.attempt_counts)
        overall_throughput = total_successes / total_attempts if total_attempts > 0 else 0.0
        per_agent_throughput = [
            self.sim.success_counts[a] / self.sim.attempt_counts[a]
            if self.sim.attempt_counts[a] > 0 else 0.0
            for a in range(self.sim.num_agents)
        ]
        fairness = jains_fairness(per_agent_throughput)

        # Initialize result window
        root = tk.Tk()
        self.root = root
        root.title(f"Simulation Metrics - {self.sim.algorithm}")
        root.geometry("1000x700")
        root.configure(bg='#f8f9fa')

        # Header label
        tk.Label(root,
                 text=(f"Simulation: {self.sim.algorithm} | "
                       f"Agents: {self.sim.num_agents} | "
                       f"Bands: {self.sim.num_bands} | "
                       f"Steps: {self.sim.num_time_steps}"),
                 font=("Arial", 16, "bold"), bg='#f8f9fa', fg='#333') \
            .pack(fill='x', pady=(0, 10))

        tk.Label(root, text="Performance Metrics", font=("Arial", 14, "bold"),
                 bg='#f8f9fa').pack(pady=10)

        # Text frame for metrics
        text_frame = tk.Frame(root, bg='#f8f9fa')
        text_frame.pack(fill='x', padx=20, pady=10)

        metrics_text_area = tk.Text(text_frame, wrap='word', height=8, font=("Courier", 11), bg='white',
                                    relief='flat', padx=10, pady=10)
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=metrics_text_area.yview)
        metrics_text_area.config(yscrollcommand=scrollbar.set)

        content = (f"SIMULATION COMPLETE!\n\n"
                   f"Overall Throughput (Success/Attempt): {overall_throughput:.3f}\n"
                   f"Jain's Fairness Index: {fairness:.3f}\n\n"
                   f"Throughput Per Agent (Success/Attempt):\n")
        for agent_idx, throughput in enumerate(per_agent_throughput, 1):
            content += f"Agent {agent_idx}: {throughput:.3f}\n"

        metrics_text_area.insert(1.0, content)
        metrics_text_area.config(state='disabled')
        metrics_text_area.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Add the sliding window graph
        self._add_sliding_window_graph()

        # Add restart button
        self.restart_btn = tk.Button(root, text="Restart Simulation", command=self._restart,
                                     bg='#4CAF50', fg='white', font=("Arial", 10, "bold"),
                                     relief='raised', bd=2, cursor='hand2')
        self.restart_btn.pack(pady=10)

        root.mainloop()

    def _add_sliding_window_graph(self):
        """
        Creates and embeds a matplotlib graph showing sliding window averages of
        key performance metrics (throughput, collisions, idle time).
        """
        num_steps = self.sim.num_time_steps
        window_size = max(1, int(num_steps * 0.05))

        # Compute sliding averages
        agent_th_ma = [
            self._compute_sliding_average(self.agent_step_th[a], window_size)
            for a in range(self.sim.num_agents)
        ]
        coll_ma = [
            self._compute_sliding_average(self.agent_step_collisions[a], window_size)
            for a in range(self.sim.num_agents)
        ]
        idle_ma = [
            self._compute_sliding_average(self.band_step_idle[b], window_size)
            for b in range(self.sim.num_bands)
        ]

        # Compute combined averages
        avg_coll_full = [
            sum(coll_ma[k][j] for k in range(self.sim.num_agents)) / self.sim.num_agents
            for j in range(len(coll_ma[0]))
        ]
        avg_idle_full = [
            sum(idle_ma[k][j] for k in range(self.sim.num_bands)) / self.sim.num_bands
            for j in range(len(idle_ma[0]))
        ]

        # Determine plot range (skip initial 100 steps for large simulations)
        start_idx = 0
        x_plot = list(range(1, num_steps + 1))
        avg_coll_ma = avg_coll_full
        avg_idle_ma = avg_idle_full

        if num_steps > 1000:
            start_idx = 100
            x_plot = list(range(101, num_steps + 1))
            for i in range(self.sim.num_agents):
                agent_th_ma[i] = agent_th_ma[i][start_idx:]
                coll_ma[i] = coll_ma[i][start_idx:]
            for i in range(self.sim.num_bands):
                idle_ma[i] = idle_ma[i][start_idx:]
            avg_coll_ma = avg_coll_full[start_idx:]
            avg_idle_ma = avg_idle_full[start_idx:]

        # Create graph frame
        graph_frame = tk.LabelFrame(self.root,
                                    text=(f"Combined Sliding Window Performance Graph "
                                          f"(Window Size: {window_size})"),
                                    font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                    padx=10, pady=10)
        graph_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Create matplotlib figure and axes
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot per-agent throughputs
        for agent_idx in range(self.sim.num_agents):
            ax.plot(x_plot, agent_th_ma[agent_idx], '-', linewidth=1.33,
                    label=f'Agent {agent_idx + 1} Throughput', color=f'C{agent_idx}')

        # Plot average collisions and idle time
        ax.plot(x_plot, avg_coll_ma, 'r--', linewidth=1.33, label='Avg Collisions per Agent')
        ax.plot(x_plot, avg_idle_ma, 'g--', linewidth=1.33, label='Avg Idle per Band')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Performance Metric')
        ax.set_title('Combined Performance Metrics over Time (Sliding Window Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Embed figure in Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    @staticmethod
    def _compute_sliding_average(data: List[Optional[float]], window_size: int) -> List[float]:
        """
        Computes the sliding window average of a list, handling None values by skipping them.

        Args:
            data: List of numerical data points (floats or None).
            window_size: The size of the sliding window.

        Returns:
            A list of sliding averages.
        """
        moving_average = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i + 1]
            # Filter out None values
            valid_data = [x for x in window_data if x is not None]
            if not valid_data:
                moving_average.append(0.0)
            else:
                moving_average.append(sum(valid_data) / len(valid_data))
        return moving_average

    def _restart(self):
        """
        Closes the result window and reopens the parameter window.
        """
        if self.root:
            self.root.destroy()
        ParamWindow().mainloop()


class Visualiser:
    """
    Tkinter window for live visualization of the simulation, including a
    transmission grid, progress bar, and updated metrics.
    """
    def __init__(self, sim: WirelessSimulation):
        self.sim = sim
        self.root: Optional[tk.Tk] = None
        self.completed_steps = 0
        self.overall_history: List[float] = []
        self.agent_step_th: List[List[Optional[float]]] = [[] for _ in range(self.sim.num_agents)]
        self.agent_step_collisions: List[List[int]] = [[] for _ in range(self.sim.num_agents)]
        self.band_step_idle: List[List[int]] = [[] for _ in range(self.sim.num_bands)]
        self.labels: List[List[Optional[tk.Label]]] = []
        self.prog_var = tk.DoubleVar()
        self.prog_bar: Optional[ttk.Progressbar] = None
        self.step_lbl: Optional[tk.Label] = None
        self.met_txt: Optional[tk.Text] = None
        self.restart_btn: Optional[tk.Button] = None

    def run(self):
        """
        Initializes the main visualization window and starts the update loop.
        """
        self.root = tk.Tk()
        self.root.title(f"Dynamic Wireless Channel Allocation – {self.sim.algorithm}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')

        # -------------------- Layout Setup -----------------------------
        main_frame = tk.Frame(self.root, bg='#f8f9fa')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title/Header
        tk.Label(main_frame,
                 text=(f"Simulation: {self.sim.algorithm} | "
                       f"Agents: {self.sim.num_agents} | "
                       f"Bands: {self.sim.num_bands} | "
                       f"Steps: {self.sim.num_time_steps}"),
                 font=("Arial", 16, "bold"), bg='#f8f9fa', fg='#333').pack(fill='x', pady=(0, 10))

        # ---- Transmission Grid Frame ------------------------------------
        grid_frame = tk.LabelFrame(main_frame, text="Transmission Grid (Bands Y / Time X)",
                                   font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                   padx=5, pady=5)
        grid_frame.pack(fill='x', pady=(0, 10))

        # Grid Header Label
        tk.Label(grid_frame, text="Bands (Y) \\ Time Steps (X)",
                 font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333') \
            .grid(row=0, column=0, columnspan=self.sim.num_time_steps + 1,
                  pady=5, sticky='ew')

        # Band labels (left column)
        for band_idx in range(self.sim.num_bands):
            tk.Label(grid_frame, text=f"B{band_idx+1}", width=10, relief="solid",
                     borderwidth=1, bg='#2196F3', fg='white',
                     font=("Arial", 10, "bold")) \
                .grid(row=band_idx + 1, column=0, sticky='nsew', padx=1)

        # Cells (Actual Grid)
        self.labels = [[None] * self.sim.num_time_steps for _ in range(self.sim.num_bands)]
        for band_idx in range(self.sim.num_bands):
            for time_idx in range(self.sim.num_time_steps):
                label = tk.Label(grid_frame, text="", width=10, height=2,
                                 relief="solid", borderwidth=1,
                                 font=("Arial", 9), justify='center',
                                 bg='white', fg='#333')
                label.grid(row=band_idx + 1, column=time_idx + 1, sticky='nsew', padx=1, pady=1)
                self.labels[band_idx][time_idx] = label

        # Grid column and row weights for resizing
        for col_idx in range(self.sim.num_time_steps + 1):
            grid_frame.grid_columnconfigure(col_idx, weight=1)
        for row_idx in range(self.sim.num_bands + 1):
            grid_frame.grid_rowconfigure(row_idx, weight=1)

        # ---- Progress Bar -----------------------------------------------
        progress_frame = tk.Frame(main_frame, bg='#f8f9fa')
        progress_frame.pack(fill='x', pady=(0, 10))
        tk.Label(progress_frame, text="Progress:", font=("Arial", 10, "bold"),
                 bg='#f8f9fa', fg='#333').pack(side='left')
        self.prog_var.set(0)
        self.prog_bar = ttk.Progressbar(progress_frame, variable=self.prog_var,
                                        maximum=self.sim.num_time_steps, length=300)
        self.prog_bar.pack(side='left', padx=10)
        self.step_lbl = tk.Label(progress_frame, text="Step 0/0", bg='#f8f9fa',
                                 font=("Arial", 10), fg='#333')
        self.step_lbl.pack(side='left')

        # ---- Metrics Display --------------------------------------------
        metrics_frame = tk.LabelFrame(main_frame, text="Performance Metrics",
                                      font=("Arial", 12, "bold"), bg='#f8f9fa',
                                      fg='#333', padx=10, pady=10)
        metrics_frame.pack(fill='x', pady=(0, 10))

        self.met_txt = tk.Text(metrics_frame, height=8, width=80,
                               font=("Courier", 10), state='disabled',
                               bg='white', relief='flat')
        scrollbar_metrics = tk.Scrollbar(metrics_frame, orient='vertical',
                                         command=self.met_txt.yview)
        self.met_txt.config(yscrollcommand=scrollbar_metrics.set)
        self.met_txt.pack(side='left', fill='both', expand=True)
        scrollbar_metrics.pack(side='right', fill='y')

        # Start loop
        self.update_metrics()
        self.root.after(200, self._step_loop)
        self.root.mainloop()

    def _step_loop(self):
        """
        Executes one simulation step, updates the UI, and schedules the next step.
        """
        if self.completed_steps < self.sim.num_time_steps:
            step_idx = self.completed_steps
            prev_succ = list(self.sim.success_counts)
            prev_att = list(self.sim.attempt_counts)
            self.sim.step(step_idx)

            # Collect per-agent throughput and collisions
            for agent_idx in range(self.sim.num_agents):
                successful_transmissions = self.sim.success_counts[agent_idx] - prev_succ[agent_idx]
                attempts = self.sim.attempt_counts[agent_idx] - prev_att[agent_idx]
                throughput = successful_transmissions / attempts if attempts > 0 else None
                self.agent_step_th[agent_idx].append(throughput)
                collisions = attempts - successful_transmissions
                self.agent_step_collisions[agent_idx].append(collisions)

            # Collect per-band idle slots
            for band_idx in range(self.sim.num_bands):
                grid_text = self.sim.grid_texts[band_idx][step_idx]
                has_users = bool(grid_text and any(c.isdigit() for c in grid_text))
                idle = 0 if has_users else 1
                self.band_step_idle[band_idx].append(idle)

            # Update grid cell colors/text
            for band_idx in range(self.sim.num_bands):
                grid_text = self.sim.grid_texts[band_idx][step_idx]
                users = [
                    int(x) for x in grid_text.replace('...', '').split(',') if x.isdigit()
                ] if grid_text else []
                label = self.labels[band_idx][step_idx]
                if not grid_text:
                    label.config(text="", bg="white")
                elif len(users) == 1:
                    # Single user: successful transmission
                    label.config(text=grid_text, bg="#4CAF50", fg="white")
                else:
                    # Multiple users or '...' (collision/complex state)
                    label.config(text=grid_text, bg="#F44336", fg="white")

            self.prog_var.set(self.completed_steps + 1)
            self.step_lbl.config(text=f"Step {self.completed_steps + 1}/{self.sim.num_time_steps}")
            self.completed_steps += 1
            self.update_metrics()
            self.root.after(200, self._step_loop)
        else:
            # Simulation complete
            self.sim.print_grid()
            self.step_lbl.config(text="Simulation Complete")
            self._add_sliding_window_graph()
            if not self.restart_btn and self.prog_bar and self.prog_bar.master:
                self.restart_btn = tk.Button(self.prog_bar.master,
                                             text="Restart Simulation",
                                             command=self._restart,
                                             bg='#4CAF50', fg='white',
                                             font=("Arial", 10, "bold"))
                self.restart_btn.pack(side='left', padx=10)

    def _add_sliding_window_graph(self):
        """
        Creates and embeds a matplotlib graph showing sliding window averages of
        key performance metrics (throughput, collisions, idle time).
        """
        if not self.step_lbl or not self.step_lbl.master or not self.step_lbl.master.master:
            return

        num_steps = self.sim.num_time_steps
        window_size = max(1, int(num_steps * 0.05))

        # Compute sliding averages
        agent_th_ma = [
            self._compute_sliding_average(self.agent_step_th[a], window_size)
            for a in range(self.sim.num_agents)
        ]
        coll_ma = [
            self._compute_sliding_average(self.agent_step_collisions[a], window_size)
            for a in range(self.sim.num_agents)
        ]
        idle_ma = [
            self._compute_sliding_average(self.band_step_idle[b], window_size)
            for b in range(self.sim.num_bands)
        ]

        # Compute combined averages
        avg_coll_full = [
            sum(coll_ma[k][j] for k in range(self.sim.num_agents)) / self.sim.num_agents
            for j in range(len(coll_ma[0]))
        ]
        avg_idle_full = [
            sum(idle_ma[k][j] for k in range(self.sim.num_bands)) / self.sim.num_bands
            for j in range(len(idle_ma[0]))
        ]

        # Determine plot range (skip initial 100 steps for large simulations)
        start_idx = 0
        x_plot = list(range(1, num_steps + 1))
        avg_coll_ma = avg_coll_full
        avg_idle_ma = avg_idle_full

        if num_steps > 1000:
            start_idx = 100
            x_plot = list(range(101, num_steps + 1))
            for i in range(self.sim.num_agents):
                agent_th_ma[i] = agent_th_ma[i][start_idx:]
                coll_ma[i] = coll_ma[i][start_idx:]
            for i in range(self.sim.num_bands):
                idle_ma[i] = idle_ma[i][start_idx:]
            avg_coll_ma = avg_coll_full[start_idx:]
            avg_idle_ma = avg_idle_full[start_idx:]

        # Create graph frame
        graph_frame = tk.LabelFrame(self.step_lbl.master.master,  # main_frame
                                    text=(f"Combined Sliding Window Performance Graph "
                                          f"(Window Size: {window_size})"),
                                    font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                    padx=10, pady=10)
        graph_frame.pack(fill='both', expand=True, pady=(10, 0))

        # Create matplotlib figure and axes
        fig = Figure(figsize=(12, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Plot per-agent throughputs
        for agent_idx in range(self.sim.num_agents):
            ax.plot(x_plot, agent_th_ma[agent_idx], '-', linewidth=1.33,
                    label=f'Agent {agent_idx + 1} Throughput', color=f'C{agent_idx}')

        # Plot average collisions and idle time
        ax.plot(x_plot, avg_coll_ma, 'r--', linewidth=1.33, label='Avg Collisions per Agent')
        ax.plot(x_plot, avg_idle_ma, 'g--', linewidth=1.33, label='Avg Idle per Band')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Performance Metric')
        ax.set_title('Combined Performance Metrics over Time (Sliding Window Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Embed figure in Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    @staticmethod
    def _compute_sliding_average(data: List[Optional[float]], window_size: int) -> List[float]:
        """
        Computes the sliding window average of a list, handling None values by skipping them.

        Args:
            data: List of numerical data points (floats or None).
            window_size: The size of the sliding window.

        Returns:
            A list of sliding averages.
        """
        moving_average = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i + 1]
            # Filter out None values
            valid_data = [x for x in window_data if x is not None]
            if not valid_data:
                moving_average.append(0.0)
            else:
                moving_average.append(sum(valid_data) / len(valid_data))
        return moving_average

    def update_metrics(self):
        """
        Calculates and updates the displayed performance metrics.
        """
        if not self.met_txt:
            return

        completed = self.completed_steps
        total_successes = sum(self.sim.success_counts)
        total_attempts = sum(self.sim.attempt_counts)

        overall = 0.0
        per_agent_th = [0.0] * self.sim.num_agents
        fairness = 1.0

        if total_attempts > 0:
            overall = total_successes / total_attempts
            per_agent_th = [
                self.sim.success_counts[a] / self.sim.attempt_counts[a]
                if self.sim.attempt_counts[a] > 0 else 0.0
                for a in range(self.sim.num_agents)
            ]
            fairness = jains_fairness(per_agent_th)

        self.overall_history.append(overall)
        title = "Initial State" if completed == 0 else f"After Step {completed}"
        content = (f"{title}:\n\n"
                   f"Throughput (overall, Success/Attempt): {overall:.3f}\n"
                   f"Fairness Index: {fairness:.3f}\n\n"
                   f"Per-Agent Throughput (Success/Attempt):\n")
        for agent_idx, throughput in enumerate(per_agent_th, 1):
            content += f"Agent {agent_idx}: {throughput:.3f}\n"

        self.met_txt.config(state='normal')
        self.met_txt.delete(1.0, tk.END)
        self.met_txt.insert(1.0, content)
        self.met_txt.config(state='disabled')

    def _restart(self):
        """
        Closes the visualisation window and reopens the parameter window.
        """
        if self.root:
            self.root.destroy()
        ParamWindow().mainloop()


if __name__ == "__main__":
    ParamWindow().mainloop()