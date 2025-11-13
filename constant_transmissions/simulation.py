#!/usr/bin/env python3
"""
simulation.py
-------------
All Tkinter UI code:
* Parameter window
* Visualisation grid + progress bar + live metrics
* Headless result window
* Restart logic
"""

import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import only the core engine – no UI code lives in main.py
from main import WirelessSimulation


def jains_fairness(throughputs):
    if not throughputs:
        return 1.0
    n = len(throughputs)
    sum_x = sum(throughputs)
    sum_x2 = sum(x * x for x in throughputs)
    if sum_x == 0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


class ParamWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wireless Channel Allocation Simulation")
        self.geometry("400x400")
        self.configure(bg='#f0f0f0')

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Title
        tk.Label(self,
                 text="Dynamic Wireless Channel Allocation Simulation",
                 font=("Arial", 16, "bold"),
                 bg='#f0f0f0', fg='#333').pack(pady=10)

        # Description
        tk.Label(self,
                 text="Configure the simulation parameters below. "
                      "Each agent has a configurable number of packets to transmit across available bands.",
                 bg='#f0f0f0', fg='#666', wraplength=350, justify='center').pack(pady=5)

        # Variables
        self.num_agents_var = tk.IntVar(value=4)
        self.num_bands_var = tk.IntVar(value=3)
        self.num_steps_var = tk.IntVar(value=100)
        self.alg_var = tk.StringVar(value='Reinforcement Learning')
        self.vis_var = tk.BooleanVar(value=True)

        # Input frame
        input_frame = tk.LabelFrame(self, text="Parameters", font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#333', padx=10, pady=10)
        input_frame.pack(pady=10, padx=20, fill='x')

        # ---- entries ----------------------------------------------------
        tk.Label(input_frame, text="Number of Agents (1-10):", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_agents_var, width=10,
                 font=("Arial", 10)).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Label(input_frame, text="Number of Bands (1-10):", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_bands_var, width=10,
                 font=("Arial", 10)).grid(row=1, column=1, padx=5, pady=5, sticky='w')

        tk.Label(input_frame, text="Number of Time Steps:", bg='#f0f0f0',
                 font=("Arial", 10)).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Entry(input_frame, textvariable=self.num_steps_var, width=10,
                 font=("Arial", 10)).grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # ---- algorithm --------------------------------------------------
        alg_frame = tk.Frame(input_frame, bg='#f0f0f0')
        alg_frame.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        tk.Label(alg_frame, text="Algorithm:", bg='#f0f0f0',
                 font=("Arial", 10)).pack(side='left')
        ttk.Combobox(alg_frame, textvariable=self.alg_var,
                     values=['Random', 'Reinforcement Learning', 'Machine Learning'],
                     state='readonly', width=20).pack(side='left', padx=(5, 0))

        # ---- visualisation checkbox --------------------------------------
        tk.Checkbutton(input_frame, text="Enable Visualization (Grid Display)",
                       variable=self.vis_var, bg='#f0f0f0', selectcolor='#e3f2fd',
                       font=("Arial", 10)).grid(row=4, column=0, columnspan=2,
                                                sticky='w', padx=5, pady=10)

        # ---- start button ------------------------------------------------
        tk.Button(self, text="Start Simulation", command=self.start_sim,
                  bg='#4CAF50', fg='white', font=("Arial", 12, "bold"),
                  relief='raised', bd=2, cursor='hand2').pack(pady=20)

    def start_sim(self):
        try:
            na = self.num_agents_var.get()
            nb = self.num_bands_var.get()
            ns = self.num_steps_var.get()
            alg = self.alg_var.get()
            vis = self.vis_var.get()

            if not (1 <= na <= 10 and 1 <= nb <= 10 and ns >= 1):
                raise ValueError("Agents/Bands: 1-10, Steps: >=1")

            self.destroy()
            sim = WirelessSimulation(na, nb, ns, alg)
            if vis:
                Visualiser(sim).run()
            else:
                HeadlessResult(sim).run()
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))


# ------------------------------------------------------------------------- #
# Headless result window (shown when visualisation is disabled)
# ------------------------------------------------------------------------- #
class HeadlessResult:
    def __init__(self, sim: WirelessSimulation):
        self.sim = sim
        self.overall_history = []

    def run(self):
        # Initial metrics (before any steps)
        self.overall_history.append(0.0)

        # Run simulation step by step to collect history
        for step in range(self.sim.num_time_steps):
            self.sim.step(step)
            denom = (step + 1) * self.sim.num_bands
            total_succ = sum(self.sim.success_counts)
            overall = total_succ / denom
            self.overall_history.append(overall)

        # Final metrics
        denom = self.sim.num_time_steps * self.sim.num_bands
        total_succ = sum(self.sim.success_counts)
        overall = total_succ / denom
        per_agent = [self.sim.success_counts[a] / denom for a in range(self.sim.num_agents)]
        fairness = jains_fairness(per_agent)

        root = tk.Tk()
        self.root = root
        root.title(f"Simulation Metrics - {self.sim.algorithm}")
        root.geometry("1000x700")
        root.configure(bg='#f8f9fa')

        tk.Label(root, text="Performance Metrics", font=("Arial", 14, "bold"),
                 bg='#f8f9fa').pack(pady=10)

        # Text frame for metrics (fixed height)
        text_frame = tk.Frame(root, bg='#f8f9fa')
        text_frame.pack(fill='x', padx=20, pady=10)

        txt = tk.Text(text_frame, wrap='word', height=8, font=("Courier", 11), bg='white',
                      relief='flat', padx=10, pady=10)
        sb = tk.Scrollbar(text_frame, orient='vertical', command=txt.yview)
        txt.config(yscrollcommand=sb.set)

        content = (f"SIMULATION COMPLETE!\n\n"
                   f"Overall Throughput: {overall:.3f}\n"
                   f"Jain's Fairness Index: {fairness:.3f}\n\n"
                   f"Throughput Per Agent:\n")
        for a, th in enumerate(per_agent, 1):
            content += f"Agent {a}: {th:.3f}\n"

        txt.insert(1.0, content)
        txt.config(state='disabled')
        txt.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        # Add the sliding window graph
        self._add_sliding_window_graph()

        # Add restart button
        self.restart_btn = tk.Button(root, text="Restart Simulation", command=self._restart,
                                     bg='#4CAF50', fg='white', font=("Arial", 10, "bold"),
                                     relief='raised', bd=2, cursor='hand2')
        self.restart_btn.pack(pady=10)

        root.mainloop()

    def _add_sliding_window_graph(self):
        ns = self.sim.num_time_steps
        nb = self.sim.num_bands
        window_size = max(1, int(ns * 0.05))

        # Compute total successes up to each step (excluding initial 0)
        total_success = [self.overall_history[t + 1] * (t + 1) * nb for t in range(ns)]

        # Compute instantaneous throughput per step
        inst_throughput = []
        prev_total = 0
        for ts in total_success:
            inst_throughput.append((ts - prev_total) / nb)
            prev_total = ts

        # Compute sliding window moving average
        def compute_sliding_average(data, w_size):
            ma = []
            for i in range(len(data)):
                start_idx = max(0, i - w_size + 1)
                window_data = data[start_idx:i + 1]
                ma.append(sum(window_data) / len(window_data))
            return ma

        y_ma = compute_sliding_average(inst_throughput, window_size)

        # Create graph frame
        graph_frame = tk.LabelFrame(self.root,
                                    text=f"Sliding Window Throughput Graph (Window Size: {window_size})",
                                    font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                    padx=10, pady=10)
        graph_frame.pack(fill='both', expand=True, pady=10)

        # Create matplotlib figure
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        x = list(range(1, ns + 1))
        ax.plot(x, y_ma, 'b-', linewidth=2, label='Sliding Window Average Throughput')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Throughput')
        ax.set_title('Throughput over Time (Sliding Window Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _restart(self):
        self.root.destroy()
        ParamWindow().mainloop()


# ------------------------------------------------------------------------- #
# Full visualisation (grid + live metrics + progress)
# ------------------------------------------------------------------------- #
class Visualiser:
    def __init__(self, sim: WirelessSimulation):
        self.sim = sim
        self.completed_steps = 0
        self.overall_history = []

    def run(self):
        self.root = tk.Tk()
        self.root.title(f"Dynamic Wireless Channel Allocation – {self.sim.algorithm}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')

        # ----------------------------------------------------------------- #
        # Layout
        # ----------------------------------------------------------------- #
        main = tk.Frame(self.root, bg='#f8f9fa')
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # title
        tk.Label(main,
                 text=f"Simulation: {self.sim.algorithm} | "
                      f"Agents: {self.sim.num_agents} | "
                      f"Bands: {self.sim.num_bands} | "
                      f"Steps: {self.sim.num_time_steps}",
                 font=("Arial", 16, "bold"), bg='#f8f9fa', fg='#333').pack(fill='x', pady=(0, 10))

        # ---- grid ------------------------------------------------------- #
        grid_frame = tk.LabelFrame(main, text="Transmission Grid (Bands Y / Time X)",
                                   font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                   padx=5, pady=5)
        grid_frame.pack(fill='x', pady=(0, 10))

        # header
        tk.Label(grid_frame, text="Bands (Y) \\ Time Steps (X)",
                 font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333')\
            .grid(row=0, column=0, columnspan=self.sim.num_time_steps + 1,
                  pady=5, sticky='ew')

        # band labels (left)
        for b in range(self.sim.num_bands):
            tk.Label(grid_frame, text=f"B{b+1}", width=10, relief="solid",
                     borderwidth=1, bg='#2196F3', fg='white',
                     font=("Arial", 10, "bold"))\
                .grid(row=b + 1, column=0, sticky='nsew', padx=1)

        # time labels (top)
        for t_idx, t_val in enumerate(range(1, self.sim.num_time_steps + 1)):
            tk.Label(grid_frame, text=f"t={t_val}", width=10, relief="solid",
                     borderwidth=1, bg='#FF9800', fg='white',
                     font=("Arial", 10, "bold"))\
                .grid(row=0, column=t_idx + 1, sticky='nsew', padx=1)

        # cells
        self.labels = [[None] * self.sim.num_time_steps for _ in range(self.sim.num_bands)]
        for b in range(self.sim.num_bands):
            for t in range(self.sim.num_time_steps):
                lbl = tk.Label(grid_frame, text="", width=10, height=2,
                               relief="solid", borderwidth=1,
                               font=("Arial", 9), justify='center',
                               bg='white', fg='#333')
                lbl.grid(row=b + 1, column=t + 1, sticky='nsew', padx=1, pady=1)
                self.labels[b][t] = lbl

        # grid weights
        for c in range(self.sim.num_time_steps + 1):
            grid_frame.grid_columnconfigure(c, weight=1)
        for r in range(self.sim.num_bands + 1):
            grid_frame.grid_rowconfigure(r, weight=1)

        # ---- progress --------------------------------------------------- #
        prog_fr = tk.Frame(main, bg='#f8f9fa')
        prog_fr.pack(fill='x', pady=(0, 10))
        tk.Label(prog_fr, text="Progress:", font=("Arial", 10, "bold"),
                 bg='#f8f9fa', fg='#333').pack(side='left')
        self.prog_var = tk.DoubleVar()
        self.prog_bar = ttk.Progressbar(prog_fr, variable=self.prog_var,
                                        maximum=self.sim.num_time_steps, length=300)
        self.prog_bar.pack(side='left', padx=10)
        self.step_lbl = tk.Label(prog_fr, text="Step 0/0", bg='#f8f9fa',
                                 font=("Arial", 10), fg='#333')
        self.step_lbl.pack(side='left')

        # ---- metrics ---------------------------------------------------- #
        met_fr = tk.LabelFrame(main, text="Performance Metrics",
                               font=("Arial", 12, "bold"), bg='#f8f9fa',
                               fg='#333', padx=10, pady=10)
        met_fr.pack(fill='x', pady=(0, 10))

        self.met_txt = tk.Text(met_fr, height=8, width=80,
                               font=("Courier", 10), state='disabled',
                               bg='white', relief='flat')
        sb_met = tk.Scrollbar(met_fr, orient='vertical',
                              command=self.met_txt.yview)
        self.met_txt.config(yscrollcommand=sb_met.set)
        self.met_txt.pack(side='left', fill='both', expand=True)
        sb_met.pack(side='right', fill='y')

        # ----------------------------------------------------------------- #
        self.restart_btn = None
        self.update_metrics()
        self.root.after(200, self._step_loop)
        self.root.mainloop()

    # --------------------------------------------------------------------- #
    def _step_loop(self):
        if self.completed_steps < self.sim.num_time_steps:
            step_idx = self.completed_steps
            self.sim.step(step_idx)

            # update grid colours
            for b in range(self.sim.num_bands):
                txt = self.sim.grid_texts[b][step_idx]
                users = [int(x) for x in txt.replace('...', '').split(',') if x.isdigit()] if txt else []
                lbl = self.labels[b][step_idx]
                if not txt:
                    lbl.config(text="", bg="white")
                elif len(users) == 1:
                    lbl.config(text=txt, bg="#4CAF50", fg="white")
                else:
                    lbl.config(text=txt, bg="#F44336", fg="white")

            self.prog_var.set(self.completed_steps + 1)
            self.step_lbl.config(text=f"Step {self.completed_steps + 1}/{self.sim.num_time_steps}")
            self.completed_steps += 1
            self.update_metrics()
            self.root.after(200, self._step_loop)
        else:
            self.sim.print_grid()
            self.step_lbl.config(text="Simulation Complete")
            self._add_sliding_window_graph()
            if not self.restart_btn:
                self.restart_btn = tk.Button(self.prog_bar.master,
                                             text="Restart Simulation",
                                             command=self._restart,
                                             bg='#4CAF50', fg='white',
                                             font=("Arial", 10, "bold"))
                self.restart_btn.pack(side='left', padx=10)

    # --------------------------------------------------------------------- #
    def _add_sliding_window_graph(self):
        ns = self.sim.num_time_steps
        nb = self.sim.num_bands
        window_size = max(1, int(ns * 0.05))

        # Compute total successes up to each step (excluding initial 0)
        total_success = [self.overall_history[t + 1] * (t + 1) * nb for t in range(ns)]

        # Compute instantaneous throughput per step
        inst_throughput = []
        prev_total = 0
        for ts in total_success:
            inst_throughput.append((ts - prev_total) / nb)
            prev_total = ts

        # Compute sliding window moving average
        def compute_sliding_average(data, w_size):
            ma = []
            for i in range(len(data)):
                start_idx = max(0, i - w_size + 1)
                window_data = data[start_idx:i + 1]
                ma.append(sum(window_data) / len(window_data))
            return ma

        y_ma = compute_sliding_average(inst_throughput, window_size)

        # Create graph frame
        graph_frame = tk.LabelFrame(self.step_lbl.master.master,  # main
                                    text=f"Sliding Window Throughput Graph (Window Size: {window_size})",
                                    font=("Arial", 12, "bold"), bg='#f8f9fa', fg='#333',
                                    padx=10, pady=10)
        graph_frame.pack(fill='both', expand=True, pady=(10, 0))

        # Create matplotlib figure
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        x = list(range(1, ns + 1))
        ax.plot(x, y_ma, 'b-', linewidth=2, label='Sliding Window Average Throughput')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Throughput')
        ax.set_title('Throughput over Time (Sliding Window Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # --------------------------------------------------------------------- #
    def update_metrics(self):
        completed = self.completed_steps
        denom = completed * self.sim.num_bands if completed > 0 else 0
        if denom == 0:
            overall = 0.0
            per_agent_th = [0.0] * self.sim.num_agents
            fairness = 1.0
        else:
            total_succ = sum(self.sim.success_counts)
            overall = total_succ / denom
            per_agent_th = [self.sim.success_counts[a] / denom for a in range(self.sim.num_agents)]
            fairness = jains_fairness(per_agent_th)
        self.overall_history.append(overall)
        title = "Initial State" if completed == 0 else f"After Step {completed}"
        content = (f"{title}:\n\n"
                   f"Throughput (overall): {overall:.3f}\n"
                   f"Fairness Index: {fairness:.3f}\n\n"
                   f"Per-Agent Throughput:\n")
        for a, th in enumerate(per_agent_th, 1):
            content += f"Agent {a}: {th:.3f}\n"

        self.met_txt.config(state='normal')
        self.met_txt.delete(1.0, tk.END)
        self.met_txt.insert(1.0, content)
        self.met_txt.config(state='disabled')

    # --------------------------------------------------------------------- #
    def _restart(self):
        self.root.destroy()
        ParamWindow().mainloop()


# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    ParamWindow().mainloop()