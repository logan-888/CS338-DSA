import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
import pprint
import math
import numpy as np

class WirelessSimulatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Dynamic Wireless Allocation Simulator")
        self.master.geometry("950x600")
        self.master.configure(bg="#f4f4f4")
        self.font_main = ("Segoe UI", 11)
        self.running = False

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.configure("TLabel", font=self.font_main, background="#f4f4f4")
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background="#f4f4f4")

        self.create_input_ui()

    def create_input_ui(self):
        for widget in self.master.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.master, padding=20)
        frame.pack(expand=True)

        ttk.Label(frame, text="Dynamic Wireless Allocation Simulator", style="Header.TLabel").pack(pady=15)

        form = ttk.Frame(frame)
        form.pack(pady=10)

        ttk.Label(form, text="Number of Agents (1–10):").grid(row=0, column=0, sticky="e", pady=5, padx=5)
        self.agents_entry = ttk.Entry(form, width=10)
        self.agents_entry.grid(row=0, column=1, pady=5)

        ttk.Label(form, text="Number of Bands (1–10):").grid(row=1, column=0, sticky="e", pady=5, padx=5)
        self.bands_entry = ttk.Entry(form, width=10)
        self.bands_entry.grid(row=1, column=1, pady=5)

        ttk.Label(form, text="Number of Time Steps:").grid(row=2, column=0, sticky="e", pady=5, padx=5)
        self.time_entry = ttk.Entry(form, width=10)
        self.time_entry.grid(row=2, column=1, pady=5)

        ttk.Label(form, text="Algorithm:").grid(row=3, column=0, sticky="e", pady=5, padx=5)
        self.algo_var = tk.StringVar(value="Random")
        self.algo_combo = ttk.Combobox(form, textvariable=self.algo_var, values=["Random", "ML approach", "RL approach"], state="readonly")
        self.algo_combo.grid(row=3, column=1, pady=5)

        self.random_var = tk.BooleanVar(value=False)
        self.random_check = ttk.Checkbutton(form, text="Enable Random Transmission", variable=self.random_var, command=self.toggle_prob_ui)
        self.random_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.prob_frame = ttk.Frame(form)
        ttk.Label(self.prob_frame, text="Transmission Probability (%, 0-100):").grid(row=0, column=0, sticky="e", pady=5, padx=5)
        self.prob_entry = ttk.Entry(self.prob_frame, width=10)
        self.prob_entry.grid(row=0, column=1, pady=5)
        self.prob_entry.insert(0, "50")

        self.jammer_var = tk.BooleanVar(value=False)
        self.jammer_check = ttk.Checkbutton(form, text="Enable Laser Jammer", variable=self.jammer_var, command=self.toggle_jammer_ui)
        self.jammer_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.jammer_frame = ttk.Frame(form)
        ttk.Label(self.jammer_frame, text="Impacted Bands (1-10):").grid(row=0, column=0, sticky="e", pady=5, padx=5)
        self.jammed_bands_entry = ttk.Entry(self.jammer_frame, width=10)
        self.jammed_bands_entry.grid(row=0, column=1, pady=5)
        self.jammed_bands_entry.insert(0, "1")

        ttk.Label(self.jammer_frame, text="Impacted Time Steps:").grid(row=1, column=0, sticky="e", pady=5, padx=5)
        self.jammed_times_entry = ttk.Entry(self.jammer_frame, width=10)
        self.jammed_times_entry.grid(row=1, column=1, pady=5)
        self.jammed_times_entry.insert(0, "1")

        self.jammer_frame.grid_remove()

        self.visual_var = tk.BooleanVar(value=False)
        self.visual_check = ttk.Checkbutton(form, text="Disable Visual Mode", variable=self.visual_var)
        self.visual_check.grid(row=8, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        # Initial UI states
        self.toggle_prob_ui()
        self.toggle_jammer_ui()

        ttk.Button(frame, text="Start Simulation", command=self.start_simulation).pack(pady=15)

    def toggle_prob_ui(self):
        if self.random_var.get():
            self.prob_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        else:
            self.prob_frame.grid_remove()

    def toggle_jammer_ui(self):
        if self.jammer_var.get():
            self.jammer_frame.grid(row=7, column=0, columnspan=2, sticky="w", pady=5)
        else:
            self.jammer_frame.grid_remove()

    def start_simulation(self):
        try:
            self.num_agents = int(self.agents_entry.get())
            self.num_bands = int(self.bands_entry.get())
            self.time_steps = int(self.time_entry.get())
            if not (1 <= self.num_agents <= 10 and 1 <= self.num_bands <= 10):
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integer values within the specified ranges.")
            return

        self.selected_algo = self.algo_var.get()

        self.random_transmission = self.random_var.get()

        if self.random_transmission:
            try:
                prob = int(self.prob_entry.get())
                if not 0 <= prob <= 100:
                    raise ValueError
                self.trans_prob = prob / 100.0
            except ValueError:
                messagebox.showerror("Input Error", "Transmission probability must be an integer between 0 and 100.")
                return
        else:
            self.trans_prob = 1.0  # always transmit if not random

        self.enable_jammer = self.jammer_var.get()
        self.jammed_set = set()
        if self.enable_jammer:
            try:
                num_jb = int(self.jammed_bands_entry.get())
                num_jt = int(self.jammed_times_entry.get())
                if not (1 <= num_jb <= self.num_bands and 1 <= num_jt <= self.time_steps):
                    raise ValueError("Impacted values out of range.")
                jammed_bands_list = random.sample(range(self.num_bands), num_jb)
                jammed_times_list = random.sample(range(self.time_steps), num_jt)
                self.jammed_set = {(band, ts) for band in jammed_bands_list for ts in jammed_times_list}
            except ValueError as e:
                messagebox.showerror("Input Error", str(e) if "out of range" in str(e) else "Invalid impacted values.")
                return

        self.collisions = 0
        self.successful = 0
        self.t = 0
        self.occupancy = np.zeros((self.time_steps, self.num_bands, self.num_agents), dtype=bool)
        self.colors = np.full((self.num_bands, self.time_steps), "#FFFFFF", dtype='<U7')
        self.agent_success = np.zeros(self.num_agents, dtype=int)

        # Pre-set jammed cells
        for band, ts in self.jammed_set:
            self.colors[band, ts] = "#D50000"

        self.visual_enabled = not self.visual_var.get()

        for widget in self.master.winfo_children():
            widget.destroy()

        if self.visual_enabled:
            # Enlarge the window for the simulation UI
            self.master.geometry("1600x900")

            self.create_sim_ui()
        else:
            self.master.geometry("950x600")

            frame = ttk.Frame(self.master, padding=20)
            frame.pack(expand=True, fill="both")

            ttk.Label(frame, text="Dynamic Wireless Allocation Simulator - Text Mode", style="Header.TLabel").pack(pady=15)

            self.stats_label = ttk.Label(frame, text="Running simulation...", font=("Segoe UI", 12), justify="left")
            self.stats_label.pack(pady=10)

            ttk.Button(frame, text="Restart", command=self.create_input_ui).pack(pady=10)

            self.run_full_simulation()

    def run_full_simulation(self):
        for _ in range(self.time_steps):
            self.simulate_step()
            self.t += 1
        self.finish_simulation()

    def create_sim_ui(self):
        top = ttk.Frame(self.master)
        top.pack(pady=10)

        ttk.Label(top, text="Dynamic Wireless Allocation Simulation", style="Header.TLabel").pack()

        self.canvas = tk.Canvas(self.master, width=1400, height=600, bg="white", highlightthickness=1, highlightbackground="#666")
        self.canvas.pack(pady=10)

        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=5)

        self.next_button = ttk.Button(control_frame, text="Next Step", command=self.next_step)
        self.next_button.grid(row=0, column=0, padx=5)

        self.play_button = ttk.Button(control_frame, text="▶ Auto Run", command=self.toggle_run)
        self.play_button.grid(row=0, column=1, padx=5)

        self.reset_button = ttk.Button(control_frame, text="↻ Restart", command=self.create_input_ui)
        self.reset_button.grid(row=0, column=2, padx=5)

        self.stats_label = ttk.Label(self.master, text="", font=("Segoe UI", 12), justify="left")
        self.stats_label.pack(pady=10)

        self.cell_size_x = 1300 / self.time_steps
        self.cell_size_y = 550 / self.num_bands
        self.draw_grid()

    def simulate_step(self):
        # Agent transmissions based on selected algorithm
        if self.selected_algo == "Random":
            if self.random_transmission:
                for agent in range(self.num_agents):
                    if random.random() < self.trans_prob:
                        band = np.random.randint(0, self.num_bands)
                        self.occupancy[self.t, band, agent] = True
            else:
                for agent in range(self.num_agents):
                    band = np.random.randint(0, self.num_bands)
                    self.occupancy[self.t, band, agent] = True
        elif self.selected_algo == "ML approach":
            self.ml_approach()
        elif self.selected_algo == "RL approach":
            self.rl_approach()

        for b in range(self.num_bands):
            count = np.sum(self.occupancy[self.t, b])
            is_jammed = (b, self.t) in self.jammed_set
            if is_jammed:
                self.colors[b, self.t] = "#D50000"
                if count > 0:
                    self.collisions += 1
            else:
                if count == 0:
                    self.colors[b, self.t] = "#FFD600"  # yellow (empty)
                elif count == 1:
                    self.colors[b, self.t] = "#00C853"  # bright green (success)
                    self.successful += 1
                    agent = np.argwhere(self.occupancy[self.t, b])[0, 0]
                    self.agent_success[agent] += 1
                else:
                    self.colors[b, self.t] = "#D50000"  # bright red (collision)
                    self.collisions += 1

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.num_bands):
            for j in range(self.time_steps):
                x0 = 50 + j * self.cell_size_x
                y0 = 30 + i * self.cell_size_y
                x1 = x0 + self.cell_size_x
                y1 = y0 + self.cell_size_y
                color = self.colors[i, j]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#333", width=1.8)

                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                active_agents = np.where(self.occupancy[j, i])[0] + 1
                agent_str = ",".join(map(str, active_agents)) if len(active_agents) > 0 else ""
                is_jammed = (i, j) in self.jammed_set

                if is_jammed:
                    if len(active_agents) > 0:
                        self.canvas.create_text(mid_x, mid_y - 6, text="JAMMED", fill="white", font=("Segoe UI", 10, "bold"))
                        self.canvas.create_text(mid_x, mid_y + 6, text=agent_str, fill="white", font=("Segoe UI", 8))
                    else:
                        self.canvas.create_text(mid_x, mid_y, text="JAMMED", fill="white", font=("Segoe UI", 11, "bold"))
                else:
                    if len(active_agents) > 0:
                        text_color = "white" if color == "#D50000" else "black"
                        font_size = 11 if len(agent_str) < 10 else 9
                        self.canvas.create_text(mid_x, mid_y, text=agent_str, fill=text_color, font=("Segoe UI", font_size, "bold"))

        self.canvas.create_text(700, 15, text=f"Time Step: {self.t}/{self.time_steps}", font=("Segoe UI", 12, "bold"))

    def next_step(self):
        if self.t >= self.time_steps:
            self.finish_simulation()
            return

        self.simulate_step()
        self.t += 1
        self.draw_grid()
        if self.t >= self.time_steps:
            self.finish_simulation()

    def ml_approach(self):
        """
        Improved Thompson Sampling with exponential forgetting + epsilon exploration.
        - Each agent has Beta(alpha,beta) per band.
        - Apply exponential decay to alpha/beta each timestep to handle nonstationarity.
        - Small epsilon chance of uniform random action to avoid getting stuck.
        - Updates are based ONLY on each agent's own previous action/outcome.
        """
        decay = 0.98  # exponential forgetting per global timestep (0.98 -> remembers but adapts)
        epsilon = 0.05  # small personal exploration probability

        # initialize structures
        if not hasattr(self, "ml_state"):
            # ml_state: (agents, bands, 2) for alpha beta
            self.ml_state = np.full((self.num_agents, self.num_bands, 2), 1.0)
            self.ml_prev_actions = [None] * self.num_agents
            # small pseudo-count to avoid stuck priors
            self.ml_pseudo = 1.0

        # choose actions for this timestep
        for agent in range(self.num_agents):
            if self.random_transmission and random.random() >= self.trans_prob:
                # agent does not transmit this timestep
                self.ml_prev_actions[agent] = None
                continue

            # epsilon random exploration
            if random.random() < epsilon:
                action = np.random.randint(0, self.num_bands)
                self.occupancy[self.t, action, agent] = True
                self.ml_prev_actions[agent] = action
                continue

            # Thompson sampling: sample from Beta for each band
            dists = self.ml_state[agent]
            samples = np.array([random.betavariate(max(0.001, d[0]), max(0.001, d[1])) for d in dists])
            action = np.argmax(samples)
            self.occupancy[self.t, action, agent] = True
            self.ml_prev_actions[agent] = action

        # Learning from the previous timestep (only local info)
        if self.t > 0:
            # apply forgetting to all agents' distributions (exponential decay)
            self.ml_state[:, :, 0] = np.maximum(0.01, (self.ml_state[:, :, 0] - 1.0) * decay + 1.0)
            self.ml_state[:, :, 1] = np.maximum(0.01, (self.ml_state[:, :, 1] - 1.0) * decay + 1.0)

            for agent in range(self.num_agents):
                last_action = self.ml_prev_actions[agent]
                if last_action is None:
                    continue

                count = np.sum(self.occupancy[self.t - 1, last_action])
                # Agent only observes whether *it* succeeded
                if self.occupancy[self.t - 1, last_action, agent] and count == 1 and ((last_action, self.t - 1) not in self.jammed_set):
                    # success
                    self.ml_state[agent, last_action, 0] += 1.0
                else:
                    # failure = collision OR jam OR didn't transmit alone
                    self.ml_state[agent, last_action, 1] += 1.0

    def rl_approach(self):
        """
        Independent Exp3 per agent (adversarial bandit algorithm).
        - Maintains positive weights and samples according to probability vector.
        - Updates weights using only local observed reward (success = +1, failure = 0 -> mapped to [0,1]).
        - Adaptive gamma (exploration) decays slightly over time.
        - Good for multi-agent adversarial settings where other agents' actions look like adversary.
        """
        # parameters
        gamma_init = 0.2
        gamma_decay = 0.999  # slowly reduce exploration
        min_gamma = 0.03

        if not hasattr(self, "exp3_state"):
            # per-agent weights (positive), one weight per band
            self.exp3_state = np.ones((self.num_agents, self.num_bands))
            self.exp3_prev_actions = [None] * self.num_agents
            self.exp3_gamma = gamma_init

        # decay gamma
        self.exp3_gamma = max(min_gamma, self.exp3_gamma * gamma_decay)

        # choose actions by sampling probability distribution for each agent
        for agent in range(self.num_agents):
            if self.random_transmission and random.random() >= self.trans_prob:
                self.exp3_prev_actions[agent] = None
                continue

            weights = self.exp3_state[agent]
            total_w = np.sum(weights)
            if total_w <= 0:
                # reinitialize small positive weights
                self.exp3_state[agent] = np.ones(self.num_bands)
                total_w = self.num_bands

            # base distribution from weights
            probs = (1 - self.exp3_gamma) * (weights / total_w) + (self.exp3_gamma / self.num_bands)

            # sample an action according to probs
            action = np.random.choice(self.num_bands, p=probs)

            self.occupancy[self.t, action, agent] = True
            self.exp3_prev_actions[agent] = (action, probs[action])  # store chosen action and its prob

        # update weights from previous timestep using only local feedback
        if self.t > 0:
            for agent in range(self.num_agents):
                prev = self.exp3_prev_actions[agent]
                if prev is None:
                    continue
                action, prob_action = prev

                count = np.sum(self.occupancy[self.t - 1, action])
                # local agent success detection
                # reward in [0,1] for Exp3 (1 for success, 0 for failure)
                if self.occupancy[self.t - 1, action, agent] and count == 1 and ((action, self.t - 1) not in self.jammed_set):
                    reward = 1.0
                else:
                    reward = 0.0

                # importance-weighted estimated reward
                x_hat = reward / max(prob_action, 1e-12)

                # update rule: w_i <- w_i * exp(gamma * x_hat / K)
                K = self.num_bands
                eta = self.exp3_gamma / K
                # multiply only the selected action's weight
                self.exp3_state[agent, action] *= np.exp(eta * x_hat)
                # clamp to avoid over/underflow
                self.exp3_state[agent, action] = np.clip(self.exp3_state[agent, action], 1e-6, 1e6)

    def finish_simulation(self):
        total_slots = self.time_steps * self.num_bands
        overall_throughput = self.successful / total_slots if total_slots > 0 else 0
        agent_throughputs = self.agent_success / self.time_steps
        sum_x = np.sum(agent_throughputs)
        sum_x2 = np.sum(agent_throughputs ** 2)
        jain_fairness = (sum_x ** 2) / (self.num_agents * sum_x2) if sum_x2 > 0 else 0

        text = f"✅ Simulation Complete\n"
        text += f"Collisions: {self.collisions}   Successful: {self.successful}   Overall Throughput: {overall_throughput:.2f} ({self.successful}/{total_slots})\n"
        text += "Agent Throughputs: "
        for i, tp in enumerate(agent_throughputs, 1):
            text += f"Agent{i}: {tp:.2f} "
        text += f"\nJain's Fairness: {jain_fairness:.2f}"
        self.stats_label.config(text=text)
        if self.visual_enabled:
            self.play_button.config(state="disabled")
            self.next_button.config(state="disabled")

    def toggle_run(self):
        if not self.running:
            self.running = True
            self.play_button.config(text="⏸ Pause")
            threading.Thread(target=self.auto_run, daemon=True).start()
        else:
            self.running = False
            self.play_button.config(text="▶ Auto Run")

    def auto_run(self):
        while self.running and self.t < self.time_steps:
            self.master.after(0, self.next_step)
            time.sleep(0.5)  # Add a small delay for visualization
        if self.t >= self.time_steps:
            self.running = False
            self.master.after(0, lambda: self.play_button.config(text="▶ Auto Run"))

if __name__ == "__main__":
    root = tk.Tk()
    WirelessSimulatorApp(root)
    root.mainloop()