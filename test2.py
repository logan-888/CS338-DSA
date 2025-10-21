import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
import math
import numpy as np
import matplotlib.pyplot as plt

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

        self.visual_var = tk.BooleanVar(value=False)
        self.visual_check = ttk.Checkbutton(form, text="Disable Visual Mode", variable=self.visual_var)
        self.visual_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        # Initial UI states
        self.toggle_prob_ui()

        ttk.Button(frame, text="Start Simulation", command=self.start_simulation).pack(pady=15)

    def toggle_prob_ui(self):
        if self.random_var.get():
            self.prob_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        else:
            self.prob_frame.grid_remove()

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

        self.collisions = 0
        self.successful = 0
        self.t = 0
        self.occupancy = np.zeros((self.time_steps, self.num_bands, self.num_agents), dtype=bool)
        self.colors = np.full((self.num_bands, self.time_steps), "#FFFFFF", dtype='<U7')
        self.agent_success = np.zeros(self.num_agents, dtype=int)
        self.success_per_step = np.zeros(self.time_steps, dtype=int)
        self.pending = np.zeros(self.num_agents, dtype=int)
        self.arrival_prob = 0.05

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
        # Packet arrivals
        for agent in range(self.num_agents):
            if random.random() < self.arrival_prob:
                self.pending[agent] += 1

        # Agent transmissions based on selected algorithm
        if self.selected_algo == "Random":
            for agent in range(self.num_agents):
                if self.pending[agent] > 0:
                    transmit = not self.random_transmission or random.random() < self.trans_prob
                    if transmit:
                        band = np.random.randint(0, self.num_bands)
                        self.occupancy[self.t, band, agent] = True
        elif self.selected_algo == "ML approach":
            self.ml_approach()
        elif self.selected_algo == "RL approach":
            self.rl_approach()

        # Resolve transmissions
        for b in range(self.num_bands):
            count = np.sum(self.occupancy[self.t, b])
            if count == 0:
                self.colors[b, self.t] = "#FFFFFF"
            elif count == 1:
                agent = np.argwhere(self.occupancy[self.t, b])[0, 0]
                self.colors[b, self.t] = "#00C853"  # bright green (success)
                self.successful += 1
                self.success_per_step[self.t] += 1
                self.agent_success[agent] += 1
                self.pending[agent] -= 1
                # Update learning for success
                if self.selected_algo == "ML approach":
                    self.eg_rewards[agent, b] += 1.0
                elif self.selected_algo == "RL approach":
                    pass  # Exp3 updates before choice
            else:
                self.colors[b, self.t] = "#D50000"  # bright red (collision)
                self.collisions += count

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
        Epsilon-Greedy with decaying epsilon for exploration (Level 1: Static Environment).
        Actions include bands and idle to enable coordination like alternating.
        """
        num_actions = self.num_bands + 1
        idle_idx = num_actions - 1

        if not hasattr(self, "eg_plays"):
            self.eg_plays = np.zeros((self.num_agents, num_actions), dtype=int)
            self.eg_rewards = np.zeros((self.num_agents, num_actions), dtype=float)

        eps = max(0.01, 1.0 / math.sqrt(self.t + 1))

        for agent in range(self.num_agents):
            if self.pending[agent] == 0:
                continue
            total_plays = np.sum(self.eg_plays[agent])
            if total_plays == 0:
                action = np.random.randint(0, num_actions)
            else:
                if random.random() < eps:
                    action = np.random.randint(0, num_actions)
                else:
                    avgs = np.divide(self.eg_rewards[agent], self.eg_plays[agent],
                                     out=np.full(num_actions, -np.inf),
                                     where=self.eg_plays[agent] != 0)
                    action = np.argmax(avgs)

            self.eg_plays[agent, action] += 1

            if action == idle_idx:
                continue  # idle

            band = action
            self.occupancy[self.t, band, agent] = True

    def rl_approach(self):
        """
        Exp3 with proper update before choice.
        """
        gamma_init = 0.2
        gamma_decay = 0.999
        min_gamma = 0.03

        num_actions = self.num_bands + 1
        idle_idx = num_actions - 1

        if not hasattr(self, "exp3_state"):
            self.exp3_state = np.ones((self.num_agents, num_actions))
            self.exp3_prev_actions = [None] * self.num_agents
            self.exp3_gamma = gamma_init
            self.gamma_decay = gamma_decay
            self.min_gamma = min_gamma
            self.num_actions = num_actions  # store for eta
            self.idle_idx = idle_idx

        # Update from previous timestep
        if self.t > 0:
            for agent in range(self.num_agents):
                prev = self.exp3_prev_actions[agent]
                if prev is None:
                    continue
                prev_action, prob_action = prev

                if prev_action == self.idle_idx:
                    reward = 0.0
                else:
                    count = np.sum(self.occupancy[self.t - 1, prev_action])
                    reward = 1.0 if count == 1 else 0.0

                x_hat = reward / max(prob_action, 1e-12)

                K = self.num_actions
                eta = self.exp3_gamma / K
                self.exp3_state[agent, prev_action] *= math.exp(eta * x_hat)
                self.exp3_state[agent, prev_action] = min(max(self.exp3_state[agent, prev_action], 1e-6), 1e6)

        # Decay gamma
        self.exp3_gamma = max(self.min_gamma, self.exp3_gamma * self.gamma_decay)

        # Choose actions for current timestep
        for agent in range(self.num_agents):
            if self.pending[agent] == 0:
                self.exp3_prev_actions[agent] = None
                continue
            weights = self.exp3_state[agent]
            total_w = np.sum(weights)
            if total_w <= 0:
                self.exp3_state[agent] = np.ones(self.num_actions)
                total_w = self.num_actions

            probs = (1 - self.exp3_gamma) * (weights / total_w) + (self.exp3_gamma / self.num_actions)

            action = np.random.choice(self.num_actions, p=probs)

            if action != self.idle_idx:
                band = action
                self.occupancy[self.t, band, agent] = True

            self.exp3_prev_actions[agent] = (action, probs[action])

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

        # Display throughput graph
        throughput_per_step = self.success_per_step / self.num_bands
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.time_steps), throughput_per_step)
        plt.xlabel('Timestep')
        plt.ylabel('Throughput (Successful / Bands)')
        plt.title('Throughput per Timestep')
        plt.grid(True)
        plt.show()

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