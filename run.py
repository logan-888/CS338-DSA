import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
import pprint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from dsa_env import DSAEnvironment

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
        self.algo_combo = ttk.Combobox(form, textvariable=self.algo_var, values=["Random", "RL approach"], state="readonly")
        self.algo_combo.grid(row=3, column=1, pady=5)

        self.random_var = tk.BooleanVar(value=False)
        self.random_check = ttk.Checkbutton(form, text="Enable Random Transmission", variable=self.random_var, command=self.toggle_prob_ui)
        self.random_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.prob_frame = ttk.Frame(form)
        ttk.Label(self.prob_frame, text="Transmission Probability (0-1):").grid(row=0, column=0, sticky="e", pady=5, padx=5)
        self.prob_entry = ttk.Entry(self.prob_frame, width=10)
        self.prob_entry.grid(row=0, column=1, pady=5)
        self.prob_entry.insert(0, "50")

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
                prob = float(self.prob_entry.get())
                if not 0 <= prob <= 1:
                    raise ValueError
                self.trans_prob = prob
            except ValueError:
                messagebox.showerror("Input Error", "Transmission probability must be between 0 and 1.")
                return
        else:
            self.trans_prob = 1.0  # always transmit if not random     


        self.collisions = 0
        self.successful = 0
        self.t = 0
        self.occupancy = [[[[] for _ in range(self.num_bands)] for _ in range(self.time_steps)]]
        self.colors = [["#FFFFFF" for _ in range(self.time_steps)] for _ in range(self.num_bands)]
        self.agent_success = [0] * self.num_agents


        self.env = DSAEnvironment(self.num_agents, self.num_bands, self.trans_prob)

        for widget in self.master.winfo_children():
            widget.destroy()

        # Enlarge the window for the simulation UI
        self.master.geometry("1600x900")

        self.create_sim_ui()

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

        self.stats_label = ttk.Label(self.master, text="", font=("Segoe UI", 12))
        self.stats_label.pack(pady=10)

        self.cell_size_x = 1300 / self.time_steps
        self.cell_size_y = 550 / self.num_bands
        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.num_bands):
            for j in range(self.time_steps):
                x0 = 50 + j * self.cell_size_x
                y0 = 30 + i * self.cell_size_y
                x1 = x0 + self.cell_size_x
                y1 = y0 + self.cell_size_y
                color = self.colors[i][j]
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#333", width=1.8)

                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                agents = self.occupancy[0][j][i]
                agent_str = ",".join(map(str, agents)) if agents else ""

                
                if agents:
                    text_color = "white" if color == "#D50000" else "black"
                    font_size = 11 if len(agent_str) < 10 else 9
                    self.canvas.create_text(mid_x, mid_y, text=agent_str, fill=text_color, font=("Segoe UI", font_size, "bold"))

        self.canvas.create_text(700, 15, text=f"Time Step: {self.t}/{self.time_steps}", font=("Segoe UI", 12, "bold"))



    def next_step(self):
        if self.t >= self.time_steps:
            self.finish_simulation()
            return

        # Agent transmissions based on selected algorithm
        if self.selected_algo == "Random":
            if self.random_transmission:
                for agent in range(self.num_agents):
                    if random.random() < self.trans_prob:
                        band = random.randint(0, self.num_bands - 1)
                        self.occupancy[0][self.t][band].append(agent + 1)
            else:
                for agent in range(self.num_agents):
                    band = random.randint(0, self.num_bands - 1)
                    self.occupancy[0][self.t][band].append(agent + 1)
        elif self.selected_algo == "RL approach":
            self.rl_approach()

        for b in range(self.num_bands):
            agents = self.occupancy[0][self.t][b]
            count = len(agents)
            
            if count == 0:
                self.colors[b][self.t] = "#FFD600"  # yellow (empty)
            elif count == 1:
                self.colors[b][self.t] = "#00C853"  # bright green (success)
                self.successful += 1
                agent = agents[0] - 1
                self.agent_success[agent] += 1
            else:
                self.colors[b][self.t] = "#D50000"  # bright red (collision)
                self.collisions += 1

        #pprint.pprint(self.occupancy)
        self.t += 1
        self.draw_grid()

        if not hasattr(self, "throughput_history"):
            self.throughput_history = []
        current_successes = sum(1 for b in range(self.num_bands) if len(self.occupancy[0][self.t - 1][b]) == 1)
        self.throughput_history.append(current_successes / self.num_bands)  # normalize per band




    def rl_approach(self):
        actions, rewards = self.env.step()
        for agent, act in enumerate(actions):
            if act < self.num_bands:
                self.occupancy[0][self.t][act].append(agent + 1)



    def finish_simulation(self):
        total_slots = self.time_steps * self.num_bands
        overall_throughput = self.successful / total_slots if total_slots > 0 else 0
        agent_throughputs = [s / self.time_steps if self.time_steps > 0 else 0 for s in self.agent_success]
        sum_x = sum(agent_throughputs)
        sum_x2 = sum(x ** 2 for x in agent_throughputs)
        jain_fairness = (sum_x ** 2) / (self.num_agents * sum_x2) if sum_x2 > 0 else 0

        text = f"✅ Simulation Complete\n"
        text += f"Collisions: {self.collisions}   Successful: {self.successful}   Overall Throughput: {overall_throughput:.2f} ({self.successful}/{total_slots})\n"
        text += "Agent Throughputs: "
        for i, tp in enumerate(agent_throughputs, 1):
            text += f"{tp:.2f},  "
        text += f"\nJain's Fairness: {jain_fairness:.2f}"
        self.stats_label.config(text=text)
        self.play_button.config(state="disabled")
        self.next_button.config(state="disabled")

        # Compute sliding window throughput
        window = 15
        if hasattr(self, "throughput_history") and len(self.throughput_history) > 0:
            history = np.array(self.throughput_history)
            kernel = np.ones(window) / window
            sliding_avg = np.convolve(history, kernel, mode="valid")

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(sliding_avg, color="#2196F3", linewidth=1.5)
            ax.set_title("Sliding-Window Throughput", fontsize=12)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"Throughput (avg over {window})")
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()

            # --- Create a new toplevel window for the graph ---
            plot_window = tk.Toplevel(self.master)
            plot_window.title("Throughput Graph")

            # Make window resizable & scrollable
            canvas_container = tk.Canvas(plot_window)
            canvas_container.pack(side="left", fill="both", expand=True)

            y_scroll = tk.Scrollbar(plot_window, orient="vertical", command=canvas_container.yview)
            y_scroll.pack(side="right", fill="y")
            canvas_container.configure(yscrollcommand=y_scroll.set)

            frame_inside = tk.Frame(canvas_container)
            canvas_container.create_window((0, 0), window=frame_inside, anchor="nw")

            # Embed Matplotlib figure
            graph_canvas = FigureCanvasTkAgg(fig, master=frame_inside)
            graph_canvas.draw()
            graph_widget = graph_canvas.get_tk_widget()
            graph_widget.pack(fill="both", expand=True, padx=10, pady=10)

            # Update scrollregion dynamically
            def _on_frame_configure(event):
                canvas_container.configure(scrollregion=canvas_container.bbox("all"))

            frame_inside.bind("<Configure>", _on_frame_configure)

            # Set reasonable minimum window size
            plot_window.geometry("900x400")

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
        if self.t >= self.time_steps:
            self.running = False
            self.master.after(0, lambda: self.play_button.config(text="▶ Auto Run"))
            self.finish_simulation()

if __name__ == "__main__":
    root = tk.Tk()
    WirelessSimulatorApp(root)
    root.mainloop()