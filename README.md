# Simulation of Dynamic Wireless Allocation

## 🛰️ Project Overview
This project simulates real-world Wi-Fi network conditions to study and design **adaptive algorithms** for **dynamic wireless resource allocation**.  
The simulation environment models realistic network behavior, interference, and agent dynamics—providing a testbed for evaluating algorithmic performance under varying network conditions.

### 🎯 Key Features
- Models realistic **multi-band Wi-Fi communication**.
- Includes **collision detection** and retransmission logic.
- Simulates **probabilistic transmission behavior** (non-deterministic agent decisions).
- Supports **customizable parameters** for flexible experimentation.
- Enables testing of **adaptive decentralized algorithms** for wireless scheduling.

---

## ⚙️ Simulation Details

### Agents
- Represent devices on a Wi-Fi network (e.g., laptops, phones, IoT sensors).
- Each agent independently decides **whether to transmit** and **on which frequency band** each time step.

### Time Steps
- Each simulation step corresponds to approximately **20 microseconds**.
- During each step:
  - Agents may choose to transmit on one of the available frequency bands.
  - If multiple agents transmit on the same band simultaneously, a **collision** occurs, and **no packets are successfully delivered**.
  - Agents involved in collisions must **reschedule and retransmit** later.

### Probabilistic Transmission Model
To mimic realistic, non-constant network behavior:
- When an agent **transmits**, it has:
  - **80% chance** to transmit again next step.
  - **20% chance** to remain idle.
- When an agent **does not transmit**, it has:
  - **80% chance** to stay idle.
  - **20% chance** to start transmitting.

This probabilistic rule introduces variability, preventing deterministic transmission patterns and making the simulation closer to real-world Wi-Fi conditions.

---

## 🧩 Configurable Parameters

Users can adjust the following parameters to experiment with different network conditions:

| Parameter | Description |
|------------|--------------|
| `num_agents` | Number of Wi-Fi devices (agents) in the simulation. |
| `num_bands` | Number of available frequency bands for transmission. |
| `num_steps` | Total number of time steps to simulate. |
| `algorithm` | Choice of adaptive allocation algorithm (centralized or decentralized). |

---

## 🎓 Project Goal

The primary goal is to **design and evaluate adaptive algorithms** that dynamically allocate network resources—such as time slots and frequency bands—across agents to optimize:

- **Throughput** — maximize total successful transmissions.  
- **Fairness** — prevent any single agent from being starved of bandwidth.  
- **Efficiency** — minimize collisions and wasted transmissions.

---

## 🧠 Algorithmic Challenge

Each agent must make decisions **independently**, based only on **local feedback** (success/failure of recent transmissions).  
No agent has global knowledge of the network state.

This **decentralized constraint** closely mimics real-world Wi-Fi behavior, where each device must learn and adapt dynamically.  
In contrast, a **centralized algorithm** (with full global knowledge) would be straightforward but unrealistic for most wireless environments.

The challenge:  
> Achieve near-optimal network performance **without centralized control or shared state**.

---

## 🚀 Example Use Cases
- Benchmarking adaptive MAC-layer algorithms.
- Studying the effects of probabilistic transmission in distributed networks.
- Teaching or research on decentralized wireless coordination.
- Experimenting with reinforcement learning–based scheduling strategies.

---

## 🧪 Future Extensions
- Integration with **reinforcement learning** or **multi-agent RL** frameworks.
- Visualization of agent-band-time allocations.
- Support for **variable transmission ranges** and **signal interference models**.
- Comparison between **centralized** and **decentralized** algorithms.

---

## 👤 Author
**Rohun Gargya** and **Logan Mappes**

---

