# 2048 Deep Q-Network Agent

A fully autonomous Deep Reinforcement Learning agent trained to play the game 2048 using a custom PyTorch Convolutional Neural Network (CNN).

## Architecture & Technical Highlights

* **Deep Q-Network (DQN):** Utilizes a custom PyTorch CNN to process 16-layer one-hot encoded board states, extracting complex spatial hierarchies and tile configurations.
* **Action Masking:** Solved catastrophic policy collapse and infinite loop traps by filtering the neural network's Q-value predictions to strictly output physically valid environment transitions.
* **Strategic Reward Shaping:** Overcame the "greedy agent trap" by engineering custom reward heuristics—including empty-space multipliers and non-corner maximum tile penalties—to enforce long-term structural planning over immediate point gains.
* **Custom Physics Engine:** Built a lightweight 2048 environment from scratch using NumPy to handle matrix rotations, sliding mechanics, and state evaluation.

## Repository Structure

```text
├── src/
│   ├── agent.py        # DQN Agent, Memory Replay, and Action Masking logic
│   ├── environment.py  # 2048 Matrix Physics and Reward Shaping heuristics
│   ├── model.py        # PyTorch Convolutional Neural Network architecture
│   ├── train.py        # Main training loop with Epsilon-Greedy decay
│   ├── watch.py        # Terminal-based exhibition script
│   └── ui.py           # Real-time Pygame graphical visualization
├── 2048_brain.pth      # Pre-trained model weights (Max Tile: 512)
├── requirements.txt    # Project dependencies
└── README.md
```
Installation & Setup
Clone the repository:

```Bash
git clone [https://github.com/YOUR_USERNAME/2048-dqn.git](https://github.com/YOUR_USERNAME/2048-dqn.git)
cd 2048-dqn
```

Install the required dependencies:

```Bash
pip install -r requirements.txt
```

## Usage
Watch the Trained AI Play (Graphical UI)

You can instantly watch the pre-trained neural network play using the Pygame visualizer. It runs inference locally using the saved 2048_brain.pth weights.

```Bash
python src/ui.py
```

## Train from Scratch
To wipe the agent's memory and train a new model from absolute scratch, delete the 2048_brain.pth file and run the training engine:

```Bash
python src/train.py
```

## Results
The current iteration of the model successfully navigates the high-variance environment of 2048, successfully synthesizing 512 tiles entirely autonomously.
