# RL_Programming

Solutions for four Reinforcement Learning assignment problems for CSCN8020.  
The file `CSCN8020_Assignment1.pdf` contains the official problem statements.

## Repository contents

- `CSCN8020_Assignment1.pdf`  
  Assignment handout with all problem statements.

- `solutions/`  
  Final, submission-ready writeups in Markdown:
  - `Problem_Solution.md` (overview or combined submission, if used)
  - `Problem1_Solution.md` (Pick-and-place robot MDP design)
  - `Problem2_Solution.md` (2x2 gridworld value iteration, 2 iterations)
  - `Problem4_Solution.md` (Off-policy Monte Carlo with importance sampling)

- `code/`  
  Python implementations used to generate results, logs, and plots:
  - `Problem3_Code.py` (5x5 gridworld value iteration and variations)
  - `Problem4_Code.py` (off-policy Monte Carlo with importance sampling)

- `logs/`  
  CSV logs produced by the scripts (useful for debugging, reporting, and plotting):
  - `vi_standard_log.csv`
  - `vi_inplace_log.csv`
  - `vi_convergence_log.csv`
  - `mc_convergence_log.csv`

- `images/`  
  Figures referenced by the Markdown solutions:
  - `mdp1.png`, `mdp2.png` (MDP diagrams for Problem 1)
  - `vi_convergence.png`, `convergence_plot.png` (Value Iteration convergence)
  - `mc_convergence.png` (Monte Carlo convergence)

## Problems covered

### Problem 1: Pick-and-Place Robot (MDP design)
Designs the task as an MDP by specifying:
- State representation (including joint positions and velocities)
- Continuous action space suitable for motor-level control
- Transition dynamics (simulated physics and stochasticity)
- Reward design to encourage fast, smooth, and safe pick-and-place behavior

See: `solutions/Problem1_Solution.md`

### Problem 2: 2x2 Gridworld (Value Iteration, 2 iterations)
Performs two iterations of Value Iteration for a 2x2 gridworld with state-based rewards and wall-bounce transitions.

See: `solutions/Problem2_Solution.md`

### Problem 3: 5x5 Gridworld (Value Iteration and in-place variation)
Implements standard (synchronous) Value Iteration and in-place (Gauss-Seidel style) Value Iteration. Produces:
- Optimal value function and greedy policy tables
- Convergence logs (CSV)
- Convergence plots (PNG)

See:
- Code: `code/Problem3_Code.py`
- Outputs: `logs/`, `images/`

### Problem 4: Off-policy Monte Carlo with Importance Sampling
Implements off-policy Monte Carlo using a fixed behavior policy to generate episodes and a greedy target policy updated via importance sampling. Produces:
- Estimated value function
- Convergence logs and plots
- Comparison discussion versus Value Iteration

See:
- Code: `code/Problem4_Code.py`
- Writeup: `solutions/Problem4_Solution.md`
- Outputs: `logs/`, `images/`

## How to run the code locally

### 1) Create and activate a virtual environment

Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
This repository uses only common scientific Python packages.

```bash
python -m pip install -U pip
pip install numpy matplotlib
```

### 3) Run Problem 3
```bash
python code/Problem3_Code.py
```

Expected outputs:
- CSV logs written to `logs/`
- Plots written to `images/`
- Console prints of value and policy tables

### 4) Run Problem 4
```bash
python code/Problem4_Code.py
```

Expected outputs:
- Monte Carlo logs written to `logs/`
- Plots written to `images/`
- Console prints of estimated values and policy

## Notes

- The Markdown writeups in `solutions/` are intended to be the primary submission artifacts.
- The `logs/` and `images/` folders contain reproducible evidence of computation (tables, convergence curves, and debugging traces).
- If you reorganize file paths, update image links inside the Markdown files accordingly.

## License
This repository is provided for academic coursework and personal learning. If you plan to reuse any part in another context, follow your course academic integrity rules.
