# 2x2 Gridworld: Two Iterations of Value Iteration (No Code)

## Given
- **States**: $S = \{s_1, s_2, s_3, s_4\}$
- **Actions**: $A = \{\text{up}, \text{down}, \text{left}, \text{right}\}$
- **Transitions**: deterministic if the move is valid; otherwise the agent stays in the same state ($s' = s$).
- **Rewards (state-based, same for all actions)**:
  - $R(s_1)=5$
  - $R(s_2)=10$
  - $R(s_3)=1$
  - $R(s_4)=2$
- **Discount factor**: $(\gamma \in [0.9, 1.0])$
- **Terminal state assumption**: $s_4$ is terminal (bottom-right). Once reached, the episode ends.
  - Terminal handling: $V(s_4)=R(s_4)=2$

### Grid layout
```text
s1 (top-left)    s2 (top-right)
s3 (bottom-left) s4 (bottom-right, terminal)
```

---

## Value Iteration Update Rule

For non-terminal states:
- $V_{k+1}(s) = R(s) + \gamma \max_{a \in A} V_k(s')$

For the terminal state:
- $V_{k+1}(s_4) = R(s_4) = 2$

---

## Iteration 1

### 1) Initialize $V_0$
- $V_0(s_1)=0,\ V_0(s_2)=0,\ V_0(s_3)=0,\ V_0(s_4)=0$

### 2) Compute $V_1$ from $V_0$
Since all $V_0(\cdot)=0$, the future term is zero in the first update.

- $V_1(s_1)=R(s_1)+\gamma\max_a V_0(s') = 5 + \gamma \cdot 0 = 5$
- $V_1(s_2)=10 + \gamma \cdot 0 = 10$
- $V_1(s_3)=1 + \gamma \cdot 0 = 1$
- $V_1(s_4)=2$ (terminal)

### 3) Values after Iteration 1
- $V_1(s_1)=5$
- $V_1(s_2)=10$
- $V_1(s_3)=1$
- $V_1(s_4)=2$

### (Optional) Greedy policy after Iteration 1
Pick the action that leads to the next state with the highest $V_1$.

- From $s_1$: best is **right** (to $s_2$)
- From $s_2$: best is **up** or **right** (both stay in $s_2$)
- From $s_3$: best is **up** (to $s_1$)
- $s_4$ is terminal

---

## Iteration 2

### Valid next states from each state
- From $s_1$: up $\to s_1$, left $\to s_1$, down $\to s_3$, right $\to s_2$
- From $s_2$: up $\to s_2$, right $\to s_2$, left $\to s_1$, down $\to s_4$
- From $s_3$: down $\to s_3$, left $\to s_3$, up $\to s_1$, right $\to s_4$
- $s_4$ terminal

### Compute $V_2$ using $V_1$

#### $V_2(s_1)$
- Next-state values: $V_1(s_1)=5$, $V_1(s_3)=1$, $V_1(s_2)=10$
- $\max = 10$
- $V_2(s_1) = 5 + \gamma \cdot 10 = 5 + 10\gamma$

#### $V_2(s_2)$
- Next-state values: $V_1(s_2)=10$, $V_1(s_1)=5$, $V_1(s_4)=2$
- $\max = 10$
- $V_2(s_2) = 10 + \gamma \cdot 10 = 10 + 10\gamma$

#### $V_2(s_3)$
- Next-state values: $V_1(s_3)=1$, $V_1(s_1)=5$, $V_1(s_4)=2$
- $\max = 5$
- $V_2(s_3) = 1 + \gamma \cdot 5 = 1 + 5\gamma$

#### $V_2(s_4)$
- Terminal: $V_2(s_4)=2$

---

## Final values after two iterations
- $V_2(s_1)=5 + 10\gamma$
- $V_2(s_2)=10 + 10\gamma$
- $V_2(s_3)=1 + 5\gamma$
- $V_2(s_4)=2$

### Example (if $\gamma=0.9$)
- $V_2(s_1)=5 + 10(0.9)=14$
- $V_2(s_2)=10 + 10(0.9)=19$
- $V_2(s_3)=1 + 5(0.9)=5.5$
- $V_2(s_4)=2$
