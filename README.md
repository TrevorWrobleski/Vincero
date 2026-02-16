# Project Vincero 

# Monopoly POMDP: A Bayesian Simulation Engine for Monopoly Win Probabilities

A rules-exact Monopoly simulation built on a **Partially Observable Markov Decision Process (POMDP)** architecture, with **Monte Carlo win probability estimation** and an optional **PPO reinforcement learning** agent. Designed for statistical research into the emergent economics of Monopoly.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)


---

## Motivation

Monopoly is a great way to ...add some energy... to an evening with family. The system presents a **constrained resource markets** (32 houses, 12 hotels), **information asymmetry** (hidden card decks), **cascading bankruptcy events**, and **nonlinear rent curves** that create tipping points. Most Monopoly simulators treat it as a fully observable, single-agent optimization problem. Here, we take a different approach.

We model Monopoly as a **POMDP** — a decision process where agents cannot see the full game state. The deck order is hidden. Opponent willingness to trade is a latent variable. The agent must reason under uncertainty, as a human player does. This architecture lets us ask questions that simpler models cannot:

- **When does a game become unwinnable?** At what point does one player's advantage become statistically irreversible?
- **What is the real value of a property?** Expected lifetime rent collected — which depends on landing frequency, building potential, and position relative to Jail.
- **Does going first actually matter?** And by how much, across different player counts?
- **What is the probability of winning?** Given specific property and cash holdings, what is an individual player's chance of winning? 

## Key Results

Results from a 10,000-game baseline simulation with 4 heuristic players:

| Metric | Value |
|---|---|
| Avg game length | ~170 turns total (~42 per player) |
| Player 0 (first mover) win rate | ~31% |
| Player 1–3 win rate | ~22–24% each |

**First-mover advantage is real and measurable.** Going first yields roughly a 6 percentage-point edge over the fair baseline of 25%. This is consistent with published Monopoly analysis and emerges naturally from the simulation without being hardcoded — the first player simply lands on unowned properties first, completes color sets earlier, and begins collecting building rent sooner.

**The Bayesian win estimator captures momentum shifts.** In a traced game, a player's estimated win probability rose from 0.27 at turn 20 to 0.93 by turn 120 as they accumulated buildings on high-traffic properties. The tipping point— where one player's advantage becomes effectively irreversible — typically occurs around turns 80–120, when developed monopolies begin generating rents that exceed opponents' ability to recover via GO salary.

**Buildings are the game.** Without buildings, base rents are trivially small (Mediterranean: $2, Boardwalk: $50). The GO salary of $200 per lap easily absorbs these costs, creating an economic equilibrium where no one ever goes bankrupt. The simulation only produces realistic game lengths when players actively build houses and hotels — confirming that **monopoly formation and development is the core economic engine**, not property collection alone.

**Aggression is a winning strategy.** Sensitivity analysis across the risk profile parameter θ (ranging from 0.1 to 1.0) shows a clear positive correlation between aggressiveness and win rate. Players with high θ values — who keep smaller cash reserves, buy more readily, build earlier, and pay bail to stay active — consistently outperform conservative players. This aligns with competitive Monopoly wisdom: the game rewards players who convert cash into rent-generating assets as quickly as possible, because liquid cash earns nothing while buildings compound. The player who develops first forces opponents into a defensive cycle of paying rent, mortgaging to survive, and falling further behind.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  TrueState                       │
│  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  ObservableState  │  │    HiddenState        │ │
│  │  · Player cash    │  │  · Deck permutations  │ │
│  │  · Positions      │  │  · Opponent profiles   │ │
│  │  · Property       │  │    (θ ∈ [0,1])        │ │
│  │    ownership      │  │                        │ │
│  │  · Building       │  └──────────────────────┘ │
│  │    counts         │                            │
│  │  · Deck prob      │                            │
│  │    vectors        │                            │
│  └──────────────────┘                            │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────┐    ┌────────────────────────┐
│    MonopolyPOMDP        │    │ WinLikelihoodEstimator  │
│    (Game Engine)        │───▶│ (Monte Carlo Rollouts)  │
│  · Strict rules impl   │    │ · Sample N deck perms   │
│  · Heuristic agents     │    │ · Sample K profiles     │
│  · Trade/auction logic  │    │ · Rollout to terminal   │
└─────────────────────────┘    │ · Output: P(win|state)  │
              │                 └────────────────────────┘
              ▼
┌─────────────────────────┐
│    MonopolyGymEnv       │
│  (Gymnasium Wrapper)    │
│  · 246-dim observation  │
│  · 90 discrete actions  │
│  · PPO via SB3          │
└─────────────────────────┘
```

### Files
I placed these within a Jupyter notebook for convenience, but they are technically seperate elements that could be used individually.

| File | Purpose |
|---|---|
| `monopoly_pomdp.py` | Core engine: board, cards, state, rules, POMDP, heuristic agents |
| `monopoly_rl.py` | Win estimator, Gymnasium environment, PPO training/evaluation |
| `monopoly_scenario.py` | Scenario builder for custom board states and parameter sweeps |

*Notebook:** [`Project_Vincero_Mono.ipynb`](Project_Vincero_Mono.ipynb)

## Installation

```bash
git clone https://github.com/TrevorWrobleski/Vincero.git
cd Vincero
pip install numpy matplotlib seaborn gymnasium stable-baselines3
```

Then open the notebook:

```bash
jupyter notebook Project_Vincero_Mono.ipynb
```

## Quick Start

Run the notebook cells in order. The early cells define the core engine classes; later cells use them for analysis. All examples below assume you have already run the engine cells.

### Run a baseline experiment

```python
wins = {i: 0 for i in range(4)}
for _ in range(1000):
    game = MonopolyPOMDP(num_players=4)
    winner = game.simulate_game(max_turns=2000)
    wins[winner] = wins.get(winner, 0) + 1

for i in range(4):
    print(f"Player {i}: {wins[i]/10:.1f}%")
```

### Estimate win probability from a custom board state

```python
sb = ScenarioBuilder(num_players=3)

sb.set_player(0,
    cash=400,
    properties=["boardwalk", "park place", "reading rr"],
    houses={"boardwalk": 5, "park place": 5},  # 5 = hotel
)

sb.set_player(1,
    cash=600,
    properties=["st james", "tennessee", "new york", "b&o"],
    houses={"st james": 3, "tennessee": 3, "new york": 3},
)

sb.set_player(2,
    cash=1800,
    properties=["illinois", "kentucky", "electric"],
)

sb.print_scenario()
probs = sb.estimate_win_probabilities(num_deck_samples=50, num_profile_samples=10)
# → Player 0: 38.2%, Player 1: 45.6%, Player 2: 16.2%
```

### One-liner estimation

```python
quick_estimate([
    {"cash": 200, "properties": ["boardwalk", "park place"],
     "houses": {"boardwalk": 5, "park place": 5}},
    {"cash": 1000, "properties": ["st james", "tennessee", "new york"],
     "houses": {"st james": 3, "tennessee": 3, "new york": 3}},
])
```

### Train a reinforcement learning agent

```python
model, env = train_ppo(total_timesteps=200_000, seed=2629)
evaluate_agent(model, env, n_games=500)
```

## What You Can Configure

### Game Parameters

| Parameter | Where | Default | What it controls |
|---|---|---|---|
| `num_players` | `MonopolyPOMDP()` | 4 | Number of players (2–8) |
| `seed` | `MonopolyPOMDP()` | 2629 | RNG seed for reproducibility |
| `max_turns` | `simulate_game()` | 2000 | Turn cap before declaring a draw |

### Scenario Builder — Custom Board States

The `ScenarioBuilder` accepts any combination of:

| Setting | Example | Notes |
|---|---|---|
| `cash` | `cash=800` | Any non-negative integer |
| `properties` | `properties=["boardwalk", "b&o"]` | Names or board positions (0–39) |
| `houses` | `houses={"boardwalk": 3}` | 0–4 for houses, 5 for hotel. Must own full color set. Even building rule enforced. |
| `mortgaged` | `mortgaged=["reading rr"]` | Cannot have buildings in the same color set |
| `in_jail` | `in_jail=True` | Player starts in jail |
| `jail_turns` | `jail_turns=2` | How many turns already spent in jail (0–3) |
| `has_goojf_chance` | `has_goojf_chance=True` | Holds Chance "Get Out of Jail Free" card |
| `has_goojf_cc` | `has_goojf_cc=True` | Holds Community Chest GOOJF card |
| `is_bankrupt` | `is_bankrupt=True` | Player is already eliminated |

Property names are fuzzy-matched. You can use `"boardwalk"`, `"park place"`, `"park"`, `"b&o"`, `"reading rr"`, `"electric"`, `"illinois"`, `"st james"`, etc. Call `list_properties()` for the full reference table.

### Win Probability Estimator

| Parameter | Default | Effect |
|---|---|---|
| `num_deck_samples` | 30 | Deck permutations sampled. More = smoother estimates. |
| `num_profile_samples` | 10 | Opponent behavior profiles per deck. More = captures behavioral uncertainty. |
| `max_rollout_turns` | 500 | How far forward each simulated game runs. |
| `seed` | None | Reproducibility. |

**Accuracy guide:** Total rollouts = `deck_samples × profile_samples`. At 300 rollouts (the default), expect estimates within ±5 percentage points. For tighter confidence, use 1000+ rollouts (`num_deck_samples=100, num_profile_samples=10`).

### Parameter Sweeps

```python
# How does Player 0's cash affect their win probability?
results = sb.sweep_parameter(
    player_idx=0,
    parameter="cash",
    values=[100, 300, 500, 800, 1200, 2000],
    samples=30
)
```

### PPO Reinforcement Learning

| Parameter | Default | Notes |
|---|---|---|
| `total_timesteps` | 100,000 | Training steps. 50k for sanity check, 500k+ for real training. |
| `focus_player` | 0 | Which player the RL agent controls. Others use heuristics. |
| `learning_rate` | 3e-4 | Standard PPO learning rate. |
| `n_steps` | 2048 | Rollout buffer size per update. |
| `ent_coef` | 0.01 | Entropy bonus for exploration. |

### Heuristic Agent Behavior

Each non-RL player has a hidden **risk profile** θ ∈ [0, 1] that controls:

| Behavior | Low θ (conservative) | High θ (aggressive) |
|---|---|---|
| Cash reserve when buying | Keeps ~$400 back | Keeps ~$150 back |
| Building eagerness | Builds with large reserve | Builds down to reserve |
| Auction bidding | Bids up to ~50% of price | Bids up to ~100%+ for set completion |
| Jail strategy | Tries to roll doubles | Pays bail early to stay active |
| Trade acceptance | Demands higher premium | Accepts lower premium |

## Limitations

### Simplifications in the Heuristic Agents

- **Trading is limited.** The heuristic trade logic only handles simple "buy the missing piece" deals — cash for a single property. It does not negotiate multi-property swaps, three-way trades, or trades involving GOOJF cards. This means color sets take longer to form than in games between experienced humans, and some sets may never form if multiple players each hold pieces.
- **No housing shortage exploitation.** A well-known expert strategy is to build exactly 4 houses on each property and never upgrade to hotels, deliberately depleting the house supply to block opponents. The heuristic agents do not do this — they always upgrade to hotels when possible.
- **No strategic mortgaging.** Players only mortgage reactively (when they cannot pay a debt), never proactively to fund an aggressive building campaign on a different color set.
- **Auction intelligence is basic.** Players bid up to a valuation based on their profile and set completion status, but they do not reason about other players' cash, needs, or bluffing.

### Simplifications in the Rules Engine

- **Rent collection is automatic.** Per the official rules, an owner must *ask* for rent before the next player rolls, or they forfeit it. The simulation always collects rent. This slightly advantages property owners compared to real play.
- **Turn order in auctions is not strict.** The rules say "you don't need to follow turn order" for auctions. The simulation models this as sequential evaluation with randomized bid increments, which approximates but does not perfectly replicate the chaotic real-time bidding of physical play.
- **No time-limited games.** Some house rules and tournament formats end the game after a fixed time and count assets. The simulation only supports turn-count limits.

### Limitations of the Win Estimator

- **Rollout quality depends on heuristic accuracy.** The Monte Carlo estimator plays games forward using heuristic agents. If the heuristics make systematically different decisions than real players (or than the RL agent), the probability estimates will be biased accordingly.
- **Computational cost scales linearly.** Each probability estimate requires hundreds of full game rollouts. At 300 rollouts, a single estimate takes ~0.5–1 second. Sweeping many parameter values or taking frequent snapshots during a game can be slow.
- **Opponent profiles are sampled uniformly.** The estimator samples θ from a uniform distribution. If you know an opponent is aggressive or conservative, there is no interface to condition on that (yet).

### Limitations of the RL Agent

- **Single decision per step.** The Gymnasium wrapper gives the agent one action per turn step. In reality, a player might need to make several decisions in sequence (roll, buy, build on multiple properties, trade). The current action space is a simplification.
- **No multi-agent learning.** Only one player is controlled by the RL agent; all others use fixed heuristics. The agent learns to exploit heuristic tendencies, not to play against adaptive opponents.
- **Observation space is hand-engineered.** The 246-dimensional state vector was designed manually. A learned representation (e.g., via attention over property features) might capture strategic structure more effectively.

### Adversarial and Metagame Strategies

The heuristic agents play to maximize their own position — they buy properties that benefit them, build when they can afford to, and trade when it completes their own color sets. They do not engage in **adversarial metagame strategies** that are technically legal but motivated by *blocking opponents* rather than directly advancing their own position. Examples include:

- **Kingmaking**: A losing player deliberately making trades or decisions that advantage one specific opponent over another, effectively choosing the winner.
- **Spite blocking**: Refusing to trade a property not because you need it, but solely to prevent an opponent from completing a set — even when the trade would be profitable for you.
- **Collusive or retaliatory deal structures**: "I'll sell you Park Place for $1 if you promise never to build on it," or refusing all future trades with a player who outbid you at auction.
- **Deliberate overbidding in auctions**: Driving up the price of a property you don't want, purely to drain an opponent's cash before a dangerous stretch of the board.
- **Strategic bankruptcy targeting**: Accumulating debt to a specific player intentionally, knowing that your bankruptcy will transfer properties to them and create a cascading advantage against a third player.

These behaviors are all permitted under the official rules and are common in competitive play between experienced humans. They represent a class of **socially mediated strategy** — decisions that depend on reading opponents' intentions, forming implicit alliances, and reasoning about second-order effects on the player ecosystem rather than just your own balance sheet. Their absence from the current model likely understates game variance and may bias win probability estimates toward players with strong board positions, since the simulation does not account for the ability of weaker players to coordinate (explicitly or implicitly) against a leader. Modeling these strategies is a meaningful direction for future work, potentially through multi-agent reinforcement learning with communication channels or theory-of-mind opponent models.

## Research Directions

This is a starting point. In addition to the adversarial and metagame strategies noted above, some questions worth pursuing include the following:

- **Card counting value**: Compare win rates of an agent that sees the deck probability vector vs. one that doesn't. How many percentage points is card counting worth?
- **Housing shortage strategies**: Modify the heuristic to never build hotels and measure the impact on opponents' development rates and win probabilities.
- **Optimal trading thresholds**: At what premium should you sell a property that completes an opponent's set? Sweep the trade acceptance threshold and find the Nash equilibrium.
- **Player count effects**: How does first-mover advantage, game length, and the probability of any monopoly forming change with 2 vs. 6 players?
- **Bankruptcy cascades**: When one player goes bankrupt to another, the creditor inherits properties. How often does this inheritance create a new monopoly that triggers further bankruptcies?
- **Self-play RL**: Replace the heuristic opponents with copies of the trained agent and train via self-play. Does a qualitatively different strategy emerge?
