# Figgie Agent

Self-play RL agent for the card game [Figgie](https://www.figgie.com/how-to-play)
(Jane Street), combining MCTS, neural policy/value networks, and belief-state
reasoning. See `DESIGN.md` for the full architecture.

## Status

M0 ✅: rule-complete simulator, order-book trading engine, random and heuristic
baseline agents (heuristic beats random ~+$80/round, ~67% win share), benchmark
harness, tests.

M1 ✅: exact card-flow belief tracker (`src/belief/`) with a calibration
harness (`src/eval/calibrate.py`). The exact strategy-free posterior is
mildly overconfident and weakly informative on current opponents (ECE ~0.02–0.08,
goal-suit argmax 39–41%). A behavioral-likelihood probe (`src/eval/probe.py`)
showed opponent buys carry almost no goal signal against the heuristic bots
(buy-goal alignment 0.31 vs 0.25 chance), so behavior conditioning is deferred
to P2; M2 builds on the strategy-free posterior. See `DESIGN.md` §10.

M2 ✅: determinized IS-MCTS (`src/agents/is_mcts.py`) over the exact belief,
with a depth-limited tree, UCB1, and a closed-form "hold" leaf value
(`cash − ante + 10·Σ hand·P(goal)`). Search is restricted to belief-supported
suits (`--buy-only --top-k`). Against a common random pool at 120 ticks it
beats the heuristic baseline ~+33/round (MCTS +110–115 vs heuristic +76–85,
0.85 vs 0.65 win share). Engine had two negative-state bugs (stale asks,
over-spent bids) that were fixed. See `DESIGN.md` §10.

M3 ✅: neural search agent + self-play training loop. `src/learning/` provides a
41-float obs encoder over a 21-action buy-only space (`obs.py`), an MLP
policy/value net (`net.py`), and AlphaZero-style training
(`selfplay.py`/`replay.py`/`trainer.py`). `src/agents/neural.py` runs the same
determinized IS-MCTS tree with PUCT selection on the learned policy prior and
net value leaves. After training, the neural agent beats the M2 agent by
~+13/round on the common random pool at 120 ticks (neural +123–127 vs M2
+110–115 vs heuristic +76–85). The policy head drives the edge; the value head
is near-constant (MSE ≈ irreducible outcome noise); see `DESIGN.md` §10 for
the calibration finding. Checkpoint:
`experiments/checkpoints/figgie_net.pt`.

M4 ✅: population-based training (DESIGN.md §8). `src/agents/pool.py` adds a
fast greedy `PolicyAgent` (net policy head only, no search) and a `SnapshotPool`
of bounded past-net snapshots. The trainer snapshots the net each iteration and
generates self-play games against a pool mixture (25% greedy self, 35% past
selves, 25% heuristic, 15% random), then checks **no collapse vs past selves**
(each current net beats every snapshot it trained against) and runs a final
**full-pool eval** (neural vs IS-MCTS + heuristic + random). Warm-started from
the M3 checkpoint, the M4 agent keeps the random-pool edge (+128.8/round, 0.97
win share at the M3 bench config vs M3's +123.3, 0.95) and beats the full bot
pool: neural +49.3/round (0.56 win share) vs ismcts +9.5, heuristic −8.8,
random −50.0; vs every past self it holds +33–60/round. Checkpoint:
`experiments/checkpoints/figgie_net_m4.pt`.

P1 ✅: trade-history features + opponent (multi-player) value head. The obs
encoder adds a last-8-trade window (`history_features`, 40 floats); the net adds
a history pathway whose output is a **zero-initialized residual** over the trunk
(exact warm start: history contributes nothing until it learns) and a **4-player
value head** (self first) whose final layer is zero-initialized so search starts
policy-prior-driven. A diagnostic confirmed the opponent value head is genuinely
informative; zeroing it collapses the random-pool result to +49.9/0.50 vs
+111/0.89 with it (M4's scalar head was near-constant). Random-pool bench at the
M3/M4 config (120r/120s/d5/seed7): +125.8/round, 0.94 win share ≈ M4's
+128.8/0.97 (parity; the first concat-combine attempt had regressed to +108.9).
Checkpoint: `experiments/checkpoints/figgie_net_p1.pt`.

P2 ✅: self-attention over trade history + learned behavior-conditioned belief.
The history pathway is single-head self-attention over the trade window
(`--history attn`); the net gains a 12-way `belief_head` over suit-count
assignments (zero-init). The learned belief is **decoupled**: the obs always
encodes the exact strategy-free posterior, and the belief head (trained by
cross-entropy on the true assignment in self-play, `--belief-w`) refines it into
a prior that re-weights MCTS world sampling inside the exact-consistent set;
it can never suggest impossible worlds. Random-pool bench (same config):
+127.75/0.95 vs P1's +125.8. Full-pool eval (40r/100s/d4): +29.75 with the
learned belief vs +24.50 without (a +5.25 delta; P1 measured +24.50). Headline
bench numbers remain within noise of M4, but the learned belief is the
measurable P2 win. Checkpoint: `experiments/checkpoints/figgie_net_p2.pt`.

P2b ✅: Bayes-combine the learned belief into the obs (`--bayes-combine`). At
decision time the agent feeds `P(a) ∝ P_exact(a) · P_learned(a)` to the obs and
to world sampling (pure eval-time change, no retraining). Random-pool bench
unchanged within noise; the mixed full-pool (40r/100s/d4, seed 12345) improves
+43.25/0.53 → +49.00/0.57; behavior conditioning only helps against
behavior-driven opponents, matching P2's finding.

P1v ✅: value-head recalibration. Re-trains P1 with `--value-w 1.0`
(`figgie_net_p1v.pt`): loss_v drops to ~0.02, random-pool evals hold +120–130
(parity), and the top calibration bin is now calibrated (pred +2.27 → actual
+2.31 vs P1's +1.76 → +2.28). `src/eval/value_calibrate.py` produces the affine
artifact `experiments/checkpoints/figgie_net_p1v_calib.json`.

P1b ✅: bundle quotes. The engine supports atomic two-suit `BUNDLE_BID` fills
(one buyer, two sellers at their own asks); the obs grows to 53 floats (old 41
unchanged) and the action space to 45. `load_partial` now grows both trunk
inputs and policy outputs so P2 warm-starts exactly. Trained
(`figgie_net_p1b.pt`), verified by 4 new engine invariants, but the **honest
negative result** is that the net over-posts non-crossing bundle bids (~63% of
turns, only 2/138 trades fill as bundles) and the mixed full-pool is +28.50 vs
P2's +43.25; the mechanism is correct but not used profitably.

P3 ✅: playable CLI. `src/play_cli.py` runs one 4-player round (120 ticks) where
you take seat 0 against the trained net (search), the heuristic bot, and a
random bot, with a curated legal-action menu and optional belief display.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
# Benchmark heuristic vs random agents (1000 rounds)
.venv/bin/python -m src.eval.bench --rounds 1000 --seed 7

# Belief-tracker calibration curves vs simulator ground truth
.venv/bin/python -m src.eval.calibrate --rounds 300 --mode heuristic --seed 5

# Behavioral-likelihood probe (buy-goal alignment of opponent buys)
.venv/bin/python -m src.eval.probe --rounds 300 --mode heuristic --seed 5

# M2: IS-MCTS vs 3 random opponents, full 120 ticks (bench ~5 min)
.venv/bin/python -m src.eval.bench --mcts --buy-only --top-k 2 \
    --rounds 120 --ticks 120 --sims 120 --depth 5 --opponents random --seed 7

# M3: trained neural agent vs 3 random opponents (bench ~10 min)
.venv/bin/python -m src.eval.bench --neural experiments/checkpoints/figgie_net.pt \
    --rounds 120 --ticks 120 --sims 120 --depth 5 --opponents random --seed 7

# M3: retrain the neural agent from scratch (CPU; ~40 min for the default run)
.venv/bin/python -m src.learning.trainer --iterations 6 --games 40 \
    --sims 50 --depth 4 --out experiments/checkpoints/figgie_net.pt

# M4: population training warm-started from M3 (CPU; ~25 min for the default run)
.venv/bin/python -m src.learning.trainer --iterations 4 --games 24 \
    --sims 40 --depth 3 --full-eval --init experiments/checkpoints/figgie_net.pt \
    --out experiments/checkpoints/figgie_net_m4.pt

# M4: full-pool eval (neural + ismcts + heuristic + random)
.venv/bin/python -m src.eval.bench --neural experiments/checkpoints/figgie_net_m4.pt \
    --rounds 120 --ticks 120 --sims 120 --depth 5 --opponents random --seed 7

# P1: train history features + multi-player value head from M4 (CPU ~25 min)
.venv/bin/python -m src.learning.trainer --iterations 5 --games 24 \
    --sims 40 --depth 3 --value-w 0.5 --init experiments/checkpoints/figgie_net_m4.pt \
    --out experiments/checkpoints/figgie_net_p1.pt

# P2: train self-attention history + learned belief from P1 (CPU ~35 min)
.venv/bin/python -m src.learning.trainer --iterations 5 --games 24 \
    --sims 40 --depth 3 --value-w 0.5 --belief-w 0.5 --history attn \
    --init experiments/checkpoints/figgie_net_p1.pt \
    --out experiments/checkpoints/figgie_net_p2.pt

# P2: bench the neural agent using its learned belief
.venv/bin/python -m src.eval.bench --neural experiments/checkpoints/figgie_net_p2.pt \
    --rounds 120 --ticks 120 --sims 120 --depth 5 --opponents random --seed 7 --learned-belief

# P2b: same, with the Bayes-combined posterior in the obs
.venv/bin/python -m src.eval.bench --neural experiments/checkpoints/figgie_net_p2.pt \
    --rounds 120 --ticks 120 --sims 120 --depth 5 --opponents random --seed 7 \
    --learned-belief --bayes-combine

# P1v: value-head recalibration artifact for a checkpoint
.venv/bin/python -m src.eval.value_calibrate --checkpoint experiments/checkpoints/figgie_net_p1v.pt \
    --rounds 20 --sims 40 --out experiments/checkpoints/figgie_net_p1v_calib.json

# P1b: train the bundle action space warm-started from P2 (CPU ~40 min)
.venv/bin/python -m src.learning.trainer --iterations 5 --games 24 \
    --sims 40 --depth 3 --value-w 0.5 --belief-w 0.5 --history attn \
    --init experiments/checkpoints/figgie_net_p2.pt \
    --out experiments/checkpoints/figgie_net_p1b.pt

# P3: play a round vs the trained net (P0 is you)
.venv/bin/python -m src.play_cli --checkpoint experiments/checkpoints/figgie_net_p2.pt --learned-belief

# Tests
.venv/bin/python -m pytest
```

## Layout

```
src/
  env/        game engine (deck, state, order-book trading, observations)
  agents/     random + heuristic baselines, IS-MCTS, neural search agent, pool
  belief/     exact belief tracker (assignment posterior) + particle worlds
  learning/   obs encoder, policy/value net, self-play, replay buffer, trainer
  eval/       benchmark, stats, calibration harness, behavior probe
tests/        engine invariants, payout conservation, agent comparison, belief
```
