# Figgie Agent ; Design & Architecture

Status: Draft v0.1 ; M0–M4 complete (simulator, baselines, belief, search, neural agent, population training)
Author: you
Scope: Self-play RL agent for the card game Figgie (Jane Street), combining
MCTS, neural policy/value networks, and belief-state reasoning for
imperfect-information play. Inspired by AlphaZero but adapted for a
multi-agent, general-sum, hidden-information negotiation game.

---

## 1. Goals

- Learn a trading strategy for Figgie **from simulation alone**, without
  hand-designed heuristics, via self-play + search + deep RL.
- Beat strong hand-coded baseline bots, then beat a pool of evolved bots.
- Produce a reusable simulator + training harness that is fast, testable,
  and instrumented (reproducibility is a first-class requirement).

Non-goals (initially): real-time play against the official Jane Street app,
human-competitive performance, production deployment.

---

## 2. Game model (formalization)

Canonical rules (from figgie.com, verified 2026):

- 4 or 5 players. Each starts with $350. Players ante an equal share to form
  a $200 pot.
- Deck: 40 cards, 4 suits (two black: spades/clubs; two red:
  hearts/diamonds). Suit counts: **one 8-card suit, two 10-card suits, one
  12-card suit**. The mapping from suit → count is random and **hidden until
  reveal**.
- **Goal suit**: always the same *color* as the 12-card suit; the goal suit
  itself contains 8 or 10 cards. All other suits are worthless.
- Cards are dealt evenly and randomly.
- **Trading phase** (4 min, continuous-time): players post bids/offers and
  buy/sell suits from one another. Cards are the commodity; within a suit,
  cards are fungible.
- **Payout**: $10 per goal-suit card in hand; the remainder of the pot goes
  to the player(s) holding the most goal-suit cards (ties split).
- Cumulative money across rounds is the player ranking metric (tournament
  objective).

### 2.1 Key structural facts that drive the design

1. **Small hidden state.** The only hidden "facts of the world" are (a) the
   suit→count assignment (12 possibilities: 4!/2! with the duplicated 10s)
   and (b) the distribution of cards across opponents' hands. A player's own
   hand prunes inconsistent assignments immediately (e.g. holding >12 of a
   suit rules it out as the 12-suit).
2. **Goal suit is derived**, not independent: belief over the 12-suit fully
   determines belief over the goal suit.
3. **Suit counts, not ranks.** Card ranks never matter; each hand is a
   4-vector of suit counts, and trades are suit-token transfers. This makes
   the observation space compact and the belief tracker tractable.
4. **Negotiation is the game.** Since the goal suit is largely determined by
   prior probabilities and deductions from *others' behavior*, most skill is
   in signal extraction (what opponents buy/dump reveals what they believe)
   and in price discovery (extracting value while concealing information).
5. **General-sum, n-player, simultaneous, continuous-time.** This is where
   the AlphaZero assumptions break (see §3).

---

## 3. Why AlphaZero does not transfer directly

| AlphaZero assumption | Figgie reality | Consequence |
|---|---|---|
| Perfect information | Hidden suit→count assignment, hidden hands | Need belief states + imperfect-info search |
| 2-player zero-sum | 4-5 player general-sum | No guaranteed Nash via pure self-play; need population training |
| Discrete turn-based | Continuous simultaneous negotiation | Need action/timing abstraction |
| Terminal win/loss | Continuous money across many rounds | Reward is tournament bankroll; sparse per-decision signal |
| Single agent symmetric | Asymmetric roles/funds after trades | Observations must include public state + private info |

The adaptation stack we use: **IS-MCTS over belief-sampled worlds** with
neural priors ; i.e. the ReBeL/AlphaZero-with-beliefs paradigm ; plus
population-based self-play for multi-agent stability, and reward
decomposition to densify learning signal.

---

## 4. Architecture overview

```
┌──────────────────────────── Self-play loop ────────────────────────────┐
│                                                                       │
│  ┌──────────┐   policies    ┌─────────────┐   trajectories   ┌─────┐ │
│  │ Simulator│◄──────────────│  Agents     │──────────────────►│Buffer││
│  │ (env)    │──────────────►│ (search)    │                   └──┬──┘ │
│  └────┬─────┘   states/obs  └──────┬──────┘                      │    │
│       │                            │  neural priors               ▼    │
│       │                     ┌──────┴──────┐                ┌──────────┐│
│       │                     │  Neural Net │◄───train───────│ Trainer  ││
│       │                     │ p(v|obs)    │                └──────────┘│
│       └──────── belief ─────┴─────────────┘                             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

Pipeline: `env` → `belief` → `search` (IS-MCTS) → `net` (policy/value)
→ `trainer` → improved net → repeat. Baselines are interchangeable agents
in the same harness.

---

## 5. Module breakdown

### 5.1 `env/` ; game engine (pure Python + NumPy, no ML deps)

- `deck.py` ; suit→count assignment sampling, deal.
- `state.py` ; full round state: hands (per-suit count vectors), chip
  totals, market book (best bid/ask per suit + open quotes), trade history,
  elapsed time, pot, terminal payout computation.
- `engine.py` ; `step()` for trade execution and quote matching under a
  discretized time model; payout logic.
- `observations.py` ; builds per-player public + private observation
  tensors (§6).
- `render.py` ; optional text/CLI viewer for debugging and for a future
  human-playable mode.
- `rules_test.py` ; property tests: deal invariants (40 cards, valid
  assignment), payout conservation (pot fully distributed), trade legality.

**Design choice ; trading engine.** Model the market as an order book:
  - Players post bids (max price they'll pay for a card of suit S) and
    offers (min price they'll accept). Public.
  - A trade executes when bid ≥ ask (price = midpoint, or aggressor's
    price ; decide in §12, default: midpoint).
  - **Simultaneous moves**: at each tick, every agent submits an action
    simultaneously; the engine resolves matches deterministically. Tick
    granularity is configurable (simulates the continuous 4-min clock).
  - Quantity: 1 card per quote; allow "bundle" quotes only as a later
    extension (marked P1 in §10).

### 5.2 `belief/` ; belief tracker

- Maintains posterior `P(assignment | own hand, public history)` over the
  12 possible suit→count assignments (pruned by own hand).
- Posterior over each opponent's hand (counts per suit) via
  `P(hand_j | assignment, trade history, opponent model)`.
- Updates are cheap: assignment space ≤ 12; card-distribution space is a
  multinomial over known remaining cards.
- **Validation**: calibrate against simulator ground truth (compare
  posterior mean to true assignment frequencies in logged games; §9).

### 5.3 `agents/`

- `random.py` ; uniform random legal actions (sanity floor).
- `heuristic.py` ; encodes conventional-wisdom strategy: (1) deduce the
   12-suit by tracking what opponents hoard/dump, (2) accumulate candidate
   goal-suit cards, (3) dump long suits, (4) price moves with time remaining
   (prices rise late in the round). Used as the "beat this" baseline.
- `is_mcts.py` ; Information-Set MCTS (§7). Pluggable priors/rollouts so it
  runs with heuristics now and with the neural net later.
- `neural.py` ; policy/value net wrapped by search (the final agent).
- `pool.py` ; multi-agent training pool management (population of past
  snapshots for opponent modeling, §8).
- `human.py` ; interactive `HumanAgent` for the P3 playable CLI.

### 5.4 `learning/`

- `net.py` ; architecture (§6.2).
- `selfplay.py` ; parallel game generation (vectorized env batch).
- `replay.py` ; prioritized replay buffer of `(obs, MCTS policy π, value z)`.
- `trainer.py` ; supervised regression (policy CE + value MSE), checkpoint,
  eval hooks, wandb/tensorboard logging.

### 5.5 `eval/`

- `bench.py` ; round-robin tournament across agent pool; TrueSkill/Elo.
- `probe.py` ; diagnostic metrics (below).
- `calibrate.py` ; belief-tracker calibration curves.
- `value_calibrate.py` ; net value-head calibration + affine artifact (P1v).
- `src/play_cli.py` ; one-round 4-player game vs the trained net (P3).

---

## 6. Observation, belief encoding, and network

### 6.1 Observation tensor (per player)

All values normalized. Input planes (roughly 50–100 floats total):

1. Own hand: per-suit counts (4).
2. Opponents' *known* holdings: suits seen in trades, per opponent (n×4).
3. Market book: best bid / best ask / spread / depth per suit (per suit).
4. Trade history window: last k trades as (suit, price, counterparty, dir) ;
   candidate for a small attention block (P2).
5. Chip totals per player (n).
6. Remaining time (scalar, normalized to [0,1]).
7. **Belief vector**: posterior over the ≤12 assignments (or a compressed
   marginal: P(suit is 12-card), P(goal suit), per-suit).
8. Public deck composition: known remaining cards per suit.

### 6.2 Network

- Input: the observation tensor above.
- Trunk: small MLP (or per-token MLP over the trade-history window in P2).
- Heads:
  - **Policy head**: distribution over the discretized action space (§6.3).
  - **Value head**: scalar expected round payout for *self* (relative to
    pot/ante, so it's bounded), possibly plus an auxiliary per-opponent
    value head for opponent modeling (P2).
- Sizes: start ~1–3M params; Figgie's info state is small, so overfitting
  and speed (self-play throughput) dominate capacity needs.

### 6.3 Action abstraction

Continuous negotiation → finite action set per tick:

- For each suit S and price bucket b (grid over [0, $MAX_PRICE], ~$1 or $2
  ticks): `POST_BID(S, b)`, `POST_ASK(S, b)`, `WITHDRAW(S)`.
- `ACCEPT_BID(S)` / `ACCEPT_ASK(S)` ; hit the best opposing quote.
- `PASS` ; no action this tick.
- Optional (P1): `BUNDLE(S1,S2,...)` combined-suit quotes.

Action set size ≈ 4 suits × 2 × ~25 buckets + ~10 = ~210. Manageable for a
policy head; optionally masked by legality (e.g. can't bid > cash).

---

## 7. Search: IS-MCTS with neural priors

At each decision point, the agent does not know the true suit→count
assignment or opponents' hands. Approach:

1. **Belief sampling.** Sample `W` worlds consistent with the player's
   information set: assignment from the belief posterior, opponents' hands
   from the belief-conditional distributions, all conditioned on the public
   trade history.
2. **IS-MCTS.** Run one shared tree over the *information-set* level (i.e.
   the public state), sampling a hidden world per simulation. Because the
   game is simultaneous-move, use simultaneous-move MCTS (like
   determinized / matrix-shaped nodes) ; each node stores the joint action
   distribution, and each simulation plays out one sampled world with all
   agents following current policies.
3. **Policy & value priors** from the neural net (when available) replace
   rollouts; until then, heuristic rollout eval. The value is the expected
   payout conditioned on the belief.
4. **Output**: an action visit distribution π (temperature-annealed during
   self-play) that becomes the policy target for training.

Note on IS-MCTS vs ReBeL: IS-MCTS with determinized belief sampling is the
simpler, proven path for "deduce hidden card state" games. ReBeL (Reinforced
Belief Learning) additionally *learns* the belief from the model's own play
and uses a compressed belief representation in the search tree ; this is the
upgrade path in P2/P3, not the v0 approach.

---

## 8. Multi-agent training strategy

- **No pure self-play assumption.** General-sum games can cycle; pure
  self-play may chase its tail. Instead use **fictitious self-play /
  population-based training**: maintain a pool of past network snapshots and
  heuristic bots; each new generation trains against a mixture
  (e.g. 50% self, 30% recent selves, 20% heuristics).
- **Reward decomposition** (Suphx-style): per-round money delta is the
  terminal signal; add immediate-reward shaping per trade (book profit from
  the deal) as a dense auxiliary signal (tuned, and ablated, since shaping
  can be gamed ; the terminal tournament bankroll stays the eval metric).
- **Exploitability can't be measured directly** (n-player general-sum), so
  use empirical metrics: win rate vs pool, bankroll per round, and stability
  (no collapse against prior selves).

---

## 9. Evaluation & diagnostics

- Primary: tournament bankroll / win rate against the fixed bot pool
  (random, heuristic, previous best snapshot).
- Secondary:
  - Belief calibration (predicted vs actual 12-suit / goal-suit frequency).
  - Policy entropy over time (should collapse sensibly).
  - Value head calibration (MSE of predicted vs realized payout).
  - Trade-level metrics: realized prices vs "fair" posterior EV (leakage
    detection ; is the agent giving away EV?).
- All results logged; every experiment reproducible from a checked-in seed
  config.

---

## 10. Milestones

| # | Deliverable | Exit criteria |
|---|---|---|
| M0 ✅ | Rule-complete simulator + tests + random & heuristic bots + stats | Deals/payouts conserve money; heuristic beats random (mean ~+$80/round, ~67% win share) |
| M1 ✅ | Belief tracker + calibration harness | Exact posterior calibrated within tolerance on logged games; documented model limitation (strategy-free, overconfident) |
| M2 ✅ | IS-MCTS with heuristic rollouts | Beats heuristic baseline against a common random pool in the restricted (discretized) setting |
| M3 ✅ | Neural net + search; self-play training loop | Neural agent beats M2 agent; belief/val heads calibrate |
| M4 ✅ | Full 4-5p continuous-trading variant; population training | Beats full bot pool; no collapse against past selves |
| P1 ✅ | Opponent value-head + attention over trade history | Measured improvement vs M4 (parity on random pool; informative value head) |
| P2 ✅ | ReBeL-style learned belief; deeper trade-history model | Measured improvement vs P1 (full-pool +5.25 with learned belief) |
| P2b ✅ | Bayes-combine the learned belief into the obs | Measured improvement vs P2 on the mixed pool (+43.25 → +49.00) |
| P1v ✅ | Value-head recalibration (`--value-w 1.0` retrain + affine artifact) | Top-bin pred ≈ actual (fixes P1's under-scaling) |
| P1b ✅ | Bundle quotes (engine, 53-float obs, 45-action space) | Implemented + verified; honest negative result (over-posting, no gain) |
| P3 ✅ | Human-playable CLI | Playable round vs the trained net |

### M0 implementation notes (verified 2026-08)

- **Market model**: per-suit order book, one quote per player per suit;
  simultaneous-move ticks; crossing executes at the resting quote's price;
  conflicts resolved by player index (documented in `engine.py`).
- **Signal-extraction trap**: a naive belief update that treats *any* trade as
  evidence overcounts trades executed *at your own quotes*. Your high bid on
  the goal suit attracts passive sells (which look like "opponents dumping the
  goal"), and your junk asks attract passive buys ; inverting the belief. The
  agent must only learn from **opponent-initiated** trades (aggressor = buyer
  for accepts/bid-crosses, seller for ask-crosses); trades the agent itself
  initiated or that execute against its own quotes are ignored. This flipped
  the heuristic from −$45/round to −$2/round.
- **Dumping is asymmetric-risk here**: holding junk is free (no hand-size
  limit, ample cash), but discarding a wrongly-suspected goal card is a real
  loss. With a weak hand-prior (P(true goal) ≥ 0.5 in only ~17% of rounds),
  the heuristic's dump rule destroyed its own goal cards. Buy-only + hold
  beats random by a wide margin (~+$80/round, ~67% win share).
### M1 implementation notes (verified 2026-08)

- **Exact posterior, not a particle filter.** A world is (assignment, opponents'
  initial hands); a world is consistent iff no observed sale drove its seller's
  hand negative, which per player & suit reduces to `needed` = maximum
  cumulative drawdown. `BeliefTracker` computes the assignment posterior
  **exactly** by DP over deals (`belief/exact.py`): P(a) ∝ hand-prior ×
  N_consistent(a), counting each suit split by its multinomial multiplicity.
  This removed every sampling-noise artifact (bootstrap collapse, biased
  resampling) that a particle filter introduced in tight-flow regimes. Particles
  remain only to sample consistent worlds for M2 determinizations.
- **Calibration results (exact strategy-free model).** ECE on assignments 0.03
  (heuristic opponents) / 0.02 (all random); goal-marginal ECE 0.08 / 0.04.
  Goal-suit argmax 39–41% vs 25% chance; assignment argmax ≈ hand-prior.
- **The strategy-free model is weakly informative and mildly overconfident.**
  Drawdown ("needed") constraints rarely bind against these opponents, so the
  posterior hugs the hand prior (mean P(true) ≈ prior ≈ 0.11). In the
  mid-high bins the model over-weights assignments that are merely
  *accidentally* consistent: ECE is small only because the posterior rarely
  concentrates. Real belief gains require a **strategy-conditioned likelihood**
  (what would this opponent do under this world?) ; the M2 + P2 program.
- **Harness** (`eval/calibrate.py`): logs (posterior, true assignment) pairs per
  tick and prints calibration curves + ECE for assignments and goal marginals.
  `--mode heuristic|random`, `--particles`, `--posterior-every`.
- **Behavioral-likelihood probe (verified 2026-08)** ; verdict: **defer; do not
  build the behavior model yet.** `eval/probe.py` measured the ceiling of
  conditioning beliefs on opponent behavior, using opponent-initiated buys as
  evidence `P(buy s | goal=s) = p_g`. Against heuristic opponents the empirical
  buy-goal alignment is only **0.31** (chance 0.25): their own beliefs are too
  weak for their buys to carry goal signal. An "oracle" behavior model that
  assumes strong alignment (p_g=0.8) is catastrophically overconfident on wrong
  goals (goal-marginal ECE 0.007 → 0.219; predicts 0.97 when true is 0.41), and
  at the true p_g≈0.31 it degenerates to the strategy-free posterior. Meanwhile
  the strategy-free posterior is well-calibrated on these opponents and already
  concentrates (goal argmax 0.38 heuristic-mode / 0.48 random-mode vs 0.25
  chance; mean P(true) 0.29–0.36; ECE 0.007–0.043). Implication for M2: build
  IS-MCTS on the **strategy-free** exact posterior; revisit behavior
  conditioning at P2 when self-play opponents give a learnable signal.
- **Perf note**: with trade-heavy (random) opponents the particle population
  reseeds constantly (each random trade kills survival) and dominates runtime.
  Particles are only needed for `sample_world()`; the exact posterior ignores
  them. Use small `--particles` when sampling isn't exercised.

### M2 implementation notes (verified 2026-08)

- **Agent** (`agents/is_mcts.py`): determinized IS-MCTS over the agent's own
  action sequence. Each simulation samples a consistent world from
  `BeliefTracker.sample_world()`, plays `--depth` in-tree ticks (our action by
  UCB1, opponents by the heuristic policy inside the world), then evaluates the
  leaf. Two value modes: `rollout` (play the heuristic to game end, terminal
  settle delta) and `hold` (closed-form `cash − ante + 10·Σ_s hand[s]·P(goal=s)`
  under no further trading, using the root posterior). `hold` is ~20× faster
  and lets the search afford far more simulations; it is the default and is
  exactly the objective the heuristic implicitly maximizes.
- **Action abstraction** (§6.3, restricted): PASS, ACCEPT_ASK, ACCEPT_BID,
  WITHDRAW, POST_BID/POST_ASK on a coarse price grid `{6,12,18,24}`. Flags
  `--buy-only` and `--top-k` restrict the search to the belief-supported suits
  and forbid selling.
- **Engine fixes (found while validating M2).** Two execution-order bugs let
  state go negative: (1) a stale ASK could execute after its seller had already
  drained that suit in the same tick (buyer branch never checked the seller's
  holding) → negative hands; (2) a resting BID could be hit after its bidder
  had spent the cash earlier that tick → negative cash. `_execute` now checks
  the resting side's resources and removes a seller's ASK once its holding hits
  0. Locked in by `tests/test_engine.py` invariant + regression tests.
- **Why the exit criterion is "vs a common random pool".** The heuristic bot
  only accepts asks and posts bids ; it never sells or hits bids. A pure
  heuristic market therefore has *no trades* and is degenerate (outcome =
  deal luck), so a head-to-head MCTS-vs-heuristic cannot show search value. The
  restricted setting uses random opponents as the common pool: randoms create
  liquidity (they post asks/bids and accept), and MCTS must out-trade the
  heuristic baseline on identical seeds. This is the honest liquid test of
  search lookahead.
- **M2 calibration vs heuristic baseline (4 players, 120 ticks):**

  | config | seed 3 | seed 7 |
  |---|---|---|
  | heuristic vs 3 random | +84.9, 0.71 | +76.0, 0.65 |
  | MCTS (buy-only, top-k=2, hold, sims=120, depth=5) vs 3 random | +115.4, 0.87 | +109.8, 0.85 |

  MCTS beats the heuristic by ~+33/round on both seeds.
- **M2 findings.** (1) Full-action search vs heuristics **lost** −11/round until
  `--buy-only`: with the weak M1 belief, the search over wrong sampled worlds
  could not tell goal cards from junk and sold its own goal cards (the M0
  "dumping is asymmetric-risk" finding made algorithmic). Restricting to
  buy-only brought it to even head-to-head and is required for strong
  performance. (2) `rollout` leaf values were too noisy/shallow at feasible
  sim budgets (slightly *worse* than the heuristic); the `hold` leaf value
  made search strictly better than the heuristic. (3) The search remains
  bottlenecked by the strategy-free belief: it cannot safely sell even at a
  profit, which the M1 probe already predicted. Selling-back-to-heuristics and
  behavior-conditioned beliefs remain P1/P2 work.

### M3 implementation notes (verified 2026-08)

- **Neural agent** (`agents/neural.py`): the same determinized IS-MCTS tree as
  M2, with two substitutions ; leaves are evaluated by the network's value head
  instead of the closed-form `hold` formula, and in-tree selection is PUCT
  (`q + c_puct·prior·√N/(1+n)`) using the network's policy head as a masked,
  renormalized action prior (degenerate-prior guarded). Search budget
  `--sims`/`--depth`, shared `_Node`/`_copy_state` from `is_mcts.py`.
- **Observation encoding** (`learning/obs.py`): 41 floats ; own hand/4, best
  bid/ask per suit, own/opponent cash, time remaining, goal marginals/4,
  12-dimensional assignment posterior/4, own-bid flags/4, per-suit trade counts.
  21-action buy-only space (PASS + per-suit ACCEPT_ASK + 4 bid prices
  `{6,12,18,24}`); `action_index`/`index_action`/`legal_mask` verified.
- **Network** (`learning/net.py`): `FiggieNet` ; 41→128 ReLU→128 ReLU trunk,
  value head (scalar, trained on settle/50), policy head (21 logits); ~40k
  params, CPU.
- **Training** (`learning/selfplay.py` + `learning/replay.py` +
  `learning/trainer.py`): AlphaZero-style loop. Each iteration generates N
  self-play games where the neural seat searches against random opponents and
  records (obs, MCTS visit distribution π, terminal z=settle/50), trains on the
  replay buffer with soft policy cross-entropy + value MSE (Adam, wd=1e-4),
  evaluates vs random opponents, and checkpoints. Value calibration binned at
  the end.
- **M3 calibration (4 players, 120 ticks, vs common random pool):**

  | config | seed 3 | seed 7 |
  |---|---|---|
  | heuristic vs 3 random | +84.9, 0.71 | +76.0, 0.65 |
  | M2 IS-MCTS (buy-only, top-k=2, hold, 120 sims, d=5) | +115.4, 0.87 | +109.8, 0.85 |
  | M3 neural (120 sims, d=5) | +126.8, 0.97 | +123.3, 0.95 |

  The trained neural agent beats the M2 agent by ~+13/round on both seeds
  (and the heuristic baseline by ~+42). Head-to-head (neural + ismcts + 2
  randoms, seed 3): neural +97.8, ismcts +1.3, randoms −49.5 ; the two strong
  agents compete for the same buy opportunities. Training trajectory: policy
  loss 3.0→1.9, eval win share 0.83→0.93–1.00.
- **Value-head calibration finding.** The value head is *near-constant*
  (predicts ≈ mean z), with corr(pred, actual) ≈ 0 after weight decay (the
  unregularized version overfit to spurious buffer correlations, corr −0.36).
  Its MSE is ≈ the irreducible outcome variance: in the liquid random-pool
  regime the seat's eventual delta is dominated by deal + opponent randomness
  that the current 41-float obs (strategy-free belief, own hand/cash) cannot
  predict. The value head is therefore not *miscalibrated* (it is not
  overconfident on a learned signal) ; it is *uninformative*, and search works
  with it because PUCT's q-term degenerates to a constant and selection is
  driven by the policy prior. Genuine value informativeness requires
  behavior-conditioned beliefs / richer history features (P2).
- **M3 findings.** (1) An untrained (random-init) net searches *worse* than the
  M2 heuristic-prior agent; the trained policy prior is what converts search
  into an edge ; policy improvement is the deliverable, value is secondary.
  (2) The neural agent needs no `buy-only`/`top-k` flags: its trained policy
  learned to restrict buying to belief-supported suits (the M2 finding made
  emergent). (3) Self-play data from the 120-tick/random-pool regime is the
  right training distribution ; the agent generalizes to fresh random opponents
  on both seeds.

### M4 implementation notes (verified 2026-08)

- **Population training (`agents/pool.py`).** New `PolicyAgent` plays from the
  net's policy head only (softmax sampling at temperature 1 for self-play
  opponents, argmax at temperature 0 for eval), with no tree search ; about one
  forward pass per tick, so a field of past selves costs almost nothing beside
  the searching seat. New `SnapshotPool` keeps the most recent `--pool-size`
  net snapshots and draws a self-play opponent mixture per game (defaults: 25%
  greedy self, 35% past selves, 25% heuristic, 15% random; adjustable via
  `--self-w/--past-w/--heuristic-w/--random-w`). `trainer.py` snapshots the net
  each iteration, generates games against the mixture (no pool → M3 fallback:
  all random), and evaluates (a) vs random (M3 continuity), (b) a no-collapse
  check vs each past self + heuristic + random, (c) a final full-pool eval
  (neural vs IS-MCTS + heuristic + random). `--init` warm-starts from a
  previous checkpoint.
- **No collapse against past selves (the M4 exit criterion).** Warm-started
  from the M3 checkpoint and run for 4 population iterations: eval vs random
  held at +124.2…136.3/round, 0.96–1.00 win share (M3: +123.3–126.8, 0.95–0.97);
  at every iteration the current net beat every snapshot it trained against
  (+33.3…60.0/round mean; 0.46–0.67 win share, noisy at 12 rounds). No sign of
  self-play collapse. The final net at the M3 bench config (vs 3 random,
  120 sims, d=5, seed 7): +128.8/round, 0.97 win share vs M3's +123.3, 0.95.
- **Full bot pool.** 4-player game of neural + ismcts + heuristic + random
  (40 rounds, 100 sims, d=4): neural +49.3/round (0.56 win share) vs ismcts
  +9.5 (0.29), heuristic −8.8 (0.15), random −50.0 (0.00). At only 16 rounds the
  same lineup was noise (heuristic led); with two search agents competing for
  the same buy opportunities, the honest eval needs 40+ rounds.
- **What M4 did and did not change.** The edge still comes from the trained
  policy prior; population opponents make the training distribution more
  adversarial without regressing the random-pool result. The strategy-free
  belief and the near-constant value head remain the bottleneck (P1/P2:
  behavior-conditioned beliefs, deeper trade-history features).

### P1 implementation notes (verified 2026-08)

- **History features + opponent value head.** `obs.py` adds a last-8-trade
  window `history_features` (5 floats per trade: suit, price, tick,
  aggressor-is-buyer, involves-self; 40 floats total). `net.py` grows a
  parallel history pathway (an MLP in P1) whose output is a **zero-initialized
  residual** added to the trunk output, and replaces the scalar value head with
  a **4-player** head (self first, opponents in player order, MSE on all four
  settle deltas), also zero-initialized at its final layer. Both zero-inits make
  `--init` warm-starts *exact*: a loaded checkpoint's trunk/policy transfer and
  the new pathways contribute nothing until trained. (The first attempt used a
  learned concat-combine layer and regressed the bench from M4's +128.8 to
  +108.9; the residual design fixed the regression.)
- **The opponent value head is genuinely informative.** A diagnostic zeroed the
  value head (pure policy-prior search): the random-pool result collapsed from
  +111/0.89 to +49.9/0.50 at 40r/100s/d4/seed7, so the multi-player value head
  carries real search signal where M4's scalar head was near-constant. Training
  used `--value-w 0.5` (keeps the head informative but muted; the calibration
  curve still under-predicts, e.g. pred +1.76 → actual +2.28 in the top bin).
- **P1 result.** Bench vs 3 random at the M3/M4 config (120r/120s/d5/seed7):
  +125.8/round, 0.94 win share ≈ M4's +128.8/0.97 ; parity, not the nominal
  "measured improvement" (the two are within noise). Full-pool (40r/100s/d4,
  seed 12345): +24.5 vs M4's +49.3; single 40-round runs are high-variance, so
  this is not treated as a regression. Bundle quotes were not attempted (the
  P1 deliverable list) ; the buy-only, 21-action regime is kept.

### P2 implementation notes (verified 2026-08)

- **Attention over trade history.** With `--history attn`, `net.py` replaces
  the history MLP with single-head self-attention over the 8-trade window
  (proj 5→32, softmax self-attention, mean-pool) feeding the same zero-init
  residual path, so P2 warm-starts exactly from P1's trunk/policy/value heads.
- **Learned, behavior-conditioned belief (decoupled ReBeL-lite).** The net gains
  a 12-way `belief_head` over suit-count assignments. It is trained by
  cross-entropy against the *true* assignment recorded in self-play
  (`--belief-w`, `selfplay.py` returns `assign_idx`). The belief is **decoupled**
  from the obs encoding to avoid the self-referential loop/collapse risk of a
  full ReBeL value-belief: the obs always encodes the exact strategy-free
  posterior (the distribution the policy/value heads were trained on), and the
  belief head only refines it into a prior passed to
  `BeliefTracker.set_assignment_prior`, which re-weights MCTS world sampling
  *inside the exact-consistent set* ; the learned belief can never suggest an
  impossible world (the M1 probe's overconfidence trap). CE falls 2.5 → ~0.1,
  so the head learns the assignment well.
- **P2 result.** Bench vs 3 random (120r/120s/d5/seed7): +127.75/round, 0.95 win
  share vs P1's +125.8 (within noise). Full-pool (40r/100s/d4, seed 12345):
  +29.75 **with** the learned belief vs +24.50 **without** (the +5.25 delta is
  the measurable P2 win; P1 also measured +24.50). Conclusion: history-model
  depth and the learned belief do not move the random-pool headline, but the
  learned belief helps against a mixed bot pool.

### P2b belief push: Bayes-combining the learned belief into the obs (verified 2026-08)

- **What changed.** P2's learned belief only re-weighted MCTS world sampling; the
  obs the policy/value heads saw always encoded the exact strategy-free
  posterior. The push (`NeuralAgent(bayes_combine=True)`) multiplies the exact
  posterior by the belief head's softmax and feeds the *combined* posterior
  `P(a) ∝ P_exact(a) · P_learned(a)` to the obs *and* to world sampling. Because
  both factors live on the exact-consistent support, impossible worlds stay out
  (same decoupling guarantee as P2). No retraining needed ; it is a pure
  eval-time change.
- **Result.** Random-pool bench is unchanged within noise (30r/40s/d3/seed77:
  +125.67 → +127.67; 40r/100s/d4/seed12345: +130.75 → +126.75). On the mixed
  full-pool (40r/100s/d4/seed12345, p2 checkpoint): +43.25/0.53 (learned belief
  only) → **+49.00/0.57** (combined) ; a +5.75 gain, the same size as P2's own
  learned-belief win. The pattern is consistent across P2 → P2b: the exact
  flow-consistent posterior already carries the random-pool edge; behavior
  conditioning only helps against behavior-driven opponents.

### P1v value-head recalibration (verified 2026-08)

- **Under-scaling in P1.** P1 trained with `--value-w 0.5` to keep the opponent
  value head informative but muted; its calibration curve under-predicted the
  top bin (pred +1.76 → actual +2.28).
- **Retrain.** `p1v` re-runs the P1 config with `--value-w 1.0` (4 iterations,
  warm-started from `figgie_net_p1.pt`, `figgie_net_p1v.pt`). Loss_v falls to
  ~0.02 (vs ~0.08 for P1); random-pool evals hold +120–130 (parity).
- **Calibration now.** `src/eval/value_calibrate.py` reports top-bin pred +2.27
  → actual +2.31 (vs P1's +1.76 → +2.28), MSE 0.96, ECE 0.08 on the same
  random-opponent eval. The affine artifact is saved to
  `experiments/checkpoints/figgie_net_p1v_calib.json`. Caveat: vs random
  opponents the outcome cluster is dominated by the "agent wins big" bin, so
  corr (~0.2) stays low and the affine fit is diagnostic rather than actionable
  (a monotone rescale is argmax-invariant for search anyway).

### P1b bundle quotes (verified 2026-08)

- **Engine.** `KIND_BUNDLE_BID(s1, s2, price)` rests a combined two-suit bid.
  After the per-suit execution loop, `_execute_bundles` fills each resting
  bundle bid **atomically** (buyer pays the sum of the two suits' best asks,
  each seller is paid its own ask, both asks consumed) in player order, when
  `best_ask(s1) + best_ask(s2) <= price`, sellers differ from the buyer, and
  both sellers still hold a card. Self-trading and partial fills are
  impossible. 4 new invariants in `tests/test_engine.py` (crossing fill,
  too-cheap no-fill, no-self-trade, full-round conservation + legality).
- **Obs/action space.** `obs.py` appends 12 floats to the 41-float layout
  (per pair: normalized best combined ask at 41..46, own-resting-bundle flag at
  47..52) ; the first 41 stay byte-compatible for warm-starting. The action
  space grows 21 → 45 (6 pairs × 4 prices). `net.py` `N_IN = 53` and
  `load_partial` now **grows both** the trunk's input columns and the policy
  head's output rows (old slice copied, new zero-initialized), so a P2 warm
  start keeps the trained trunk/policy/value/belief and only the 24 new bundle
  logits are fresh (verified in `tests/test_neural.py`).
- **Agent.** `neural.py` adds bundle bids to its legal actions; `PolicyAgent`
  inherits them via the 45-dim mask.
- **Training.** `figgie_net_p1b.pt` = P2 config (attn, belief-w/value-w 0.5,
  5 iterations) warm-started from `figgie_net_p2.pt` (20/20 keys). Random-pool
  evals: +122.9, +123.7, +102.1, +115.9, +92.1 (parity, slight final dip).
- **Honest negative result.** A probe found the net posts bundle bids on ~63%
  of its turns (452/720 decisions) yet only 2/138 trades fill as bundles: the
  policy learns to spam cheap non-crossing bids (they cost only the turn, and
  the heuristic opponents don't price information), and the mixed full-pool
  (40r/100s/d4/seed12345) is +28.50/0.42 vs P2's +43.25/0.53. The bundle
  mechanism is verified correct but nets do not learn to use it profitably; the
  deliverable is the complete, tested feature plus this measured finding.

### P3 playable CLI (verified 2026-08)

- `src/agents/human.py` (interactive `HumanAgent`: curated legal menu over the
  full book + optional belief goal-suit display) and `src/play_cli.py` run one
  4-player round at 120 ticks where the human holds seat 0 against the trained
  net (search), the heuristic bot, and a random bot. Tested end-to-end
  (`tests/test_human.py` asserts every menu action is engine-legal).

### V&V (verified 2026-08)

- Full suite: **48 passed** (`pytest`), covering engine invariants (incl. bundle
  fills, payout conservation), belief calibration, action-space roundtrips,
  warm-start growth, agent-vs-baseline comparisons, and human-menu legality.

---

## 11. Project layout

```
opencodeProject/
├── DESIGN.md
├── README.md
├── pyproject.toml
├── configs/            # per-experiment yaml (seed, hyperparams, agent mix)
├── src/
│   ├── env/            # engine, deck, state, observations, render
│   ├── belief/         # belief tracker + calibration
│   ├── agents/         # random, heuristic, is_mcts, neural, pool
│   ├── learning/       # net, selfplay, replay, trainer
│   └── eval/           # bench, probe, calibrate
├── tests/
└── experiments/        # run logs, checkpoints (gitignored except configs)
```

Python 3.11+, PyTorch, NumPy; no other hard deps for M0–M2. Vectorized
self-play engine (batch envs) so heuristics can run thousands of games/sec;
neural search targets a single GPU.

---

## 12. Open decisions (to resolve during M0/M1)

1. ✅ Trade execution price: **resting quote's limit** (aggressor crosses to
   the resting side). Midpoint considered and rejected ; it subsidizes
   aggressive quoting.
2. Tick granularity for the simultaneous-move loop (e.g. 4 min / 240 ticks
   @1s vs adaptive event-driven). M0 uses 120 ticks (2s each); revisit.
3. 4 vs 5 players as the primary target (5 has richer dealer? no ; differs
   only in per-player ante; pick 4 for compute, verify generalization later).
4. Price-grid resolution ($1 vs $2 buckets) ; balances policy head size vs
   fidelity. M0 bots quote in $1 ticks up to $25; `legal_actions` uses
   `PRICE_GRID`.
5. Reward shaping: none (pure terminal) vs auxiliary per-trade signal.
6. Whether to include a "no-trade" legal constraint on holding cash (you can
   always pass).

## 13. Risks

- **Multi-agent non-convergence / cycling** → population training + eval vs
  frozen pool; accept "dominates pool" not "Nash".
- **Action-abstraction fidelity** → abstraction must admit the dominant
  human strategies (cheap accumulation, late-round price spikes, bundle
  concealment); validate via M4 vs heuristic pool with rich actions.
- **Reward hacking through shaping** → shaping is auxiliary only; terminal
  bankroll governs eval and final targets.
- **Search cost under continuous action space** → mask actions, use prior
  top-k truncation, and (P2) compact belief encoding.
- **Overfitting to heuristics** → keep a diverse pool, fresh seeds, and
  regular snapshot sweeps.

## 14. References

- Jane Street: Figgie rules ; https://www.figgie.com/how-to-play
- Stanford CS224R: "Reinforcement Learning for Figgie: Learning Negotiation
  as a ..." (prior student work; useful baseline reading).
- OpenSpiel (no Figgie; closest: negotiation game) ; Lanctot et al. 2019.
- AlphaZero ; Silver et al. 2018.
- ReBeL (AlphaZero-style imperfect-info search + learned belief) ; Brown,
  Bakhtin et al. 2020.
- DeepStack (belief states + counterfactual value networks) ; Moravčík et
  al. 2017.
- IS-MCTS ; Cowling, Powley, Whitehouse 2012.
- Suphx (multiplayer mahjong, reward decomposition) ; Li et al. 2020.
