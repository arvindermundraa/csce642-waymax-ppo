# Learning Safe Closed-Loop Driving Policies with PPO in Waymax

CSCE 642 Deep Reinforcement Learning, course project
Debmalya Chatterjee and Arvinder Singh Mundra

We train a PPO policy that drives one car (the ego) in closed loop through real scenes
from the Waymo Open Motion Dataset (WOMD) with the Waymax simulator, and compare it with
a hand-designed controller (pure pursuit with IDM speed control), Waymax's IDM policy
driving the ego, and the logged human trajectory, on the same held-out scenes.

- Code: https://github.com/arvindermundraa/csce642-waymax-ppo
- Video (5 min): VIDEO-URL-TO-BE-ADDED
- Report: [`report/report.pdf`](report/report.pdf)

Everything is in one notebook, **`csce642_waymax_ppo.ipynb`**, saved with the outputs of
the full run: data analysis, environment, reward, PPO, the reference controllers, all
four experiments from the proposal, statistics, failure analysis, stretch goals, a
follow-up that fixes the main failure, and the written findings (section 15).

## Notebook contents

1. Setup (Waymax install and the two JAX compatibility patches)
2. Data: loading, scene selection, routes, traffic lights, dataset summary
3. Environment: observation, action, reward, termination, reactive IDM traffic,
   sanity checks with the logged human
4. PPO (JAX/Flax, CleanRL-style)
5. First result: one scene end to end (proposal §3.4)
6. Training: 12 runs (3 reward settings × 3 seeds, plus 3 seeds trained with IDM traffic)
7. Evaluation protocol, metrics, failure classifier, bootstrap confidence intervals
8. Experiment 1 (H1): PPO vs the hand-designed controller, paired tests
9. Experiment 2 (H2): replayed vs reactive traffic, and a check of our IDM agents
   against Waymax's `IDMRoutePolicy`
10. Experiment 3 (H3): reward study and stop-and-wait
11. Experiment 4: failure analysis with rendered episodes, and a check of the failure
    labels by eye
12. Smoothness
13. Stretch goals: behaviour cloning, BC → PPO, richer observation, a policy trained only
    on intersection scenes (3 seeds each)
14. Improving the policy: lead-vehicle features, time-to-collision and red-light terms
15. Findings

## Running it

**Colab (recommended).** Put this folder in `My Drive/csce642/csce642-waymax-ppo`, put
the two shards in its `data/womd/train` and `data/womd/val` folders, open the notebook
with a GPU runtime and run all cells. The first cell mounts Drive and installs Waymax.

**Locally.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax"
jupyter lab csce642_waymax_ppo.ipynb
```

The cells that render episodes write mp4 files through `mediapy`, which needs `ffmpeg`
on the `PATH`.

Every expensive step writes its output to disk and is skipped when it exists, so the
notebook can be rerun after an interruption. `SMOKE=1 jupyter lab ...` runs a tiny
version of every cell in about three minutes and writes its outputs under `smoke/`, so
it does not touch the real results. The full run (27 PPO runs of 2M environment steps)
took about 8 hours on a 2-core CPU; a GPU is much faster. With the checkpoints in place,
rerunning the notebook only repeats the evaluation and rendering.

## Data

Register at https://waymo.com/open, then download one training and one validation shard
of the v1.1.0 `tf_example` split:

```
data/womd/train/training_tfexample.tfrecord-00000-of-01000      (607 MB, 478 scenes)
data/womd/val/validation_tfexample.tfrecord-00000-of-00150      (368 MB, 288 scenes)
```

All 766 scenes are used: 478 for training and 288 held out (168 *driving* scenes where
the human covers at least 20 m, 120 *waiting* scenes). Each scene keeps the 64 objects
closest to the ego's path (Waymax's loader otherwise keeps the first N in file order and
drops nearby cars), the 3,000 map points closest to the route, and the traffic lights
that control the ego's route.

## Main results (288 held-out scenes, replayed traffic)

PPO: mean ± sd over 3 seeds; brackets are 95 % bootstrap CIs.

| Method | Collision % | Off-road % | Ran red % | Success % |
|---|---|---|---|---|
| Logged human | 0.3 | 1.7 | 1.4 | 97.9 |
| IDM (Waymax) ego | 13.2 | 8.3 | 5.2 | 77.1 |
| Hand-designed | 7.3 [4.5, 10.4] | 3.8 | 2.8 | 85.8 [81.6, 89.6] |
| PPO balanced | 28.6 ± 1.1 [24.2, 33.1] | 4.2 ± 0.3 | 10.5 ± 1.7 | 66.6 ± 0.9 [61.7, 71.3] |
| PPO progress-focused | 54.2 ± 2.8 | 11.7 ± 0.7 | 13.3 ± 0.3 | 32.1 ± 4.6 |
| PPO safety-focused | 21.4 ± 1.3 | 4.6 ± 1.7 | 5.2 ± 0.3 | 53.5 ± 3.7 |
| PPO balanced, trained with IDM | 33.0 ± 2.3 | 6.0 ± 0.3 | 6.7 ± 0.6 | 54.6 ± 3.4 |
| BC → PPO | 28.4 ± 2.6 | 5.1 ± 0.4 | 8.4 ± 0.6 | 64.8 ± 3.5 |
| PPO, K=10 + road edges | 34.5 ± 2.6 | 5.9 ± 0.9 | 9.3 ± 0.3 | 56.2 ± 2.8 |
| PPO trained on intersections | 33.4 ± 1.8 | 10.0 ± 1.1 | 13.2 ± 0.3 | 55.6 ± 2.3 |
| PPO + lead/TTC/red terms (section 14) | 9.5 ± 2.7 [6.8, 12.4] | 7.8 ± 1.2 | 2.8 ± 0.5 | 81.8 ± 2.8 [78.1, 85.4] |
| same, lead measured along the route | 11.1 ± 1.0 [8.2, 14.1] | 6.5 ± 0.4 | 3.0 ± 0.3 | 80.7 ± 1.8 [76.9, 84.4] |

- **H1 not supported.** PPO collides more than the hand-designed controller in every
  setting (balanced: 68 scenes where only PPO collides vs 9, McNemar p < 0.001). The gap
  is mostly in waiting scenes (46.7 % vs 6.7 % collisions): PPO does not learn to wait.
  In driving scenes it is closer (15.7 % vs 7.7 %, progress 98 % vs 96 %).
- **H2 reversed.** Replay-trained policies collide *less* with reactive IDM traffic
  (balanced 28.6 → 26.3 %, safety 21.4 → 14.0 %), as in the Waymax paper's DQN ablation;
  IDM-trained policies do worse on replayed traffic (33.0 %).
- **H3 supported.** Collisions fall 54.2 → 28.6 → 21.4 % as the penalty weight goes
  10 → 50 → 200, progress in driving scenes falls 105 → 98 → 73 %, and the safety-focused
  policy stalls in 14.5 % of driving scenes (all three seeds).
- **Stretch goals.** BC → PPO matches PPO from scratch; the richer observation is worse.
  Training only on intersection scenes lowers collisions on the 45 held-out intersection
  scenes (13.3 % vs 18.5 %) but raises off-road exits (11.9 % vs 3.7 %).
- **Follow-up (section 14).** Adding the gap, closing speed and time to collision to the
  vehicle or red stop line ahead, plus a time-to-collision and a red-light penalty, cuts
  collisions from 28.6 % to 9.5 % and raises success from 66.6 % to 81.8 %. On replayed
  traffic it is no longer distinguishable from the hand-designed controller on
  collisions (13 vs 13 scenes, p = 1.0), but it still leaves the road more, and under
  reactive IDM traffic it collides in 21.8 % of scenes against 7.3 %.
- **Checks on the analysis.** We compared the failure labels with what happened in 70
  rendered episodes and agreed in 44: *struck from behind*, *rear-end* and *off-road on
  a straight* are reliable, *collision while turning* and *off-road on a curve* are not
  (`results/classifier_check.csv`). Most off-road events of the faster policies happen
  after the ego has passed the end of the logged path; within the logged path all
  methods are between 0.3 % and 3.2 % (`results/offroad_context.csv`).

Full tables (driving, waiting, turning and intersection slices, IDM traffic, stretch
goals, the section 14 policies) are in `results/tables.md`; figures are in `figures/`.
The numbers were regenerated from the checkpoints on a second machine; 11 of 15,840
per-episode outcomes changed through floating-point differences.

## Layout

```
csce642_waymax_ppo.ipynb   the project
results/                   per-scene CSVs for every method and traffic model, summary tables
figures/                   all plots from the notebook
runs/<name>/progress.csv   training logs (checkpoints are kept locally, not in git)
videos/                    rendered episodes (local only)
report/                    final report (LaTeX sources and PDF)
```

## References

- Gulino et al. (2023). Waymax: An accelerated, data-driven simulator for large-scale
  autonomous driving research. NeurIPS Datasets and Benchmarks.
- Ettinger et al. (2021). Large scale interactive motion forecasting for autonomous
  driving: The Waymo Open Motion Dataset. ICCV.
- Schulman et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Huang et al. (2022). CleanRL: High-quality single-file implementations of deep
  reinforcement learning algorithms. JMLR 23(274).
- Treiber, Hennecke and Helbing (2000). Congested traffic states in empirical
  observations and microscopic simulations. Physical Review E 62(2).
- Peng et al. (2024). Improving agent behaviors with RL fine-tuning for autonomous
  driving. ECCV.

```bibtex
@inproceedings{waymax,
  title     = {Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research},
  author    = {Cole Gulino and Justin Fu and Wenjie Luo and George Tucker and Eli Bronstein and Yiren Lu and Jean Harb and Xinlei Pan and Yan Wang and Xiangyu Chen and John D. Co-Reyes and Rishabh Agarwal and Rebecca Roelofs and Yao Lu and Nico Montali and Paul Mougin and Zoey Yang and Brandyn White and Aleksandra Faust and Rowan McAllister and Dragomir Anguelov and Benjamin Sapp},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year      = {2023}
}
```

This software was made using the Waymo Open Dataset, provided by Waymo LLC under the
Waymo Dataset License Agreement for Non-Commercial Use, available at
waymo.com/open/terms, and your access and use of such work are governed by the terms
and conditions therein. The dataset and the cached scenes are not included here.
