**Replayed traffic**

| Method | Seeds | Collision % | Off-road % | Route progress % | Success % | Return | Jerk (m/s³) | Δκ (1/m) |
|---|---|---|---|---|---|---|---|---|
| Logged human | 1 | 0.0 | 1.2 | 99.4 | 98.8 | 97.9 | 1.89 | 0.0007 |
| IDM (Waymax) | 1 | 11.3 | 8.3 | 120.4 | 77.4 | 88.4 | 2.99 | 0.0035 |
| Hand-designed | 1 | 6.5 | 3.0 | 96.9 | 85.1 | 89.8 | 0.76 | 0.0004 |
| PPO (balanced) | 3 | 19.2 ± 0.7 | 3.8 ± 1.0 | 97.4 ± 1.4 | 76.8 ± 1.3 | 79.4 ± 1.1 | 2.63 ± 0.35 | 0.0047 ± 0.0003 |
| PPO (progress) | 1 | 28.6 | 6.5 | 107.0 | 64.3 | 71.9 | 2.95 | 0.0069 |
| PPO (safety) | 1 | 8.3 | 2.4 | 99.2 | 88.1 | 89.6 | 2.02 | 0.0055 |

**Failure types (% of episodes, replayed traffic)**

|                    |   PPO (balanced) |   Logged human |   IDM (Waymax) |   PPO (progress) |   Hand-designed |   PPO (safety) |
|:-------------------|-----------------:|---------------:|---------------:|-----------------:|----------------:|---------------:|
| struck_from_behind |              0.8 |            0   |            2.4 |              0.6 |             4.8 |            0.6 |
| rear_end           |             10.5 |            0   |            3   |             14.3 |             0   |            3.6 |
| collision_turning  |              1   |            0   |            1.2 |              1.8 |             0   |            0.6 |
| collision_other    |              6.9 |            0   |            4.8 |             11.9 |             1.8 |            3.6 |
| offroad_curve      |              0.4 |            0   |            1.8 |              0   |             0.6 |            0   |
| offroad_straight   |              3.4 |            1.2 |            6.5 |              6.5 |             2.4 |            2.4 |
| stalled            |              0   |            0   |            0   |              0   |             0   |            0   |

**Reactive IDM traffic (same policies, no retraining)**

| Method | Seeds | Collision % | Off-road % | Route progress % | Success % | Return | Jerk (m/s³) | Δκ (1/m) |
|---|---|---|---|---|---|---|---|---|
| Logged human | 1 | 13.1 | 1.2 | 94.5 | 85.7 | 86.5 | 1.84 | 0.0007 |
| Hand-designed | 1 | 8.9 | 3.0 | 95.6 | 76.2 | 86.9 | 0.81 | 0.0004 |
| PPO (balanced) | 3 | 20.6 ± 2.7 | 4.8 ± 0.8 | 97.0 ± 0.6 | 74.2 ± 3.0 | 76.6 ± 2.0 | 2.76 ± 0.34 | 0.0048 ± 0.0002 |
| PPO (safety) | 1 | 19.0 | 2.4 | 94.7 | 78.0 | 79.0 | 2.01 | 0.0056 |
