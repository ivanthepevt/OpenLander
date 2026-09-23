# OPENLANDER REAL-SEED EVALUATION

> This report describes system behavior, inference performance, decision stability, retarget behavior, runtime, and perception outputs. The seed set has no human-validated landing ground truth, so it does not measure landing accuracy.

- Total scenarios: 11
- Successful scenarios: 11
- Failed scenarios: 0
- Total processed frames: 1650
- Compute device: Apple MPS
- Total batch processing time: 33.08 min

| Scenario | Environment | Frames | Analysis Time | ms/frame | Retargets | Final Score | Target Lock | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 11_fields_straight_500m | Fields | 150 | 162.14s | 1080.4 | 0 | 59.4% | 0.100s | FINAL_APPROACH |
| 12_mountain_side_500m | Mountain | 150 | 161.59s | 1076.7 | 1 | 62.6% | 0.100s | FINAL_APPROACH |
| 12_mountain_straight_500m | Mountain | 150 | 165.86s | 1105.2 | 3 | 67.1% | 0.100s | FINAL_APPROACH |
| 13_moon_straight_500m | Moon | 150 | 165.28s | 1101.3 | 0 | 37.8% | 0.100s | FINAL_APPROACH |
| 14_mars_straight_800m | Mars | 150 | 156.12s | 1040.3 | 1 | 51.9% | 0.100s | FINAL_APPROACH |
| 19_night_side_600m | Night | 150 | 155.62s | 1036.9 | 16 | 66.6% | 0.100s | FINAL_APPROACH |
| 1_parking_side_300m | Parking | 150 | 160.2s | 1067.5 | 3 | 70.2% | 0.100s | FINAL_APPROACH |
| 1_parking_straight_400m | Parking | 150 | 182.33s | 1214.8 | 0 | 79.8% | 0.100s | FINAL_APPROACH |
| 2_warehouse_straight_350m | Warehouse | 150 | 166.18s | 1107.2 | 0 | 70.3% | 0.100s | FINAL_APPROACH |
| 4_suburban_straight_400m | Suburban | 150 | 175.46s | 1169.2 | 7 | 63.9% | 0.100s | FINAL_APPROACH |
| 8_dockyard_side_800m | Dockyard | 150 | 166.94s | 1112.4 | 3 | 63.7% | 0.100s | FINAL_APPROACH |

## 11_fields_straight_500m

Source: `11_fields_straight_500m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-01  
Final score: 59.4%  
Retargets: 0  
Analysis time: 162.14 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/11_fields_straight_500m_2026-09-23_120246_480/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/11_fields_straight_500m_2026-09-23_120246_480/dashboard_replay.mp4`  
Error: None

## Visual spot check

Final Decision and Debug Dashboard frames were inspected for `1_parking_side_300m`, `12_mountain_side_500m`, `13_moon_straight_500m`, and `19_night_side_600m`. No blank or corrupted frames were observed; overlays remained registered to the imagery, candidate markers were visible, and the cached layer layouts rendered correctly. See `outputs/evaluation/visual_spot_check.jpg`.

This inspection checks rendering and artifact integrity only. It does not validate landing correctness or accuracy against human ground truth.

## 12_mountain_side_500m

Source: `12_mountain_side_500m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 62.6%  
Retargets: 1  
Analysis time: 161.59 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/12_mountain_side_500m_2026-09-23_120542_802/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/12_mountain_side_500m_2026-09-23_120542_802/dashboard_replay.mp4`  
Error: None

## 12_mountain_straight_500m

Source: `12_mountain_straight_500m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 67.1%  
Retargets: 3  
Analysis time: 165.86 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/12_mountain_straight_500m_2026-09-23_120839_688/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/12_mountain_straight_500m_2026-09-23_120839_688/dashboard_replay.mp4`  
Error: None

## 13_moon_straight_500m

Source: `13_moon_straight_500m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-01  
Final score: 37.8%  
Retargets: 0  
Analysis time: 165.28 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/13_moon_straight_500m_2026-09-23_121139_895/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/13_moon_straight_500m_2026-09-23_121139_895/dashboard_replay.mp4`  
Error: None

## 14_mars_straight_800m

Source: `14_mars_straight_800m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 51.9%  
Retargets: 1  
Analysis time: 156.12 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/14_mars_straight_800m_2026-09-23_121436_181/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/14_mars_straight_800m_2026-09-23_121436_181/dashboard_replay.mp4`  
Error: None

## 19_night_side_600m

Source: `19_night_side_600m.png.png`  
Mission mode: SAFETY  
Result: FINAL_APPROACH  
Final selected zone: LZ-01  
Final score: 66.6%  
Retargets: 16  
Analysis time: 155.62 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/19_night_side_600m_2026-09-23_121725_760/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/19_night_side_600m_2026-09-23_121725_760/dashboard_replay.mp4`  
Error: None

## 1_parking_side_300m

Source: `1_parking_side_300m.png.png`  
Mission mode: RECOVERY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 70.2%  
Retargets: 3  
Analysis time: 160.2 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/1_parking_side_300m_2026-09-23_122014_698/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/1_parking_side_300m_2026-09-23_122014_698/dashboard_replay.mp4`  
Error: None

## 1_parking_straight_400m

Source: `1_parking_straight_400m.png.png`  
Mission mode: RECOVERY  
Result: FINAL_APPROACH  
Final selected zone: LZ-01  
Final score: 79.8%  
Retargets: 0  
Analysis time: 182.33 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/1_parking_straight_400m_2026-09-23_122310_460/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/1_parking_straight_400m_2026-09-23_122310_460/dashboard_replay.mp4`  
Error: None

## 2_warehouse_straight_350m

Source: `2_warehouse_straight_350m.png.png`  
Mission mode: RECOVERY  
Result: FINAL_APPROACH  
Final selected zone: LZ-01  
Final score: 70.3%  
Retargets: 0  
Analysis time: 166.18 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/2_warehouse_straight_350m_2026-09-23_122629_938/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/2_warehouse_straight_350m_2026-09-23_122629_938/dashboard_replay.mp4`  
Error: None

## 4_suburban_straight_400m

Source: `4_suburban_straight_400m.png.png`  
Mission mode: RECOVERY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 63.9%  
Retargets: 7  
Analysis time: 175.46 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/4_suburban_straight_400m_2026-09-23_122931_543/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/4_suburban_straight_400m_2026-09-23_122931_543/dashboard_replay.mp4`  
Error: None

## 8_dockyard_side_800m

Source: `8_dockyard_side_800m.png.png`  
Mission mode: RECOVERY  
Result: FINAL_APPROACH  
Final selected zone: LZ-02  
Final score: 63.7%  
Retargets: 3  
Analysis time: 166.94 s  
Replay: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/8_dockyard_side_800m_2026-09-23_123241_110/landing_replay.mp4`  
Dashboard: `/Users/ivan/Desktop/Design Festival/OpenLander/runs/8_dockyard_side_800m_2026-09-23_123241_110/dashboard_replay.mp4`  
Error: None
