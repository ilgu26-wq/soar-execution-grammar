#!/usr/bin/env python3
"""
EXP-36: Shadow Geometry Test (shadow geometry)
===================================================
"alpha light not... but shadowto/as Measurementbecomes"
When the sun shines light, we do not look at the sun but shadow computationdoes.

MOTIVATION:
 EXP-35until energy, frame, asymmetry directly Measurementdid.
 One step further: measuring 'structure revealed only in regions where energy has disappeared.'
 Structure is invisible where light directly reaches.
 structure always blockingbecome boundaryfrom appears.

DESIGN:
 Shadow Region: energy from the point where it first drops below 0 or ATP activates
 Light Region: Shadow before interval (energy positive, alpha survival among)

 1. Shadow Metrics (per trade):
 - Shadow Duration: shadow interval bar number/can
 - Shadow Fraction: before bar among shadow ratio
 - S_E (Shadow Energy Integral): Σ E_total(k) for k ∈ shadow
 - Shadow Axis Drift: shadow intervalfrom's/of total axis Movement
 - Zero Crossings: shadow notfrom E_total 0 exceed count
 - Shadow Recovery: shadow entry after E_total > 0 recovery whether

 2. Shadow Classification:
 - NO_SHADOW: shadow_fraction < 5% (almost all light inside)
 - CLEAN_SHADOW: S_E < 0, axis Movement small (quiet death)
 - FRACTURED_SHADOW: S_E < 0, axis Movement large (axis shattered)
 - PENUMBRA (half/anti-shadow): E_total 0 multiple times cross (boundaryfrom oscillation)

 3. Shadow-weighted Axis vs Light-weighted Axis:
 each intervalfrom |E_total|to/as among average axis eachalso

RESULTS:
 Money preserved:
 Net PnL: $1,200.00 [IDENTICAL]
 WR: 39.2% [IDENTICAL]
 Max DD: 0.42% [IDENTICAL]
 Gate: UNTOUCHED

 === Shadow Class Distribution ===

 NO_SHADOW 22.5% WR=98.5% ← shadow none = almost before wins
 CLEAN_SHADOW 43.3% WR= 7.1% ← quiet death
 FRACTURED_SHADOW 6.8% WR=25.0% ← axis destruction
 PENUMBRA 27.3% WR=45.0% ← boundary oscillation

 === Shadow Geometry by Fate (key/core Discovery) ===

 NO CLEAN FRACT PENUM Shad% S_E Recov% WR
 IMMORTAL 40 0 0 7 14.9% +21.5 14.9% 95.7%
 SURVIVED 20 0 0 1 3.8% +310.1 4.8% 95.2%
 ZOMBIE 5 4 6 36 70.0% +40.1 72.5% 62.7%
 TERMINATED 1 58 13 33 86.6% -86.7 41.9% 17.1%
 STILLBORN 0 65 1 3 100.0% -300.1 4.3% 0.0%

 === Hypothesis Tests ===

 H-36a SUPPORTED: CLOSED_ALPHA shadow=15.5% vs FAILED_OPEN=93.3% (Δ=-77.8%)
 H-36b NOT SUPPORTED: ZOMBIE CLEAN_SHADOW=7.8% vs Others=50.8%
 → ZOMBIE CLEAN not... but PENUMBRA! (70.6%)
 H-36c SUPPORTED: TERMINATED FRACTURED=12.4% vs Others=3.7%
 H-36d SUPPORTED: CONTESTED→PENUMBRA=42.6% vs Non-CONTESTED=23.8% (1.79x)

 === Shadow-First Axis Drift ===

 Light region: 4.75°/bar
 Shadow region: 12.21°/bar
 Shadow-first drift: 53.6%
 → axis Movement shadowfrom accelerationbecomes

PHYSICAL INTERPRETATION:

 ═══ 5know key/core Discovery ═══

 1. "shadow does not exist = profit exists"
 NO_SHADOW's/of WR = 98.5% (66 among 65 victory)
 IMMORTAL's/of 85% (40/47), SURVIVED's/of 95% (20/21) NO_SHADOW
 → light insidefrom endI alphaonly survive.

 2. "STILLBORN 100% shadow"
 69 all shadow. 65 CLEAN_SHADOW.
 S_E = -300.1 (most deep darkness)
 lightat one/a timealso does not reach did not — bornimmediately after darkness.

 3. "ZOMBIE half/anti-shadow(PENUMBRA)at lives"
 H-36b NOT SUPPORTEDperson/of keep:
 ZOMBIEis PENUMBRA 70.6% (36/51), not CLEAN_SHADOW
 Recovery rate = 72.5% (most high)
 lightand darkness's/of boundaryfrom continues oscillationdoes.
 Therefore ZOMBIE WR = 62.7% — come back to lifealso, dyingalso does.

 4. "TERMINATED CLEAN_SHADOW + FRACTURED_SHADOW"
 58 CLEAN (quiet death) + 13 FRACTURED (axis shattered)
 Recovery rate = 41.9% (among between)
 energy even if it returns, axis already shattered so meaningless.

 5. "CONTESTED PENUMBRAfrom born"
 H-36d SUPPORTED: 1.79x correlation
 CONTESTED's/of 42.6% PENUMBRA
 → half/anti-shadow = alpha survival does not die determination boundary interval

 ═══ sun-shadow analogy theorem/cleanup ═══

 sun(energy flow) object(alpha) illuminates
 Where light reaches = Light Region = positive energy = living alpha
 shadow = Shadow Region = energy negative = dead alpha
 Penumbra = oscillation near 0 = boundary (ZOMBIE/CONTESTED)

 Key: An object's shape is not from light but shadow's/of outlinefrom is revealed
 axis Movement shadowfrom 2.6times/multiple more fast (12.21° vs 4.75°/bar)
 → "structure lightfrom grow, shape shadowfrom is revealed"

 ═══ irreversiblenature/property structure's/of completion ═══

 EXP-23: energy trajectory (sun's/of brightness)
 EXP-24: among axis Movement (sun's/of position)
 EXP-31: observation frame (where )
 EXP-32: orbit closure (alpha return)
 EXP-34: frame transition (observation if moves)
 EXP-35: coordinates cost (Stable information )
 EXP-36: shadow geometry (structure darknessfrom revealed)

 irreversiblenature/property's/of hierarchy Structure:
 [observation frame selection] → [information asymmetry occurrence] → [shadow structure determination] → [behavior irreversiblenature/property]
 all thing belowfrom aboveto/as life long.
"""

if __name__ == '__main__':
 print("EXP-36: Shadow Geometry Test — see live_report.py for full results")
 print("Run: python experiments/live_report.py")
 print("Dataset: data/exp_evidence/exp36_shadow_geometry/")
