# v5_final — validation gate

## Gate A — PC7 controls (well-studied; should NOT top the list)

| drug | rank_final | net_direction | real_angio | novelty_real | score_v5_final | novelty_status |
|---|---|---|---|---|---|---|
| Suramin sodium salt | 62 | -4.529 | 408 | 0.129 | 0.4665 | well-studied |
| DMH4 | 98 | -0.093 | 5 | 0.741 | 0.4348 | PARTIAL |
| Sunitinib malate | 126 | -2.179 | 2860 | 0.1 | 0.4204 | well-studied |
| SU 5416 | 1058 | -1.133 | 39 | 0.466 | 0.234 | PARTIAL |

**all_anti**: True  **median rank_final**: 112  **any in top-5 anti**: False

## Gate B — top-5 composition by direction

```
{
  "anti_top5_status_breakdown": {
    "PARTIAL": 5
  },
  "pro_top5_status_breakdown": {
    "PARTIAL": 5
  },
  "anti_top5_median_real_angio": 0.0,
  "pro_top5_median_real_angio": 6.0
}
```

## Gate C — discovery set composition

- n_discovery (real_angio ≤ 2): **530**
- n_TRUE-DARK in discovery: **13**
- n_active_dark_real (real_angio ≤ 2 ∧ activity ≥ p75): **128**

### Top-10 of discovery set

| drug | rank_final | novelty_status | net_direction | activity | score_v5_final |
|---|---|---|---|---|---|
| 7-Chloro-4-hydroxy-2-phenyl-1,8-naphthyridine | 2 | PARTIAL | -3.631 | 0.8235 | 0.6443 |
| cis-4-Aminocrotonic acid | 3 | PARTIAL | -3.216 | 0.7169 | 0.6368 |
| NSC 95397 | 4 | PARTIAL | -3.586 | 0.7034 | 0.6314 |
| PD-180970 | 5 | PARTIAL | -2.896 | 0.7024 | 0.631 |
| Tetraethylthiuram disulfide | 6 | PARTIAL | -3.24 | 0.4879 | 0.6246 |
| IC 261 | 7 | PARTIAL | -4.705 | 0.7314 | 0.6074 |
| IMS2186 | 8 | PARTIAL | -2.685 | 0.4344 | 0.6032 |
| Z-L-Phe chloromethyl ketone | 9 | PARTIAL | -2.115 | 0.5963 | 0.5885 |
| IPA-3 | 11 | PARTIAL | -3.418 | 0.5127 | 0.5717 |
| Bromoacetyl alprenolol menthane | 13 | PARTIAL | -1.896 | 0.533 | 0.5632 |

## Gate D — Suramin / Y-27632 demotion (the headline assertion)

| drug | rank (v5) | rank_final | Δ rank | real_angio | novelty_real | corpus_novelty (v5) |
|---|---|---|---|---|---|---|
| Suramin sodium salt | 1 | 62 | +61 | 408 | 0.1294 | 0.8997 |
| Y-27632 dihydrochloride | 11 | 539 | +528 | 559 | 0.1 | 0.767 |
| Atorvastatin calcium salt trihydrate | 22 | 668 | +646 | 1299 | 0.1 | 0.7406 |
| Sildenafil citrate salt | 26 | 739 | +713 | 835 | 0.1 | 0.7406 |
| Serotonin hydrochloride | 19 | 882 | +863 | 3412 | 0.1 | 0.8997 |
