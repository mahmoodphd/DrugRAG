# v5_final — Validation: recovered known anti-angiogenics

Drugs whose name matches a known anti-angiogenic (Sunitinib, Colchicine, Vinblastine, 2-Methoxyestradiol, Suramin, Resveratrol, Tamoxifen, Topotecan, ...). These should appear with `net_direction < 0` and high `kg_plausibility`/`activity_magnitude`, demonstrating that the method recovers real biology even after the novelty term is moved to real PubMed (which floors their `novelty_real` to ≈0.1).

| rank_final | drug_name | cluster | net_direction | activity_magnitude | real_angio_pubmed_hits | novelty_real | novelty_status | kg_plausibility | score_v5_final | corpus_n_papers |
|---|---|---|---|---|---|---|---|---|---|---|
| 21 | Idarubicin | 7 | -2.4212 | 0.4103 | 24 | 0.5340 | PARTIAL | 0.7472 | 0.5378 | 12 |
| 62 | Suramin sodium salt | 7 | -4.5294 | 0.8722 | 408 | 0.1294 | well-studied | 0.2891 | 0.4665 | 1 |
| 74 | Colchicine | 7 | -2.5511 | 0.6158 | 557 | 0.1000 | well-studied | 0.6712 | 0.4491 | 14 |
| 126 | Sunitinib malate | 7 | -2.1792 | 0.3385 | 2860 | 0.1000 | well-studied | 1.0000 | 0.4204 | 8 |
| 137 | 2-methoxyestradiol | 7 | -3.0698 | 0.4729 | 301 | 0.1733 | well-studied | 0.6712 | 0.4176 | 10 |
| 291 | Topotecan hydrochloride hydrate | 7 | -1.9407 | 0.3055 | 178 | 0.2490 | well-studied | 0.7472 | 0.3962 | 1 |
| 499 | Vinblastine sulfate salt | 7 | -3.9829 | 0.6224 | 335 | 0.1579 | well-studied | 0.2891 | 0.3765 | 7 |
| 514 | Vincristine sulfate | 7 | -3.1489 | 0.5366 | 434 | 0.1205 | well-studied | 0.4582 | 0.3714 | 12 |
