# Real-PubMed corpus-novelty verification (v5 sidecar)

Our v5 `novelty` is **corpus-relative** — it counts hits in our 8 514-abstract angiogenesis-anchored corpus. This sidecar checks every dark / top-list candidate against real, uncapped PubMed (Entrez `esearch`, count-only) to separate genuinely understudied drugs from drugs our corpus simply missed.

**Queries (per drug):**
- `angio  = "{drug}" AND (angiogenesis OR endothelial OR VEGF OR "blood vessel" OR neovascularization OR "tube formation")`
- `total  = "{drug}"`

**Classification:**
- `TRUE-DARK`     = `n_papers == 0` AND `real_angio_hits <= 2`
- `CORPUS-MISSED` = `n_papers <= 5` AND `real_angio_hits >= 10`
- `PARTIAL`       = anything else

**Counts:** total queried = **65**, TRUE-DARK = **13**, CORPUS-MISSED = **14**, PARTIAL = **38**, QUERY-ERROR = **0**.

## TRUE-DARK (confirmed real discoveries)

| drug_name | cluster | set_dark | set_top_anti | set_top_pro | n_papers | real_angio_pubmed_hits | real_total_pubmed_hits | novelty | activity_magnitude | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| PHA 767491 hydrochloride | 7 | True | False | False | 0 | 0 | 26 | 1.0000 | 0.1462 | 225 |
| N2-Ethyl-2'-deoxyguanosine | 4 | True | False | False | 0 | 0 | 21 | 1.0000 | 0.1267 | 237 |
| Propantheline bromide | 7 | True | False | False | 0 | 0 | 701 | 1.0000 | 0.1151 | 254 |
| Tomoxetine | 3 | True | False | False | 0 | 0 | 68 | 1.0000 | 0.1121 | 259 |
| CP-471474 | 4 | True | False | False | 0 | 0 | 1 | 1.0000 | 0.1032 | 271 |
| CP-101537 | 1 | True | False | False | 0 | 0 | 0 | 1.0000 | 0.1014 | 274 |
| Sematilide monohydrochloride monohydrate | 6 | True | False | False | 0 | 0 | 0 | 1.0000 | 0.0868 | 300 |
| BNTX maleate salt hydrate | 7 | True | True | False | 0 | 1 | 125 | 1.0000 | 0.4642 | 32 |
| Ziprasidone hydrochloride monohydrate | 7 | True | False | False | 0 | 1 | 2291 | 1.0000 | 0.1628 | 202 |
| Dolasetron mesylate hydrate | 3 | True | False | False | 0 | 1 | 313 | 1.0000 | 0.1034 | 270 |
| Doxepin hydrochloride | 2 | True | False | False | 0 | 1 | 1620 | 1.0000 | 0.0940 | 286 |
| Caroverine hydrochloride | 2 | True | False | False | 0 | 2 | 53 | 1.0000 | 0.0954 | 282 |
| CP-93129 dihydrochloride hydrate | 1 | True | False | False | 0 | 2 | 61 | 1.0000 | 0.0925 | 289 |

## CORPUS-MISSED (corpus-novel but real-PubMed evidence exists)

| drug_name | cluster | set_dark | set_top_anti | set_top_pro | n_papers | real_angio_pubmed_hits | real_total_pubmed_hits | novelty | activity_magnitude | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| Benzamil hydrochloride | 7 | True | False | False | 0 | 27 | 601 | 1.0000 | 0.1495 | 216 |
| Sanguinarine chloride | 7 | False | True | False | 3 | 37 | 1188 | 0.7993 | 0.4098 | 25 |
| Valganciclovir hydrochloride hydrate | 2 | True | False | False | 0 | 66 | 2070 | 1.0000 | 0.0725 | 327 |
| Agmatine sulfate | 6 | False | False | True | 4 | 71 | 1917 | 0.7670 | 0.1141 | 37 |
| Azithromycin dihydrate | 7 | True | False | False | 0 | 95 | 15622 | 1.0000 | 0.1207 | 246 |
| Topotecan hydrochloride hydrate | 7 | False | True | False | 1 | 178 | 3947 | 0.8997 | 0.3055 | 7 |
| Zoledronic acid monohydrate | 6 | False | False | True | 2 | 351 | 6633 | 0.8410 | 0.0897 | 49 |
| Suramin sodium salt | 7 | False | True | False | 1 | 408 | 4786 | 0.8997 | 0.8722 | 1 |
| Y-27632 dihydrochloride | 6 | False | False | True | 4 | 559 | 3189 | 0.7670 | 0.3166 | 11 |
| Prazosin hydrochloride | 1 | False | False | True | 5 | 591 | 13267 | 0.7406 | 0.0975 | 67 |
| Sildenafil citrate salt | 6 | False | False | True | 5 | 835 | 9373 | 0.7406 | 0.1038 | 26 |
| Atorvastatin calcium salt trihydrate | 6 | False | False | True | 5 | 1299 | 12862 | 0.7406 | 0.1406 | 22 |
| (S)-Propranolol hydrochloride | 4 | False | False | True | 2 | 1323 | 47783 | 0.8410 | 0.1167 | 12 |
| Serotonin hydrochloride | 6 | False | False | True | 1 | 3412 | 165454 | 0.8997 | 0.1310 | 19 |

## PARTIAL

| drug_name | cluster | set_dark | set_top_anti | set_top_pro | n_papers | real_angio_pubmed_hits | real_total_pubmed_hits | novelty | activity_magnitude | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| PD-180970 | 7 | False | True | False | 2 | 0 | 24 | 0.8410 | 0.7024 | 17 |
| NSC 95397 | 7 | False | True | False | 3 | 0 | 22 | 0.7993 | 0.7034 | 23 |
| cis-4-Aminocrotonic acid | 7 | False | True | False | 5 | 0 | 66 | 0.7406 | 0.7169 | 29 |
| Tetraethylthiuram disulfide | 7 | False | True | False | 1 | 1 | 161 | 0.8997 | 0.4879 | 5 |
| 7-Chloro-4-hydroxy-2-phenyl-1,8-naphthyridine | 7 | False | True | False | 3 | 1 | 1 | 0.7993 | 0.8235 | 9 |
| IMS2186 | 7 | False | True | False | 1 | 1 | 1 | 0.8997 | 0.4344 | 10 |
| IC 261 | 7 | False | True | False | 2 | 1 | 29 | 0.8410 | 0.7314 | 14 |
| Wiskostatin | 7 | False | True | False | 6 | 4 | 47 | 0.7183 | 0.5927 | 24 |
| 5HPP-33 | 7 | False | True | False | 5 | 5 | 16 | 0.7406 | 0.6122 | 3 |
| Daphnetin | 5 | False | False | True | 11 | 6 | 265 | 0.6403 | 0.1067 | 50 |
| RepSox | 4 | False | False | True | 7 | 11 | 100 | 0.6990 | 0.1556 | 43 |
| Varenicline tartrate | 4 | False | False | True | 6 | 16 | 2761 | 0.7183 | 0.1579 | 36 |
| Idarubicin | 7 | False | True | False | 12 | 24 | 2707 | 0.6287 | 0.4103 | 18 |
| Piperlongumine | 7 | False | True | False | 13 | 31 | 531 | 0.6180 | 0.4415 | 8 |
| Bay 11-7085 | 7 | False | True | False | 14 | 35 | 241 | 0.6080 | 0.7584 | 13 |
| Benidipine hydrochloride | 3 | False | False | True | 6 | 36 | 376 | 0.7183 | 0.1202 | 27 |
| Podophyllotoxin | 7 | False | True | False | 12 | 52 | 3193 | 0.6287 | 0.4456 | 20 |
| Bay 11-7082 | 7 | False | True | False | 15 | 82 | 806 | 0.5986 | 0.8308 | 30 |
| Hydroquinone | 7 | False | True | False | 12 | 114 | 7475 | 0.6287 | 0.8007 | 2 |
| Raloxifene hydrochloride | 7 | False | True | False | 13 | 169 | 4489 | 0.6180 | 0.2938 | 16 |
| 2-methoxyestradiol | 7 | False | True | False | 10 | 301 | 1188 | 0.6529 | 0.4729 | 15 |
| Hydralazine hydrochloride | 3 | False | False | True | 10 | 316 | 6546 | 0.6529 | 0.0858 | 64 |
| Cilostazol | 3 | False | False | True | 14 | 326 | 2449 | 0.6080 | 0.0970 | 66 |
| Vinblastine sulfate salt | 7 | False | True | False | 7 | 335 | 17450 | 0.6990 | 0.6224 | 21 |
| Fenofibrate | 5 | False | False | True | 15 | 355 | 4817 | 0.5986 | 0.1459 | 47 |
| Vincristine sulfate | 7 | False | True | False | 12 | 434 | 35720 | 0.6287 | 0.5366 | 28 |
| Colchicine | 7 | False | True | False | 14 | 557 | 24452 | 0.6080 | 0.6158 | 4 |
| Forskolin | 4 | False | False | True | 15 | 898 | 20272 | 0.5986 | 0.1841 | 35 |
| Melatonin | 5 | False | False | True | 12 | 928 | 37185 | 0.6287 | 0.1218 | 57 |
| (±)-Epinephrine hydrochloride | 6 | False | False | True | 6 | 1000 | 75374 | 0.7183 | 0.1360 | 33 |
| Tamoxifen citrate | 6 | False | False | True | 11 | 1126 | 32737 | 0.6403 | 0.1242 | 54 |
| Tamoxifen | 6 | False | False | True | 13 | 1126 | 32737 | 0.6180 | 0.1242 | 59 |
| (±)-Propranolol hydrochloride | 4 | False | False | True | 12 | 1323 | 47783 | 0.6287 | 0.1167 | 51 |
| Resveratrol | 2 | False | False | True | 13 | 1654 | 21224 | 0.6180 | 0.0962 | 61 |
| Sunitinib malate | 7 | False | True | False | 8 | 2860 | 8400 | 0.6819 | 0.3385 | 6 |
| Histamine dihydrochloride | 2 | False | False | True | 8 | 3077 | 95650 | 0.6819 | 0.1282 | 31 |
| Indomethacin | 4 | False | False | True | 14 | 5015 | 45211 | 0.6080 | 0.1526 | 42 |
| Phenylephrine hydrochloride | 4 | False | False | True | 9 | 5077 | 24415 | 0.6667 | 0.1155 | 39 |
