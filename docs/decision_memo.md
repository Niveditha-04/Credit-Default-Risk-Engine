# Decision memo: which loans to fund

**TL;DR:** Fund almost every loan the model does not flag as high risk when the cost of funds is low. Pull back to a smaller, safer slice as that cost rises. The table below shows what each choice gives up and avoids.

## Question

Given a pool of accepted, already-underwritten LendingClub loans, which ones should an investor fund, using only information available at origination.

## Recommendation

| Cost of funds | Dollar-maximizing rate | Note |
|---|---|---|
| ~0% | **95-100%** of the pool | Fund almost everyone not flagged high risk |
| ~4% | **~85%** by the champion's own score | Still the dollar-maximizing point, not a guess |
| ~8% | Every tested level loses money | Weak case for funding this book at this cost |

This is descriptive, not a new policy: the deployed cutoff was chosen on validation (2014) and fixed before test was read (`reports/metrics/decision_policy.json`). The rates above come from a separate, post-hoc sweep run after that policy was fixed.

## Trade-off: what each approval rate buys

LightGBM's own decision-quality view of test (2015), exact rank (`reports/metrics/test_decision_quality.csv`):

| Approval rate | Default rate approved | Defaults caught by declining | Good loans declined per default avoided | Realized net cash |
|---|---|---|---|---|
| 70.00% | 10.40% | **51.10%** | **2.94** | $196.9M |
| 90.00% | 12.99% | 21.44% | 2.13 | $228.9M |
| 99.41% (chosen policy) | 14.70% | 1.79% | 1.23 | **$231.6M** |

A different "70%" in `docs/validation_report.md`'s fixed-cutoff sweep (69.91% actual, $196.8M): both correct under their own definition, not a disagreement.

- Declining more loans catches more defaults, but turns away far more good loans per default avoided, and shrinks realized dollars.
- The chosen, EV-maximizing policy approves **99.41%** because, under its own assumptions (no cost of funds, no operating cost), interest from good loans outweighs losses from bad ones. That balance breaks once a real cost of funds is assumed (Recommendation table above).

## Risks

| Risk | Finding |
|---|---|
| Default-rate drift | Realized 2015 default rate ran **1.15 pp** above the 2014 assumption (14.88% vs. 13.73%); explains **49.1%** of a $44.6M gap |
| Prepayment and loss severity | Test's realized interest (80.24%) and loss severity ran worse than train's assumption; **45.4%** of the same gap |
| Accepted-loans-only data | FICO floor held at 660 every year used; DTI cap rose from 34.99% (train) to 49.96% (2016, `reports/metrics/eligibility_caps_by_year.csv`). No rejected applicant ever seen, so this cannot judge LendingClub's own accept line |

## Next steps

- **Recalibration trigger:** once a funded cohort seasons past 24 months, check the calibration-in-the-large gap; **~2 pp** is a starting threshold (largest seen in-sample: 1.89-1.90 pp), not yet validated operationally.
- **Monthly default-rate movement:** **7 of 11** month-over-month 2015 changes exceed sampling noise. No single compositional driver; the model's predicted PD explains only a small share (`reports/metrics/kpi_movement_summary.json`). Read as vintage effects to monitor via the recalibration trigger.
- **Monitoring thresholds:** PSI per feature, monthly. By the end of the 2016-2018 replay, **23 of 69** features crossed watch and **6** crossed alert; none of the top-10 SHAP features reached alert.

## Which model this memo describes

The trade-off table and risk figures are v1's, evaluated on test (2015) exactly once. v2, refit with monotone constraints and chosen by a pre-registered rule on 2016 vintages, is the model serving predictions now. v1's 2015 numbers are the only ones at this level of detail; v2's own evaluation is in `docs/validation_report.md` and `docs/model_card.md`.
