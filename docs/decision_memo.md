# Decision memo: which loans should we fund

**TL;DR:** Fund almost every loan the model does not flag as high risk when borrowing money is cheap. Pull back to a smaller, safer slice only when the cost of the money you are lending goes up. The three numbers below show what you give up and what you avoid at each choice.

## Question

Given a pool of accepted, already-underwritten LendingClub loan applications, which ones should an investor actually fund, using only information available at the time each loan was issued.

## Recommendation

- **Cost of funds near 0%:** fund almost everyone the model does not flag as high risk. Across all four tested models, the dollar-maximizing selection rate is 95 to 100% of the pool (`reports/metrics/posthoc_funding_optimization_optimal.csv`).
- **Cost of funds around 4%:** pull back to roughly the top 85% by the champion's own score. That is still the dollar-maximizing point in this analysis, not a cautious guess.
- **Cost of funds around 8%:** this project's approximate cost model shows every tested funding level losing money, including the smallest one tested (5% of the pool). That is a statement about the cost model and this loan population at this approximate rate, not a recommendation to fund 5% anyway. At that cost of capital, the case for funding from this book is weak everywhere tested.

This is descriptive, not a new policy: the deployed cutoff is the one already chosen on validation (2014) and fixed before test was read (`reports/metrics/decision_policy.json`). The X% figures above come from a separate, post-hoc sweep of realized net cash after an assumed cost of funds, run after the real policy was already fixed.

## Trade-off: what each approval rate actually buys

Three points from the champion's own decision-quality view of test (2015, `reports/metrics/test_decision_quality.csv`). These approval rates are exact: the lowest-risk 70% and 90% of test loans by the champion's own score, ranked directly on test:

| Approval rate (exact, ranked on test) | Default rate among approved | % of actual defaults caught by declining | Good loans declined per default avoided | Realized net cash on the approved book |
|---|---|---|---|---|
| 70.00% | 10.40% | 51.10% | 2.94 | $196.9M |
| 90.00% | 12.99% | 21.44% | 2.13 | $228.9M |
| 99.41% (chosen policy) | 14.70% | 1.79% | 1.23 | $231.6M |

A different "70%" and "90%" figure appears in `docs/validation_report.md`'s "Fixed selection-rate sweep": a PD cutoff learned on validation (2014) and applied unchanged to test, which lands at a slightly different actual rate (69.91% and 89.47% for the champion) and a slightly different dollar figure ($196.8M and $228.3M), because it is not re-ranked on test's own score. Both tables are correct under their own definition; they answer different questions (what a fixed PD cutoff does vs. what an exact top-X% selection does), so treat the two 70% figures as reporting on two different cutoffs, not as a disagreement.

Declining more loans catches more of the defaults that happen, but it also turns away a much larger number of good loans for every default avoided, and it shrinks the realized dollars on the approved book, because most declined loans would have been fine and paid interest. The chosen, EV-maximizing policy approves almost everyone (99.41%) because, under this policy's own assumptions (no cost of funds, no operating cost), the interest paid by the many good loans in this population outweighs the loss from the relatively few that default. That balance does not hold once a real cost of funds is assumed: at 4%, the dollar-maximizing rate drops to around 85%, and at 8%, this project's approximate cost model shows every tested funding level losing money (see Recommendation above).

## Risks

- **Default-rate drift:** the decision policy's dollar assumptions were fixed on 2014, and the 2015 realized default rate ran 1.15 points higher (14.88% vs. 13.73%). This single shift explains 49.1% of the $44.6M gap between the policy's expected and realized value on test.
- **Prepayment:** fully paid test loans returned less of their scheduled interest than train-year fully paid loans did (80.24% vs. 82.62%). Taken together with loss severity, these two cash-flow assumptions account for a further 45.4% of the same gap.
- **Accepted-loans-only data:** every loan here already passed LendingClub's own underwriting. FICO stayed at or above 660 in every year used; the DTI cap held at 34.99% in train (2012-2013) but LendingClub raised it afterward, to 39.99% by 2014 and as high as 49.96% in 2016 (`reports/metrics/eligibility_caps_by_year.csv`). This model has never seen a rejected applicant, so it cannot be used to argue LendingClub's own accept/reject line is right or wrong, only to rank and price within what it already accepted.

## Next steps

- **Recalibration trigger:** once a cohort of funded loans has seasoned past 24 months, check the calibration-in-the-large gap (mean predicted PD minus actual default rate) on that cohort. A gap beyond roughly 2 percentage points (the largest seen in-sample on test is 1.89 to 1.90 points) is a starting point for triggering recalibration, not a validated operational threshold yet.
- **Monthly default-rate movement:** 7 of 11 month-over-month changes in the 2015 default rate exceed 95% sampling noise, so most moves are real. None trace to a single compositional driver (FICO band or purpose mix), and the model's own predicted PD explains only a small share of each move's size (`reports/metrics/kpi_movement_summary.json`). These read as vintage effects the model's features do not capture; the recalibration trigger above is the mechanism for catching one that grows into an actual calibration problem.
- **Monitoring thresholds:** PSI per feature, monthly, against the conventional bands (under 0.10 stable, 0.10 to 0.25 watch, over 0.25 alert). 23 of 69 features had crossed watch and 6 had crossed alert by the end of the 2016 to 2018 replay window; none of the top-10 SHAP features had reached alert.

## A note on which model this memo describes

The trade-off table and the risk figures above are v1's results, evaluated on test (2015) exactly once. v2, refit with monotone constraints and chosen by a pre-registered rule on 2016 vintages, is the model actually serving predictions today. v1's 2015 numbers are the only ones available at this level of detail (bootstrap CIs, decision-quality breakdown, the funding-fraction sweep); v2's own evaluation is in `docs/validation_report.md` and `docs/model_card.md`.
