# Proposed email pilot: can the campaign cost customers a purchase?

**Decision for the current campaign owner:** Could assigning one specific marketing email, rather than no campaign email, cause more than 10 of every 1,000 eligible customers to miss a purchase they would otherwise make in 14 days? A rise in total purchases does not answer that question. Everything below is a proposal or a synthetic example; it authorizes no traffic, access to outcomes, or business action.

The [fraction-harmed study](harm_study_result.md) deferred implementation. The historical Hillstrom men's-email versus no-email comparison was unavailable (`missing_case_manifest`); permission to analyze and report its results and the trial design were not verified. Even with permission, it would be a historical benchmark, not a proxy for a current retailer, email, audience, purchase rate, tolerance, or decision. This pilot would need its own owner and evidence.

## Run one matched trial

1. **Name the target and freeze enrollment.** Target the sponsor's contactable, eligible customers at one named entry time `T0` (UTC), in the markets and sales channels where the owner would use this exact email, with at least one qualifying purchase in `[T0 − 12 calendar months, T0)`. Freeze the eligibility rules, channels, dates, fixed even enrollment count and sampling procedure before enrollment. Document how the enrolled accounts represent that population and support the [frozen protocol's](harm_study_v1.md) independent-unit, same-population sampling assumption. A balanced assignment or a finite roster alone does not establish it. Do not generalize to unrepresented customers or future campaigns.
2. **Assign once, by stable account.** Before assignment reconcile customer identities across channels to one stable account key per customer; suppress duplicate enrollment and exclude ineligible accounts up front. Give a uniformly random half of the resulting enrolled accounts assignment `1` (send the locked email at entry), and the other half assignment `0` (no campaign email). Freeze the creative, offer, send time and sender; use complete 1:1 randomization independent of account records, not blocked or targeted by features. Log the assignment once. Suppress overlapping marketing campaigns in both arms throughout follow-up, leave ordinary transactional messages unchanged, and document protections against spillover such as shared accounts or a transferable offer. Do not send a replacement if delivery fails. Analyze every account by its original assignment (intention-to-treat), regardless of delivery, opt-out, opens, clicks, or later eligibility.
3. **Capture one outcome and make one cut.** Set `y=1` if at least one qualifying paid order is recorded at or after assignment and before `T0 + 14 × 24 hours`, otherwise `y=0`. Lock the qualifying-order definition and event clock before assignment. Capture purchases across the declared channels, not just purchases attributed to an email click. Reconcile the locked assignment roster with the order feeds, including store and online identities where in scope, prove coverage in both arms through the window, and freeze one cut after a prespecified ingestion lag. A zero requires verified complete capture, not an unmatched account or absent feed. If there are duplicate trial IDs, unresolved identity links, or missing follow-up, stop and document the failure; do not drop records or impute zeros. No interim outcome looks, outcome changes or alternate cut dates.
4. **Fix the baseline comparisons.** Let `S=1` mean exactly one qualifying purchase in the 12-month eligibility window and `S=0` mean two or more. Add `B=1` for at least one qualifying purchase in `[T0 − 6 calendar months, T0)` and `B=0` otherwise. Freeze these definitions and the cut at six months before outcomes; do not search for segments. Compare the full trial pooled (`U0`), by `S` (`F`), and by `(S,B)` (`Q`), using group shares estimated across both arms. These are three ways to bound the **same population fraction**, not separate permissions to target subgroups.

## Obtain authority before unblinding

Ask the owner with authority over the current send decision for the following, independently of the analysis results:

- **A signed, matched `DecisionCard`.** Name the owner and decision ID. Record time-zone-aware signing and unblinding times, with signature strictly before unblinding; agree and sign before assignment for this pilot. Match the trial's target, ordered arms, outcome, horizon, `one_is_beneficial` direction and permission reference exactly; cite representativeness evidence. Pin the card's SHA-256 before opening outcomes. Confirm or replace the proposed tolerance, risk ceiling, current action, alternatives, costs, cost bearers and minimum gain below *before* results are available. If the owner requires a false-action risk ceiling below `.05`, do not use this frozen procedure for the decision.
- **Trial evidence and permission.** Ask the data custodian for dated authority to analyze this local input and report aggregates, plus the unit and assignment records, design and sampling justification, and signed complete-follow-up reconciliation. Stage the restricted `owner_trial` CSV with exactly `unit_id,arm,y,s,b`, one row per ID, and `0`/`1` for the four binary fields. Supply the manifest fields `csv_path`, `expected_sha256`, `permission_ref`, `design_ref`, `followup_ref`, `unit_ref`, `sampling_target`, `analysis_permitted`, `aggregates_permitted`. The local design JSON records `arms`, `outcome`, `horizon`, `positive_direction`, `assignment` (`complete`), `treatment_probability` (`0.5`) and `assumptions_ref`. Independently verify permission and the design evidence before access. Have the custodian seal the cut and supply an independently verified SHA-256; `harm_case` checks it before parsing outcomes. Keep raw records private; retain only the required aggregate readout and receipts for the recommendation. A valid digest does not confer permission or prove completeness.

**Proposed decision terms, not approved:** `tau=.01` (10 lost purchases per 1,000 eligible accounts); `max_false_action_risk=.05` under the frozen assumptions. The current action (`abstain`) is to keep the send on hold and reserve time and resources for a decision. If **clear**, pass only the purchase-harm check and put the send forward for separate review. If **rule out**, close this email variant for this audience and release the reserved resources; no send. If **inconclusive**, keep the current hold; do not reopen this cut to find a better answer. Closing the variant is a different decision from waiting even though neither sends an email now.

Use illustrative sponsor losses in USD per campaign decision: `$120,000` for false clearance, `$60,000` for false rule-out, `$20,000` for waiting, `$5,000` incremental workflow cost and `$10,000` minimum gain. Have the owner validate the cost of lost purchases, missed opportunity, delay and analysis, and explicitly name who bears those costs for each action. These amounts are not valuations of participant welfare, forecasts, or estimates obtained from the example.

## Dry run with synthetic counts only

Let `p0` and `p1` be the probabilities of purchase under no email and email, respectively. In each baseline group the sharp harm bounds are `max(0, p0 − p1)` and `min(p0, 1 − p1)`. If `p0+p1 ≤ 1` in every group, all group upper bounds equal their control rate, so their weighted sum is the pooled control rate. The pooled upper bound equals that same rate. Splitting cannot lower that **population** upper bound in this case. But the lower bound can rise when losses in some groups were masked by gains in others.

Take 25,000 accounts per arm in each of the four atoms (200,000 total). These round counts are plausible at a large retailer and chosen to show the decision mechanics, not to recommend a sample size. The table gives exact purchase counts, with rates per 1,000 in parentheses. Assume these rates and equal atom shares for the exact population calculation; for the sampling bands, treat the counts as a possible realization and **estimate**, rather than assume, the shares.

| `(S,B)` | No email, both cases | Email: opposing effects | Email: contrast |
| --- | ---: | ---: | ---: |
| `00` | 450 (18) | 1,375 (55) | 600 (24) |
| `01` | 900 (36) | 175 (7) | 950 (38) |
| `10` | 250 (10) | 1,100 (44) | 375 (15) |
| `11` | 800 (32) | 125 (5) | 850 (34) |

In both cases the pooled rates are **24 vs 27.75 per 1,000**. Even within each `S` group the email rate rises (27 to 31 for `S=0`; 21 to 24.5 for `S=1`). Thus `U0` and `F` give exact bounds `[0, 24]` per 1,000 in both cases. In the opposing-effects case, the four group lower bounds are `0, 29, 0, 27` per 1,000. Averaging gives `Q=[14, 24]`. In the contrast all four email rates rise, so `Q=[0, 24]`, which is not a finding of no harm.

Run this from the repository root with the locked environment already installed. It reads no trial files or unit records and runs no certification campaign.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv --directory "$PWD" run --locked --no-sync python - <<'PY'
from calibration.harm_bounds import Cell, decide, summarize

maps = ((0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 2, 3))
control = (450, 900, 250, 800)
tau = 0.01


def per_thousand(interval):
    lo, hi = interval
    return f"[{1000 * lo:.3f}, {1000 * hi:.3f}]"


cases = (
    ("opposing effects", (1375, 175, 1100, 125), "rule_out"),
    ("contrast", (600, 950, 375, 850), "inconclusive"),
)
for name, treatment, expected in cases:
    print(name)
    counts = tuple(Cell(25000, c, 25000, t) for c, t in zip(control, treatment, strict=True))
    bands = [summarize(counts, mapping, atom_weights=None) for mapping in maps]
    for method, band in zip(("U0", "F", "Q"), bands, strict=True):
        print(method, "L:", per_thousand(band.L_ci), "U:", per_thousand(band.U_ci))
    for i, decision in enumerate(("inconclusive", "inconclusive", expected)):
        outer = (max(b.outer[0] for b in bands[: i + 1]), min(b.outer[1] for b in bands[: i + 1]))
        assert outer[0] <= outer[1]
        assert decide(outer, tau) == decision
        print(f"P{i}", per_thousand(outer), decide(outer, tau))
    if expected == "rule_out":
        lo, hi = outer
        for t in (lo, lo - 5e-13, hi, hi + 5e-13):
            assert decide(outer, t) == "inconclusive"
        assert decide(outer, lo - 2e-12) == "rule_out"
        assert decide(outer, hi + 2e-12) == "clear"
        print("Strict endpoint and near-equality checks passed")
    else:
        assert bands[2].L_ci[1] < tau < bands[2].U_ci[0]
        print("Q inner span straddles tolerance")
PY
```

The frozen binomial projections give these simultaneous sampling ranges for the two population endpoints, in losses per 1,000 (rounded for display):

| Method and case | Range for lower endpoint | Range for upper endpoint |
| --- | ---: | ---: |
| `U0`, both | [0.000, 0.000] | [22.603, 25.455] |
| `F`, both | [0.000, 0.724] | [21.888, 26.246] |
| `Q`, opposing effects | [11.100, 16.961] | [20.865, 27.451] |
| `Q`, contrast | [0.000, 2.963] | [20.865, 27.451] |

`P0=U0` and `P1=U0∩F` both abstain with outer band `[0, 25.455]`. With opposing effects, `P2=P1∩Q` rules out at 10 with `[11.100, 25.455]`. The change comes from raising the lower endpoint; the enclosing upper endpoint still comes from `U0`. In the contrast `P2` also abstains: the inner span `[2.963, 20.865]` straddles 10. On the joint coverage event, more data at this feature resolution cannot resolve a threshold strictly inside the population identified set. No individual harm or field error rate is observed. Decide from unrounded endpoints: clear strictly below the tolerance, rule out strictly above it, and abstain at equality, within `1e-12` in probability units of the relevant endpoint, or on discordant bands.

## Review the matched decision, not just the band

The illustrative opposition readout would change the action from waiting to ruling out. On the coverage event, every compatible harm value makes rule-out correct, so the worst compatible gain versus waiting is `$20,000 − $0 − $5,000 = $15,000`, strictly above `$10,000`. In the contrast, waiting leaves the action unchanged; guessing clear or rule-out can cost `$100,000` or `$40,000` more than waiting, respectively, before the extra workflow cost. This is a worst-case comparison across compatible harm values using the proposed loss schedule. Equality at the tolerance is wrong for either strict claim. It is **not** expected profit, an observed reduction in harm, or an estimate from a simulation of current buyers.

At the locked, authorized readout, use the [frozen field comparison](harm_study_v1.md#independent-field-decision) to record all methods, both sampling ranges, decisions and costs. Prefer the simplest conclusive changed action that strictly exceeds the gain floor; the learned split (`H`) must change an otherwise inconclusive fixed-feature decision and meet the same utility gate. After the readout, obtain an independently authenticated acknowledgment of feasibility and intent for the selected pipeline and card action, dated after the readout and tied to the exact `decision.json` and card SHA-256 digests. `harm_verdict` checks those bindings and the existing calibration receipts without reading trial outcomes. If the comparison remains inconclusive, report why and keep the hold rather than adjusting the tolerance or looking again.

No matched trial, permission, signed card or acknowledgment is evidenced here. Two constructed count tables and the earlier bounded calibration are not an authorized field decision. Keep implementation deferred without the real evidence and approvals; even a qualifying result would support separately scoping the simplest method, not sign off on a send or a public implementation.
