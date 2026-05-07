[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q78Fi597)
# Difference-in-Differences with Staggered Adoption

This assignment asks you to reproduce the core multi-period difference-in-differences ideas from the lecture source in one reproducible Python pipeline.

## Learning goals

You will work with a provided balanced-panel data-generating process, summarize cohort shares and cohort-specific average treatment effects, estimate cohort-event DID effects with never-treated and not-yet-treated controls, and compare these corrected estimates with a pooled two-way fixed effects coefficient.

## Project structure

```text
difference_in_differences_multiperiods/
├── .github/workflows/classroom.yml
├── config/assignment.json
├── input/
├── cleaned/
├── output/
├── report/solution.qmd
├── scripts/
│   ├── run_cleaning.py
│   ├── run_analysis.py
│   ├── run_pipeline.py
│   └── run_assignment.py
├── src/
│   ├── __init__.py
│   └── did_multiperiod.py
├── tests/
│   ├── test_core.py
│   ├── test_report.py
│   └── test_workflow_modifiable.py
├── pyproject.toml
└── README.md
```

## Your tasks

All constants are defined in `config/assignment.json`. Use that file as the only source of configuration values.

The assignment follows the multi-period difference-in-differences design from `main/difference_in_differences_multiperiods.Rmd`. Units are observed for `T0` pre-treatment periods and `T1` post-treatment periods. Some units are never treated and treated units first receive treatment in cohorts `T0 + 1`, `T0 + 2`, `T0 + 3`, or `T0 + 4`. The functions `logistic` and `generate_panel_data` are already implemented for you in `src/did_multiperiod.py`. They encode the lecture data-generating process and are used by `scripts/run_cleaning.py` to create `cleaned/panel_no_covariate_trend.csv` and `cleaned/panel_covariate_trend.csv`. Your graded work starts from the generated panels.

### Step 1: Provided Staggered-Adoption Data Generation

The provided data generator first creates two binary covariates from a uniform draw. In the notation of the lecture source, `x1_i` and `x2_i` form the ordered covariate types `(0,0)`, `(1,0)`, and `(1,1)`. Treatment timing is assigned by threshold probabilities from logistic indexes. Let
$$
\Lambda(\alpha_0),
$$
be the first threshold. The next two thresholds are
$$
\Lambda(\alpha_0 + \alpha_1 x_{1i})
$$
and
$$
\Lambda(\alpha_0 + \alpha_1(x_{1i} + x_{2i})),
$$
and the final threshold is
$$
\Lambda(\alpha_0 + \alpha_2(x_{1i} + x_{2i})).
$$
Here `adoption_intercept`, `adoption_slope_middle`, and `adoption_slope_late` are the corresponding config keys. The cohort variable `G_i` equals the first treated period for treated units and equals `0` for never-treated units.

The untreated potential outcome is
$$
y_{it}(0) = a_0 + 1\{G_i > 0\}(-a_i) + 1\{G_i = 0\}a_i + b_{it} + e_{it}.
$$
When `heterogeneous_trend` is false, the trend is common:
$$
b_{it} = t / T.
$$
When `heterogeneous_trend` is true, the trend depends on `x1_i` and `x2_i`, matching the lecture example where unconditional parallel trends fails:
$$
b_{it}
= (t/T)(1-x_{1i}-x_{2i})
-0.2(t/T)x_{1i}
-0.1(t/T)x_{2i}.
$$

Observed treatment is
$$
d_{it} = 1\{G_i > 0,\ t \geq G_i\}.
$$
Treatment effects are heterogeneous by cohort and time. Following the lecture source, the first treated cohort receives a positive effect, while later treated cohorts receive negative effects:
$$
\tau_{it}
=
\tau_t
\left(
|\tau_i|1\{G_i=T_0+1\}
-2.5|\tau_i|1\{G_i=T_0+2\}
-1.75|\tau_i|1\{G_i=T_0+3\}
-|\tau_i|1\{G_i=T_0+4\}
\right).
$$
Observed outcomes are
$$
y_{it} = y_{it}(0) + d_{it}\tau_{it}.
$$

The provided `generate_panel_data` returns exactly the columns `id`, `x1`, `x2`, `cohort`, `time`, `relative_time`, `d`, `y0`, `tau_it`, and `y`.

The remaining steps are the graded implementation tasks. Do not change the function names, argument names, or return-column names, because the pipeline and tests call these functions directly.

### Step 2: Summarize Cohort Shares and Treatment Effects

Implement `summarize_group_shares_and_att`. This step reproduces the lecture table that reports the fraction in each treatment group and the average treatment effect for that group. For cohort `g`, the cohort share is
$$
\widehat{p}_g
= \frac{1}{N}\sum_{i=1}^N 1\{G_i=g\},
$$
and the cohort-specific average treatment effect among treated observations is
$$
\widehat{ATT}_g
=
\frac{
\sum_{i=1}^N\sum_{t=1}^T 1\{G_i=g,\ d_{it}=1\}\tau_{it}
}{
\sum_{i=1}^N\sum_{t=1}^T 1\{G_i=g,\ d_{it}=1\}
}.
$$
Also report an `all_treated` row using all observations with `d_it = 1`. For this row, set `fraction` equal to the row-level treated share,
$$
\widehat{p}_{all}
:=
\frac{1}{NT}\sum_{i=1}^N\sum_{t=1}^T d_{it},
$$
and set `att` equal to the average of `tau_it` over all treated rows.

### Step 3: Estimate One Cohort-Event DID Effect

Implement `estimate_cohort_did`. For cohort `g` and event time `e`, define the target period as
$$
t = g + e
$$
and use `g - 1` as the baseline period. The treated group is the set of units with `G_i = g`. The DID estimate is
$$
\widehat{\theta}_{g,e}
=
\left(
\bar{Y}_{G=g,t}
- \bar{Y}_{G=g,g-1}
\right)
-
\left(
\bar{Y}_{C(g,t),t}
- \bar{Y}_{C(g,t),g-1}
\right),
$$
where `C(g,t)` is the control group. If `control_group` is `"never"`, then
$$
C(g,t)=\{i:G_i=0\}.
$$
If `control_group` is `"notyet"`, then
$$
C(g,t)=\{i:G_i=0\ \text{or}\ G_i>t\}.
$$
This second option follows the lecture discussion of using not-yet-treated units when all or most units eventually receive treatment. The rule applies at the target period `t = g + e`: any unit with `G_i > t` is not yet treated at that target period and is a valid not-yet-treated control, including for pre-treatment event times.

### Step 4: Build the Event-Study Table

Implement `estimate_event_study`. This step applies the previous DID calculation to every observed treated cohort and every configured event time. The output is a table of
$$
\left\{(g,e,\widehat{\theta}_{g,e}): g \in \mathcal{G},\ e \in \mathcal{E}\right\},
$$
where `G` is the set of treated cohorts and `E` is the event-time grid in `config/assignment.json`. Skip event times whose target period is outside the observed panel. Return rows sorted first by increasing `cohort` and then by increasing `event_time`.

### Step 5: Aggregate Post-Treatment Effects

Implement `aggregate_post_treatment_effects`. This step summarizes the post-treatment event-study estimates by averaging all estimates with nonnegative event time:
$$
\widehat{\theta}_{post}
=
\frac{1}{|\mathcal{I}_{post}|}
\sum_{(g,e)\in\mathcal{I}_{post}}
\widehat{\theta}_{g,e},
$$
where
$$
\mathcal{I}_{post}
=
\{(g,e): e \geq 0\}.
$$

### Step 6: Compare with a TWFE Coefficient

Implement `estimate_twfe_coefficient`. This step computes the pooled two-way fixed effects coefficient from the regression
$$
y_{it} = \alpha_i + \lambda_t + \beta d_{it} + u_{it}.
$$
Use the residualized representation. Let
$$
\ddot{Y}_{it}
=
Y_{it}
-\bar{Y}_{i\cdot}
-\bar{Y}_{\cdot t}
+\bar{Y}_{\cdot\cdot}
$$
and define `ddot{D}_{it}` analogously. The TWFE coefficient is
$$
\widehat{\beta}_{TWFE}
=
\frac{
\sum_i\sum_t \ddot{D}_{it}\ddot{Y}_{it}
}{
\sum_i\sum_t \ddot{D}_{it}^2
}.
$$
The point of the comparison is the same as in the lecture source: when treatment timing and treatment effects are heterogeneous, this pooled coefficient is not the same object as the cohort-event DID estimates.

Implement the required functions in `src/did_multiperiod.py` with the exact signatures below.

```python
def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame: ...
def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float: ...
def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame: ...
def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float: ...
def estimate_twfe_coefficient(data: pd.DataFrame) -> float: ...
```

`summarize_group_shares_and_att` must return columns `group`, `fraction`, and `att`. It should include one row for each treated cohort, named like `cohort_5`, and one row named `all_treated`.

`estimate_cohort_did` compares the outcome change from period `cohort - 1` to period `cohort + event_time` for one treated cohort against a valid control group. If `control_group` is `"never"`, use never-treated units. If `control_group` is `"notyet"`, use never-treated units and units whose cohort is after the target period.

`estimate_event_study` must call `estimate_cohort_did` for every configured cohort and event time that is observed in the panel. It must return columns `cohort`, `event_time`, and `estimate`, sorted by `cohort` and then `event_time`.

`aggregate_post_treatment_effects` must average the estimates with `event_time >= 0`.

`estimate_twfe_coefficient` must compute the coefficient from the two-way fixed effects regression of `y` on `d`. Implement it by residualizing both variables with respect to unit and time fixed effects, then regressing the residualized outcome on the residualized treatment.

After implementing all functions, run `scripts/run_pipeline.py` and complete `report/solution.qmd` using values from `output/results.json`.

## Report Requirements

The report file `report/solution.qmd` should remain a static Quarto document. Do not add executable code cells. After running the pipeline, read values from `output/results.json` and write the interpretation in prose.

The rendered report must contain sections with the strings `Group shares and treatment effects`, `Event-study DID estimates`, and `TWFE comparison`. It must also mention the result key `twfe_no_covariate_trend`.

The pipeline must write `output/results.json` with these keys:

```text
twfe_no_covariate_trend
twfe_covariate_trend
event_study_no_covariate_simple_att
event_study_covariate_never_simple_att
event_study_covariate_notyet_simple_att
group_summary_rows
event_study_rows
```

## Workflow

Step 1 is to accept the GitHub Classroom invitation, which creates your private repository under the course organization.

Step 2 is to clone your repository locally.

```bash
git clone https://github.com/HKUSTIO/<your-repo-name>.git
cd <your-repo-name>
```

Step 3 is to install dependencies.

```bash
uv sync
```

Step 4 is to implement the estimator and summary functions in `src/did_multiperiod.py`. You do not need to implement the data-generating process.

Step 5 is to run the pipeline and render the report.

```bash
uv run python scripts/run_pipeline.py
uv run quarto render report/solution.qmd
```

Step 6 is to commit and push your submission.

```bash
git add -A
git commit -m "your message"
git push
```

Step 7 is to check scores. Every push to `main` triggers the `Autograding Tests` workflow on GitHub Actions. Open the latest run in the Actions tab to see component scores and total points, and check the GitHub Classroom dashboard for the recorded score.

You can resubmit before the deadline by pushing again. The latest score is recorded, and there is no penalty for multiple submissions.

## Grading policy

Grading checks that the provided data-generating process still works, and then grades group summaries, cohort-event DID estimates, TWFE residualization, required pipeline files, report rendering, and required result keys in `output/results.json`. Commit-message style and other immutable history details are not grading targets.
