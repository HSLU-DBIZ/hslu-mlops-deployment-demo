# Walkthrough 3 · Efficiency (SW09)

The lecture **SW09 Efficiency Techniques** asks two questions before any trick: *where does the
time go?* and *is the model worth its compute?* This walkthrough answers them for lucerne-temp,
with real data and real numbers, on your laptop. It follows the lecture: "Measure first" (steps 1
and 2), then the six stations of "Your scale" (steps 3 to 8).

Fully local: no GCP, no Docker, no API key. It does not depend on [EXERCISE.md](EXERCISE.md);
if you skipped the Cloud Run walkthrough, start here.

> The scripts in `scripts/efficiency/` show how *one* small forecaster answers these questions.
> Use them for ideas. Your project has another model, other data and other bottlenecks: take
> the questions and the habits, write your own experiments, and do not copy these 1:1.

| Step | What | Lecture station | Time |
|---|---|---|---|
| 0 | Set up, fetch three years of history | | 5 min |
| 1 | Measure, then benchmark honestly | Measure first | 5 min |
| 2 | Profile before you optimise | Measure first | 5 min |
| 3 | Baseline ladder: is the model worth its compute? | Ladder | 5 min |
| 4 | Honest validation | Validation | 5 min |
| 5 | Speed knobs for trees, then search with a budget | Search | 10 min (7 min waiting) |
| 6 | How much history is enough? | History | 5 min |
| 7 | Retraining policy | Retrain | 5 min |
| 8 | Incremental features, storage format, float32 | Features | 5 min |

> **Status of this document.** Every command below was executed on 22.10.2026 on an Apple
> M5 Pro laptop (18 cores: 6 performance, 12 efficiency; 48 GB; macOS 26.7.1) with Python
> 3.12.13, LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.5 and Optuna 5.0.0. Each "You should
> see" block is pasted from that run. **Times will differ on your machine** (cores, threads,
> what else is running), and some of them differed between two runs on ours; the step says so
> where it matters. MAE values should agree to about the second decimal. If a command fails,
> read the error, and tell us.

All commands assume a bash or zsh shell in the repository root. On Windows use WSL.

## The data

Steps 1 and 2 use what the service uses: the committed 70-day snapshot and the config of v1.

Steps 3 to 8 need more history than 70 days. `src/lucerne_temp/features/history.py` fetches
three years of hourly temperature for Lucerne from the
[Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) (reanalysis,
no key) in one call: 1.10.2023 to 30.9.2026, 26,304 hours. It caches them in
`data/history/temperature.parquet` (256 KB, git-ignored), so later steps are offline.

- **Split:** the first two years train (Oct 2023 to Sep 2025), the last full year tests
  (Oct 2025 to Sep 2026): every season is in the test, and no training target lies in it.
- **Rows:** as in `train.py`, one model for all horizons. Every hour gives 24 rows
  (`hours_ahead` 1 to 24): 420,465 training rows, 209,940 test rows.
- **Features and target:** the 9 features of v2 (`build_features()`, feature set `daily`), and
  every model predicts the change from `temp_now` (the package's `Forecaster`). The scripts
  import these from the package; nothing is copied.
- **Metric:** MAE in °C over all horizons, as in the README; per horizon where it matters.

Archive data are reanalysis, not the forecast API the service reads, so these numbers are not
comparable with the 2.3 °C in the README (other source, other season, 14 days vs a year).
The service, v1, v2 and the tests do not change: they still train on the snapshot.

## 0 · Set up

1. Install the extra packages (LightGBM, Optuna, pyarrow) from the dependency group
   `efficiency`. The service's own install does not change. On macOS, LightGBM needs OpenMP
   first.

```bash
brew install libomp                                   # macOS only
uv sync --group efficiency
```

2. Fetch the three years (one call, under a second).

```bash
uv run --group efficiency python -m lucerne_temp.features.history
```

   You should see:

```
fetched 26304 hours, 2023-10-01 00:00:00+00:00 to 2026-09-30 23:00:00+00:00 -> data/history/temperature.parquet
26304 hours, 2023-10-01 00:00:00+00:00 to 2026-09-30 23:00:00+00:00, temperature.parquet 256 KB
```

   Run it again: no "fetched" line, it reads the cache.

Watch out:

- `Library not loaded: @rpath/libomp.dylib` means OpenMP is missing: `brew install libomp`.
- Always run with `--group efficiency`. A plain `uv sync` removes the group again.
- The raw JSON response is kept next to the table in `data/history/raw/` (606 KB). Step 8
  says why.

## 1 · Measure, then benchmark honestly

Measure the whole training run, not your feeling about it: wall time over several runs, peak
memory, artifact size. Then benchmark prediction the way the service calls it: the first call
after loading, warm single rows (median and p95 over 200 calls after a warm-up), and batches.

```bash
uv run --group efficiency python scripts/efficiency/1_measure.py
```

You should see (times differ):

```
Measure: train v1 on the snapshot
                                    what          value
            train() wall time, first run        1826 ms
      train() wall time, runs 2-7 median        1344 ms
   train() wall time, runs 2-7 min / max 1173 / 1702 ms
train() wall time, 1 thread, median of 3         221 ms
       peak Python memory during train()          17 MB
        peak process memory (RSS) so far        224 MB
                           artifact size         374 KB

Benchmark: predict with the v1 artifact
                         call  median ms  p95 ms  us per row
        joblib.load(artifact)      8.186   8.186      8185.9
         first predict, 1 row     37.175  37.175     37175.2
   warm predict, 1 row (x200)     18.711  26.021     18710.7
warm predict, 1 row, 1 thread      0.721   0.900       720.6
    features + predict, 1 row     19.871  25.524     19871.2
        24 horizons, 24 calls    451.436 611.910     18809.8
          24 horizons, 1 call     18.148  20.979       756.2
       test window, 7788 rows     19.512  21.300         2.5
```

Read it:

- **The first run is slower** (imports, caches). Never report a single run; report the median
  and the spread. Here the "same" run varied between 1.2 and 1.7 s.
- **One thread trains six times faster than eighteen.** `HistGradientBoostingRegressor` uses
  every core by default (OpenMP). On 40,000 rows, starting and synchronising threads costs more
  than the work. The same holds for prediction: 0.7 ms with one thread, 5 to 19 ms with all
  of them (4.7 ms median in an earlier run, 18.7 ms in this one, and a p95 of 26 ms in both).
- **Batch, don't loop.** 24 horizons in one `predict()` cost about as much as one row; 24
  separate calls cost 24 times as much.
- Peak Python memory (17 MB, `tracemalloc`) is what your code allocates; peak RSS (224 MB) is
  the whole process, mostly libraries. A container needs the second.

Watch out: `tracemalloc` makes code slower; measure time and memory in separate runs, as the
script does. The p95 of a laptop is noisy: close the browser before you compare two numbers.

Think: Cloud Run gives the service 1 vCPU by default. Which of the numbers above would you
expect there, and how would you check the thread count in your own container?

## 2 · Profile before you optimise

Where does the time go? First the quick way: profile any module, sorted by the time spent
inside each function (`tottime`).

```bash
uv run --group efficiency python -m cProfile -s tottime -m lucerne_temp.training.train | grep -A 10 ncalls
```

You should see (times differ):

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     2840    0.884    0.000    1.132    0.000 grower.py:486(split_next)
      189    0.288    0.002    0.288    0.002 {built-in method _imp.create_dynamic}
     2844    0.252    0.000    0.253    0.000 grower.py:461(_compute_best_split_and_push)
     1365    0.177    0.000    0.177    0.000 {built-in method _io.open_code}
     5684    0.070    0.000    0.070    0.000 {built-in method posix.stat}
     1365    0.056    0.000    0.056    0.000 {built-in method marshal.loads}
      402    0.037    0.000    0.037    0.000 loss.py:179(loss)
        1    0.028    0.028    1.363    1.363 gradient_boosting.py:393(fit)
      178    0.025    0.000    0.025    0.000 {built-in method posix.listdir}
    13157    0.024    0.000    0.024    0.000 {method 'sub' of 're.Pattern' objects}
```

The tree grower dominates, and `create_dynamic`, `open_code`, `marshal.loads` are Python
importing modules: about half a second before any work starts.

Now the typical case: the feature code is slow. The script swaps `build_features()` for a
version written the way it often is, with a Python call per rolling window
(`rolling(6).apply(lambda w: w.mean())`) and per row (`apply(..., axis=1)` for the hour), and
runs the same `train()` with and without the profiler.

```bash
uv run --group efficiency python scripts/efficiency/2_profile.py
```

You should see (times differ):

```
slow and fast build the same features: True

Without profiler (wall, median of 3) and with (share of the profiled run)
      version  wall s  of it features s  profiled s  load %  features %  train %  evaluate %  save %   MAE
   as shipped    1.33              0.01        1.34       2           3       92           1       3 2.315
slow features    1.72              0.51        2.84       0          51       47           1       1 2.314

Slow version, top 8 functions by own time (tottime):
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     2840    0.886    0.000    1.134    0.000 grower.py:486(split_next)
     2841    0.252    0.000    0.253    0.000 grower.py:461(_compute_best_split_and_push)
1158311/1157999    0.073    0.000    0.117    0.000 {built-in method builtins.isinstance}
    40788    0.054    0.000    0.113    0.000 base.py:849(_engine)
    40827    0.054    0.000    0.078    0.000 generic.py:6119(__finalize__)
122888/122887    0.052    0.000    0.052    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    40536    0.046    0.000    0.433    0.000 managers.py:2174(get_slice)
    81360    0.038    0.000    0.090    0.000 apply.py:1348(series_generator)
```

Read it:

- Same features, same MAE (2.315 vs 2.314 is float rounding), **50 times slower features**
  (0.51 s vs 0.01 s). On 70 days that is half a second; on three years of history it would be about 16 times more.
- The give-away in the profile: **call counts**. 40,000 calls of `get_slice` and `_engine` are
  one per window and row (24 horizons x 1,680 hours). Vectorised code calls each of them a
  handful of times.
- **The profiler changes what it measures.** The slow version takes 1.72 s without the profiler
  and 2.84 s with it: Python-heavy code pays the most. Use the profile for *where*, the wall
  time for *how much*.

Watch out: `cProfile` sees Python functions only. For time spent in C code, threads or a
running service, use a sampling profiler such as
[py-spy](https://github.com/benfred/py-spy) (`py-spy record -o profile.svg -- python ...`;
on macOS it needs `sudo`). Not used here.

Think: in the shipped version, 92 % of the run is training. Which knob from step 1 would you
try first, and what would you measure before and after?

## 3 · Baseline ladder: is the model worth its compute?

Climb from free to expensive and stop when a rung no longer pays. Two baselines that cost
nothing, a linear model, two tree models, a small neural net. Same rows, same features, same
target for all of them.

```bash
uv run --group efficiency python scripts/efficiency/3_ladder.py
```

You should see (train times differ):

```
train: 420465 rows, feature times 2023-10-01 05:00:00+00:00 to 2025-09-30 22:00:00+00:00
test: 209940 rows, feature times 2025-10-01 00:00:00+00:00 to 2026-09-30 22:00:00+00:00

Test year, MAE in °C (one fit each)
                          model   MAE  MAE 1h  MAE 6h  MAE 24h  train s  size KB
         Persistence (temp_now) 3.298   0.773   3.406    1.841     0.00        0
            Same hour yesterday 1.841   1.840   1.841    1.841     0.00        0
        Ridge (scaler + linear) 1.808   1.478   1.733    1.925     0.03        2
HistGradientBoosting, v2 params 1.689   0.809   1.587    1.907     1.64      392
            LightGBM, 400 trees 1.614   0.702   1.531    1.873     1.38     1149
   MLP 64-32 (small neural net) 1.520   0.568   1.421    1.845    31.06       76

MAE check: 3.298 = persistence row
```

Read it:

- **"Same hour yesterday" is a strong baseline**: 1.84 °C for free, at every horizon. Ridge
  beats it overall by only 0.03 °C, and loses to it at 24 h.
- **At 1 h, persistence (0.77) is hard to beat.** Ridge (1.48) and the shipped tree config
  (0.81) lose to it; LightGBM wins by 0.07 °C. Your forecast should know per horizon which rung
  wins, as the README's table does.
- **The trees cost 50 times the training time of Ridge** for 0.12 to 0.19 °C. The small neural
  net is best (1.52), at 1,000 times Ridge's training time. On a laptop, 31 s is still nothing;
  in a weekly retrain it is nothing either. Whether 0.1 °C is worth it is a product question,
  not a technical one.

Watch out: one fit each, no tuning. The MLP is sensitive to its seed and scaling; the trees
are not tuned (step 5). Do not rank models that are 0.01 apart.

Think: which rung would you ship, and what would make you change your mind? Write down your
own project's ladder with its first two rungs before you train anything expensive.

## 4 · Honest validation

The same LightGBM model scored five ways on the two training years, next to the test year it
never saw. `src/lucerne_temp/training/folds.py` cuts by time: a test block of 60 days, and a
training row is only used if its target was observed before the block (gap = horizon).

```bash
uv run --group efficiency python scripts/efficiency/4_validation.py
```

You should see:

```
LightGBM, 5 folds (60 days each for the time-ordered ones)
                                     scheme  CV MAE
                     random KFold, shuffled   1.553
TimeSeriesSplit, rows in time order, no gap   1.633
                         time folds, no gap   1.634
                  time folds, gap = horizon   1.632
                     holdout: the test year   1.614

Preprocessing inside or outside the folds (time folds, gap = horizon)
                         Ridge  CV MAE
scaler fitted before the split  1.7845
    scaler inside the Pipeline  1.7845
```

Read it:

- **Random KFold is optimistic** (1.553 against about 1.63): every test hour has its
  neighbours, and the same hour at other horizons, in training. The time-ordered folds agree
  with each other and are close to the test year (1.614).
- **The gap made no difference here**, because a 24-hour overlap is small next to a 60-day
  block. It matters with short test blocks, long horizons, or features that look far ahead.
  Keep it: it costs nothing.
- **The scaler leak made no difference either.** A scaler fitted on all rows barely differs
  from one fitted per fold, and Ridge does not care. Leaks that do hurt are steps that learn
  from the target or from the future: target encoding, imputing with the overall mean,
  feature selection on all rows. Put every learned step in a `Pipeline`, so the question never
  comes up.

Watch out: `make_dataset()` stacks the 24 horizons, so rows are not in time order.
`TimeSeriesSplit` silently assumes they are; the script sorts them first. Cut by timestamps,
not by row numbers.

Think: your project reports one validation number in MS3. Which scheme produced it, and would
it survive this table?

## 5 · Speed knobs for trees, then search with a budget

1. Make each fit cheap before you search: a search multiplies every fit by the number of
   trials. LightGBM on the training years, early stopping on the last 90 days of them, each
   setting fitted 3 times. Takes about 2.5 minutes.

```bash
uv run --group efficiency python scripts/efficiency/5a_knobs.py
```

   You should see (times differ):

```
LightGBM, lr 0.05, n_jobs=4 unless stated; fit = 364884 rows
                                  setting  fit s  trees  test MAE  size KB
            5000 trees, no early stopping  13.92   5000     1.659    14068
                 early stopping, n_jobs=1   4.08    833     1.626     2389
                 early stopping, n_jobs=2   3.19    833     1.626     2389
                 early stopping, n_jobs=4   2.91    833     1.626     2389
                 early stopping, n_jobs=8   4.89    833     1.626     2389
                early stopping, n_jobs=18   9.18    833     1.626     2389
                            num_leaves=15   3.28   1568     1.628     2356
                             num_leaves=7   2.95   2072     1.640     1698
                     colsample_bytree=0.5   3.99    967     1.634     2771
33 features (+lags, rolling, day of year)   1.45    130     1.695      399
                          8 features (v1)   2.50    659     1.792     1900

Top 10 of the extended features by gain: hour_cos, temp_now, same_hour_yesterday, hours_ahead, hour_sin, doy_sin, doy_cos, roll_mean_168, roll_mean_24, lag_1
```

   Read it:

   - **Early stopping is the biggest knob**: 833 trees instead of 5,000, 5 times faster, a
     6 times smaller artifact, and a *better* MAE.
   - **More threads are slower beyond 4**: 18 threads take three times as long as 4 on
     365,000 rows. Same model, same MAE. Set `n_jobs` explicitly; never trust the default.
   - Smaller trees (`num_leaves`) need more of them: no time saved here. `colsample_bytree`
     did not help either. Measure each knob; do not assume.
   - **More features made it worse**: 33 features stop after 130 trees with 1.695, against
     1.626 for the 9 of v2. Dropping the one daily feature (v1's 8) costs 0.17 °C. The right
     features matter more than the knobs.

2. Search with a budget: four ways to spend the same 60 seconds (or at most 27 trials) on
   learning rate, `num_leaves` and `min_child_samples`. A trial is scored on 3 time folds of
   30 days in the last training year; the best parameters of each search are then refitted
   and scored once on the test year. Takes about 4 minutes.

```bash
uv run --group efficiency python scripts/efficiency/5b_search.py
```

   You should see (trials and times differ):

```
Budget 60 s or 27 trials per search; CV = 3 time folds x 30 days
                                     search wall s trials pruned  best CV MAE  test MAE                                                          best params
                               grid (3x3x3)     65     16      0        1.535     1.618  {'learning_rate': 0.03, 'num_leaves': 127, 'min_child_samples': 50}
                                     random     62     19      0        1.543     1.626  {'learning_rate': 0.082, 'num_leaves': 123, 'min_child_samples': 7}
                     random + median pruner     44     27     20        1.542     1.618 {'learning_rate': 0.037, 'num_leaves': 119, 'min_child_samples': 41}
                                        TPE     61     19      0        1.549     1.634 {'learning_rate': 0.177, 'num_leaves': 109, 'min_child_samples': 29}
untuned LightGBM, defaults + early stopping                             1.574     1.630                                                                     
                              untuned Ridge                             1.732     1.808                                                                     
```

   Read it:

   - **The pruner is the cheapest win**: it stopped 20 of 27 trials after the first fold, so
     it finished all 27 trials in 44 s while the others ran out of time after 16 to 19.
   - **Tuning bought 0.01 °C on the test year** (1.618 against 1.630 untuned), less than the
     difference between the searches. The ladder (step 3) and the features (step 5.1) moved
     the error ten times more.
   - The best CV MAE is optimistic: it is the minimum over many tries, and early stopping
     watched the same fold. Only the refit on the test year is an honest estimate.

Watch out: with a wall-clock budget, the number of trials depends on your machine; with a
trial budget, the time does. Report both. Fix the seeds of sampler and model, or you cannot
compare two runs.

Think: how much of your project's compute budget would you give to search, and how many trials
is that with your fit time from step 1?

## 6 · How much history is enough?

A learning curve over the length of history: train on the last 1, 3, 6, 12 or 24 months before
the test year, score on the same test year. LightGBM stops early on the last fifth of its
window.

```bash
uv run --group efficiency python scripts/efficiency/6_history.py
```

You should see (times differ):

```
Test year from 2025-10-01, MAE in °C; a month = 30 days
 months       from   rows  Ridge MAE  Ridge ms  LightGBM MAE  LightGBM s  trees
      1 2025-09-01  16980      1.980         2         2.446        1.11    598
      3 2025-07-03  51540      1.881         3         1.981        0.41     88
      6 2025-04-04 103380      1.807         5         1.952        0.41     60
     12 2024-10-06 207060      1.784         9         1.658        1.37    418
     24 2023-10-12 414420      1.811        16         1.669        2.82    762
```

Read it:

- **Ridge needs little data**: 6 months are as good as 24. **LightGBM needs a full year**:
  with less it loses to Ridge, because it has not seen every season and trees cannot
  extrapolate. What matters is *which* data (all seasons), more than *how much*.
- **Two years are not better than one** for either model, at twice the training time. Doubling
  the history is not a free improvement.
- With 3 and 6 months, early stopping watches the last fifth of the window, a different season
  from the test year, and stops after 60 to 88 trees. Your validation window decides what
  "best" means.

Think: how much history does your project use, and have you checked that less would not do?
Where would your curve flatten?

## 7 · Retraining policy

Replay the test year month by month. At the start of each month a policy may retrain on what
is known by then, then predicts the month. Fixed LightGBM settings (400 trees), so the
policies differ only in what they train on. "Warm start" continues the previous model with
`init_model`, adding 50 trees fitted on the last month only.

```bash
uv run --group efficiency python scripts/efficiency/7_retrain.py
```

You should see (times differ):

```
Test year by month; first model: 1.3 s on 2 years
                                       policy  MAE year  MAE last quarter  retrains  retrain s total  final size KB
          never (train once, before the year)     1.614             1.573         0              0.0           1149
           monthly, full refit on all history     1.591             1.504        11             16.7           1158
         monthly, sliding window of 12 months     1.593             1.513        11             11.5           1139
monthly, warm start (+50 trees on last month)     1.973             1.582        11              2.1           2671
     monthly, warm start (+50 trees, lr 0.01)     1.928             1.754        11              2.0           2683
```

Read it:

- **Monthly retraining buys 0.02 °C over the year** and 0.07 °C in the last quarter, the part
  furthest from the first training. Weather does not drift much in a year; your data may.
- **A sliding window of 12 months is as good as all history**, and cheaper (step 6 said so).
- **Warm start is cheapest and worst here.** 50 trees on one month chase that month's errors,
  and the next month is different (October is not November). The model also grows with every
  month. Warm start suits large models with stable data; for a model that refits in seconds,
  refit.
- Compute is not the constraint at this scale: a full year of monthly refits costs 17 s. The
  real cost of retraining is the process around it: the gate, the deploy, the monitoring.

Watch out: this replay uses reanalysis data that exist in full afterwards. A live retrain only
has what arrived by then, sometimes with gaps and late corrections (step 8).

Think: schedule, new data, or a trigger (a monitored error above a threshold)? The monthly MAE
here varies with the season, so what threshold would you use, and how would you avoid
retraining every winter? Write your policy and the reason in your README.

## 8 · Incremental features, storage format, float32

Three parts, about 20 seconds; files go to `data/history/demo/`.

1. **Incremental fetch.** The script pretends the stored table ends a week early, runs
   `history.update()` (fetch from the watermark, the newest stored hour, minus a 48-hour
   overlap; merge; store) and compares with fetching everything again.
2. **Incremental features.** Features for the new week only, plus the 24 hours of lookback
   `build_features()` needs, compared with a full rebuild.
3. **Formats.** The feature table of three years (`make_dataset()`, 630,705 rows) as CSV and
   as Parquet, and in memory as float64 and float32.

```bash
uv run --group efficiency python scripts/efficiency/8_features.py
```

You should see (times differ):

```
Fetch (stored table ended 2026-09-23 23:00:00+00:00)
                                            run  hours fetched  response KB    s
update(): watermark - 48 h to end, merge, store            240            5 0.17
              full refetch: 3 years, fetch only          26304          606 0.15
merged table equals the full history: True
raw responses kept in data/history/demo/raw: 2023-10-01_2026-09-30.json, 2026-09-21_2026-09-30.json

Features
            features for  input hours  median ms (24 horizons)
   full rebuild, 3 years        26304                     22.7
new week + 24 h lookback          192                      7.4
new rows identical to the full rebuild: True

Feature table: 630705 rows x 11 columns
            format MB on disk write s  read s (median of 3)
               CSV       86.3    3.81                 1.520
           Parquet        8.1    0.07                 0.006
Parquet, 2 columns                                    0.003

Same table in memory
  dtype  memory MB  LightGBM test MAE
float64         61              1.626
float32         35              1.623
```

Read it:

- **Incremental fetch moves 120 times fewer bytes** (5 KB vs 606 KB), and the merged table is
  identical to a full refetch. It is *not* faster here: one variable for one place is a single
  quick call either way. It pays with many variables or places, with rate limits (Open-Meteo
  counts large requests as several calls), and with slow sources.
- **The overlap** refetches the newest 48 hours, because archives correct recent days; the
  newer value wins (`history.merge()`). The raw responses are kept, so a parsing bug can be
  fixed by re-parsing, without fetching again.
- **Incremental features need their lookback**: the new rows equal the full rebuild only
  because the 24 hours before them are included. With vectorised features the rebuild costs
  23 ms anyway; with step 2's slow version, or with expensive features, it would not.
- **Parquet**: 10 times smaller than CSV, 50 times faster to write, 250 times faster to read,
  keeps the dtypes and time zones, and reads only the columns you ask for.
- **float32 halves the memory** of the numeric columns (61 to 35 MB) with the same MAE: trees
  bin the features anyway. Keep the raw data in float64; cast the training matrix.

Watch out: the script deletes the 86 MB CSV after measuring it. `data/history/` is git-ignored
and not in the Docker image (`.dockerignore`); delete it with `rm -rf data/history` when you
are done.

Think: what does your feature pipeline recompute on every run, and what is its watermark?
Which data in your project are still CSV?

## What to take to your project

- **Measure before you change anything**, and report a median with its spread, not one run.
  Profile to find *where*, time to know *how much*.
- **Set the thread count explicitly** (`n_jobs`, `OMP_NUM_THREADS`) for small data, in
  training and in the container that serves.
- **Climb a ladder of baselines** and keep the cheap rungs in your report, per horizon or per
  segment. A model that does not beat "same hour yesterday" is not worth its compute.
- **Validate in time order**, with a gap, and keep every learned preprocessing step inside a
  `Pipeline`.
- **Early stopping first, search second**, with a fixed budget, fixed seeds and a pruner; then
  one honest score on data the search never saw.
- **Check how much history you need**, and which history: a learning curve takes minutes.
- **Write down a retraining policy** with its cost and its trigger, and justify it with a
  replay like step 7, not with intuition.
- **Fetch and build features incrementally** with a watermark and an overlap, keep the raw
  responses, store Parquet.

Again: these scripts answer these questions for lucerne-temp. Your model, your data and your
numbers will differ. Ask the same questions, and write your own experiments to answer them.
