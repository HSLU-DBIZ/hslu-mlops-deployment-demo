# Live demo run sheet · SW06 Deployment and Serving

Lecturer-facing. Students get [EXERCISE.md](EXERCISE.md); this is the same story told in
8 minutes from a terminal and a browser. Every command here also exists as a step of
`demo/demo.sh`, so you can drive the demo either way.

Timings are from the run on 17.09.2026 (Cloud Run, `europe-west6`).

## Before class (about 10 minutes, plus a 4-minute build)

Terminal: large font, short prompt, `clear`. Two tabs help: one for commands, one for `watch`-style checks.

```bash
gcloud config configurations list        # is the right configuration ACTIVE?
gcloud config configurations activate hslu-mlops
export PROJECT_ID=<your-project-id>      # never hard-coded in this repo
export REGION=europe-west6
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/ml/lucerne-temp
command -v jq                            # the demo pipes through jq
```

Build and push both images, deploy v1, warm it. `demo/demo.sh prepare` does exactly this.

```bash
docker build --platform linux/amd64 -t $IMAGE:v1 .     # ~1.5 min
docker push $IMAGE:v1                                  # ~30 s
uv run python scripts/promote.py 2                     # champion -> 2, so the image bakes v2
docker build --platform linux/amd64 -t $IMAGE:v2 .
docker push $IMAGE:v2
git checkout registry/registry.json                    # back to champion = 1 for the live gate
gcloud run deploy lucerne-temp --image $IMAGE:v1 --region $REGION --allow-unauthenticated
export URL=$(gcloud run services describe lucerne-temp --region $REGION --format 'value(status.url)')
curl -s $URL/health | jq                               # warm the container: no cold start on stage
```

First deploy takes about 2 minutes; deploying an already-pushed image later takes about 40 s.
Leave a browser tab open on **Cloud Run → lucerne-temp → Revisions**, logged in, and one on
**Logs**.

Last check before you start:

```bash
curl -s $URL/health | jq          # "model_version": "1", "feature_source": "live"
uv run python scripts/champion_version.py    # 1
```

## Live (about 8 minutes)

### 1 · The contract (1 s)

```bash
curl -s $URL/health | jq
curl -s -X POST $URL/predict -H 'Content-Type: application/json' -d '{"hours_ahead": 6}' | jq
```

> "Every answer carries the model version and the time the features are from."

```bash
curl -s -o /dev/null -w 'HTTP %{http_code}\n' -X POST $URL/predict \
  -H 'Content-Type: application/json' -d '{"hours_ahead": 48}'      # HTTP 422
```

> "Bad input is a 422 from pydantic, not a crash inside the model."

### 2 · Every prediction leaves a record

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="lucerne-temp" AND jsonPayload.event="prediction"' \
  --limit 5 --freshness 1h --format 'value(jsonPayload)'
```

> "One JSON line on stdout became a structured log entry: inputs, output, version, latency."

`gcloud run services logs read` does **not** show these lines. The Console page (Logs tab)
shows them expandable, which reads better on a projector.

### 3 · The gate decides, not the person (instant, local)

```bash
uv run python scripts/promote.py 2
```

```
last 7 days: candidate v2 MAE 1.925 °C, champion v1 MAE 2.159 °C
promoted: champion -> v2. Commit registry/registry.json and deploy.
```

> "Same window, same metric. Worse candidate, no promotion, and nothing gets deployed."

This writes `registry/registry.json`. Reset it with `git checkout registry/registry.json`,
or move the alias back with `uv run python scripts/promote.py 1 --force`.

### 4 · A revision nobody sees yet (about 40 s)

```bash
gcloud run deploy lucerne-temp --image $IMAGE:v2 --region $REGION --no-traffic --tag candidate
```

While it runs: a *revision* is one immutable version of the service, a *service* is the URL plus
the rule for splitting traffic across revisions. `--no-traffic` creates the revision and gives
nobody to it; `--tag candidate` gives it a private URL. The output ends with
"The revision can be reached directly at https://candidate---…".

### 5 · Two URLs, two answers

```bash
export CANDIDATE_URL="https://candidate---${URL#https://}"
curl -s $CANDIDATE_URL/health | jq -r .model_version     # 2
curl -s $URL/health | jq -r .model_version               # 1
```

> "The new model is deployed and reachable, and no user has touched it. This is the smoke test
> the deploy workflow runs."

### 6 · 10 % (the moment that lands)

```bash
gcloud run services update-traffic lucerne-temp --region $REGION --to-tags candidate=10
for i in $(seq 40); do curl -s $URL/health | jq -r .model_version; done | sort | uniq -c
```

Wait about 10 s after the traffic update before counting (the script does); the new split
needs a moment to reach every front end. On 17.09.2026 that gave 39 answers from v1 and 1 from v2. It is a random split, so expect roughly 4 in 40,
sometimes 0 or 1. Say the expected number out loud *before* running it.

### 7 · Rollback

```bash
gcloud run revisions list --service lucerne-temp --region $REGION
gcloud run services update-traffic lucerne-temp --region $REGION \
  --to-revisions <name-of-the-v1-revision>=100
curl -s $URL/health | jq -r .model_version               # 1
```

> "No rebuild, no redeploy. The old revision never went away."

### 8 · The picture

Browser: **Cloud Run → lucerne-temp → Revisions**. Point at the traffic column, the tag on the
candidate, and the revision list as the deployment history.

## If something fails

| Problem | Do this |
|---|---|
| No network, or GCP unreachable | Play the screencast (record one the week before: 3 min, steps 1–7). Then the deploy-pipeline animation in Part 3 of the deck tells the same story without a cloud. |
| Docker build fails on stage | Both images are already pushed. Skip to step 4; nothing in steps 1–7 rebuilds. |
| Gate says "not promoted" / champion is already 2 | `git checkout registry/registry.json`, then run step 3 again. |
| Wrong project or nothing found | `gcloud config configurations list` — is `hslu-mlops` ACTIVE? Then re-`export` the variables. |
| `$URL` empty or 404 | `export URL=$(gcloud run services describe lucerne-temp --region $REGION --format 'value(status.url)')` |
| Step 6 shows 40 × "1" | Fine, it is random. Run the loop once more, or raise the split to 50. |
| `--to-latest` prints "candidate (Deleting)" | Ignore it, the tag survives; `gcloud run services describe lucerne-temp --region $REGION` shows it. |
| First request is slow | Cold start. Say so: it is the trade-off of scale-to-zero, and it is on the Operate slides. |

## After

The service scales to zero, so leaving it running costs nothing and students can curl it during
the exercise. Clean-up commands are in [EXERCISE.md](EXERCISE.md), walkthrough 1, step 7.

If you promoted in step 3 and want the repo back to its committed state:

```bash
git checkout registry/registry.json
```

## Driving it from the script

```bash
./demo/demo.sh --dry-run all     # rehearse: prints every command, runs nothing
./demo/demo.sh prepare           # before class
./demo/demo.sh health            # then: predict, logs, gate, candidate, canary, rollback, status
```

Each step echoes the command before running it, so the audience sees what you typed.
`status` answers "where are we now?" at any point. `rollback` finds the previous revision
itself: the untagged revision that still serves traffic next to the candidate, so run it
after `canary`, not after `--to-latest`. If that lookup comes back empty, read the name from the list it prints and re-run with
`PREVIOUS_REVISION=<name> ./demo/demo.sh rollback`.
