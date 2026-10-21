# Exercise: deploy, update and roll back lucerne-temp

About 60 minutes. Parts A to C and F are the core; D and E are optional. Part G is an optional
home exercise of about 45 minutes, mostly waiting.

| Part | What | Time |
|---|---|---|
| 0 | Prerequisites, a GCP project with billing | before class, 5 min |
| A | Run it locally, with uv and with Docker | 10 min |
| B | Deploy to Cloud Run from your laptop or Cloud Shell | 15 min |
| C | Promote v2, deploy a candidate, split traffic, roll back | 20 min |
| D | Optional: deploy from GitHub Actions without keys | 15 min |
| E | Optional: batch job and UI | 5 min |
| F | Clean up | 5 min |
| G | Optional, at home: the same model on Agent Platform | 45 min |

> **Status of this document.** Part A and the local commands of part E were executed while
> building this repo. Parts B and C were executed on 17.09.2026 against Cloud Run in
> `europe-west6`, and the notes in those parts describe what was seen. Part G was executed on
> 21.10.2026 against Agent Platform in `europe-west6`, with the timings it quotes. Parts D and F, and
> writing predictions to `gs://` (part E), were checked against the current gcloud and GitHub
> Actions documentation but **not executed**. If a command fails for you, read the error,
> check `gcloud <command> --help`, and tell us.

Part 0 uses the gcloud commands of last semester's setup notebook; the budget command in
step 0.4 was checked against the documentation but not executed.

All commands assume a bash or zsh shell in the repository root. On Windows use WSL or Cloud Shell.

## 0 · Prerequisites and a GCP project

Tools:

1. Google Cloud education coupon redeemed (course slides, "GCP credits"). This gives you a
   billing account with the credit on it.
2. [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed.
   Or use [Cloud Shell](https://shell.cloud.google.com), which has gcloud, Docker, git and jq.
3. Docker Desktop running (not needed in Cloud Shell).
4. [uv](https://docs.astral.sh/uv/getting-started/installation/) and `jq`
   (`brew install jq`, `sudo apt install jq`).
5. Fork this repository on GitHub and clone **your fork**:

```bash
git clone https://github.com/<your-user>/hslu-mlops-deployment-demo.git
cd hslu-mlops-deployment-demo
```

A project with billing, once (about 5 minutes). Already have one from another course? Skip
to step 4 and use its ID.

1. Log in.

```bash
gcloud auth login
```

2. Create a project and make it the default. Project IDs are global across all of Google
   Cloud: `mlops-hslu-` plus your initials and a number usually works. If `create` fails,
   the ID is taken; pick another.

```bash
export PROJECT_ID=mlops-hslu-<your-id>
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
```

3. Link the billing account that came with your coupon. Without it, enabling any API in
   part B fails.

```bash
gcloud billing accounts list                     # copy the ACCOUNT_ID
gcloud billing projects link $PROJECT_ID --billing-account <ACCOUNT_ID>
gcloud billing projects describe $PROJECT_ID     # billingEnabled: true
```

4. Set a budget alert, so a forgotten resource cannot eat the credit unnoticed. Easiest in
   the Console: Billing → Budgets & alerts → Create budget, scope this project, amount
   USD 50, alerts at 50 % and 90 %. Or with gcloud (use your billing account's currency):

```bash
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create --billing-account <ACCOUNT_ID> \
  --display-name mlops-hslu --budget-amount 50USD \
  --filter-projects projects/$PROJECT_ID \
  --threshold-rule percent=0.5 --threshold-rule percent=0.9
```

   A budget only sends alerts; it does not stop spending. Part F still matters.

## A · Run it locally

1. Install and test. Everything runs offline, from the committed snapshot.

```bash
uv sync
uv run pytest -q
```

2. Train the champion and look at the numbers. Which horizon does the model *lose* against
   the persistence baseline?

```bash
uv run python -m lucerne_temp.training.train
```

3. Start the API, then call it from a second terminal.

```bash
uv run uvicorn lucerne_temp.inference.api:app --port 8080
```

```bash
curl -s localhost:8080/health | jq
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"hours_ahead": 3}' | jq
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"hours_ahead": 48}' | jq          # 422: the contract rejects it
```

   Look at the first terminal: one JSON line per prediction. Stop the API with Ctrl+C.

4. Break the data source and watch the fallback. What do `feature_source` and `feature_time` say now?

```bash
OPEN_METEO_URL=http://localhost:9 uv run uvicorn lucerne_temp.inference.api:app --port 8080
```

5. The same service as a container. The build trains the champion: there is no model file in git.

```bash
docker build -t lucerne-temp .
docker run --rm -p 8080:8080 lucerne-temp
```

```bash
curl -s localhost:8080/health | jq
```

   Port 8080 taken? Use `-p 8081:8080` and call `localhost:8081`.

## B · Deploy to Cloud Run

1. Set the project and the region, enable the two APIs. Use **your own** project ID: every
   command below reads `$PROJECT_ID` and `$REGION` (`europe-west6`, Zurich), and no project is
   hard-coded anywhere in this repo. In a new terminal, run the `export` lines again.

```bash
export PROJECT_ID=<your-project-id>
export REGION=europe-west6
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

   Planning to do part G? It needs one more API: `gcloud services enable aiplatform.googleapis.com`.

2. Create a Docker repository in Artifact Registry and let Docker push to it.

```bash
gcloud artifacts repositories create ml \
  --repository-format docker --location $REGION --description "MLOps exercise"
gcloud auth configure-docker $REGION-docker.pkg.dev
```

3. Build and push. Cloud Run runs `linux/amd64`; on an Apple Silicon Mac the `--platform`
   flag is required (the build then takes about two minutes).

```bash
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/ml/lucerne-temp
docker build --platform linux/amd64 -t $IMAGE:v1 .
docker push $IMAGE:v1
```

   No Docker? Let Cloud Build do both steps:
   `gcloud services enable cloudbuild.googleapis.com && gcloud builds submit --tag $IMAGE:v1`

4. Deploy. `--allow-unauthenticated` makes the URL public, which is what "live" means for MS4.

```bash
gcloud run deploy lucerne-temp --image $IMAGE:v1 --region $REGION --allow-unauthenticated
```

5. Read the model version from the running service.

```bash
export URL=$(gcloud run services describe lucerne-temp --region $REGION --format 'value(status.url)')
curl -s $URL/health | jq
curl -s -X POST $URL/predict -H 'Content-Type: application/json' -d '{"hours_ahead": 6}' | jq
```

   The URL looks different from the one `gcloud run deploy` printed
   (`https://lucerne-temp-<project-number>.$REGION.run.app`): a service has both addresses and
   both work. This exercise uses the one from `services describe` throughout.

6. Find your prediction in the logs: Console → Cloud Run → lucerne-temp → Logs, or the command
   below. Cloud Run parsed the JSON line from stdout, so the fields arrive as `jsonPayload`
   and you can filter on them. (`gcloud run services logs read` does not show these lines.)

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="lucerne-temp" AND jsonPayload.event="prediction"' \
  --limit 5 --freshness 1h --format 'value(jsonPayload)'
```

## C · Promote v2, canary, roll back

1. Run the promotion gate. It scores v2 and v1 on the same last 7 days and moves the alias
   only if v2 is better. Look at what changed.

```bash
uv run python scripts/promote.py 2
git diff registry/registry.json
uv run python scripts/champion_version.py      # 2
```

2. Commit the promotion on a branch (in part D a push to main deploys by itself).

```bash
git switch -c promote-v2
git commit -am "Promote temp-forecaster v2 to champion"
```

3. Build the new image and deploy it as a second revision **without traffic**, tagged `candidate`.

```bash
docker build --platform linux/amd64 -t $IMAGE:v2 .
docker push $IMAGE:v2
gcloud run deploy lucerne-temp --image $IMAGE:v2 --region $REGION --no-traffic --tag candidate
```

4. Smoke test the candidate on its own URL. Users still get v1. The tag URL is the service URL
   from `services describe` (part B5) with `candidate---` in front. The deploy output in step 3
   prints it too: "The revision can be reached directly at https://candidate---…".

```bash
export CANDIDATE_URL="https://candidate---${URL#https://}"
curl -s $CANDIDATE_URL/health | jq .model_version      # "2"
curl -s $URL/health | jq .model_version                # "1"
```

5. Canary: 10 % of the traffic to the candidate. Count which version answers.

```bash
gcloud run services update-traffic lucerne-temp --region $REGION --to-tags candidate=10
sleep 10                                               # let the new split arrive
for i in $(seq 40); do curl -s $URL/health | jq -r .model_version; done | sort | uniq -c
```

Expect about 36 answers `1` and 4 answers `2`; the split is random, so anywhere from 2 to 6
answers from v2 is fine too. Counting right
after the traffic update can show 40 × `1` even at 10 %: the new split takes a few seconds to
reach every front end. Wait and count again.

6. All good: everything to the newest revision.

```bash
gcloud run services update-traffic lucerne-temp --region $REGION --to-latest
curl -s $URL/health | jq .model_version                # "2"
```

   The output shows `candidate (Deleting)` next to the tag URL. Ignore it: the tag is still
   there afterwards, as `gcloud run services describe lucerne-temp --region $REGION` shows.

7. Roll back. List the revisions, then send all traffic to the previous one. No rebuild.

```bash
gcloud run revisions list --service lucerne-temp --region $REGION
gcloud run services update-traffic lucerne-temp --region $REGION \
  --to-revisions <name-of-the-v1-revision>=100
curl -s $URL/health | jq .model_version                # "1"
```

   Traffic is now pinned to that revision: a later `gcloud run deploy` will not receive traffic
   until you run `update-traffic --to-latest` again.

8. Think: step 7 rolled back the *service*. The alias `champion` still says 2. When is that
   the right rollback, and when would you run `uv run python scripts/promote.py 1 --force`,
   commit and redeploy instead?

## D · Optional: deploy from GitHub Actions, without keys

`.github/workflows/deploy.yml` does parts B and C on every push to main: tests, build, push,
deploy with no traffic, smoke test, shift traffic. It authenticates with Workload Identity
Federation: GitHub proves which repository the job runs for, Google hands out a short-lived
token. No key file exists. The service from part B must exist already, because gcloud
rejects `--no-traffic` when it would create a new service.

1. Create the deploy service account and give it what it needs: deploy to Cloud Run, push
   images, and act as the service's runtime account.

```bash
export REPO=<your-user>/hslu-mlops-deployment-demo
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format 'value(projectNumber)')
export DEPLOY_SA=deployer@$PROJECT_ID.iam.gserviceaccount.com

gcloud services enable iamcredentials.googleapis.com
gcloud iam service-accounts create deployer --display-name "GitHub Actions deployer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:$DEPLOY_SA --role roles/run.admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:$DEPLOY_SA --role roles/artifactregistry.writer
gcloud iam service-accounts add-iam-policy-binding \
  $PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --member serviceAccount:$DEPLOY_SA --role roles/iam.serviceAccountUser
```

2. Create the pool and the GitHub provider. The attribute condition is the important line:
   only tokens from *your* repository are accepted.

```bash
gcloud iam workload-identity-pools create github \
  --location global --display-name "GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc lucerne-temp \
  --location global --workload-identity-pool github \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository == '$REPO'"

gcloud iam service-accounts add-iam-policy-binding $DEPLOY_SA \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"
```

3. Store three repository **variables** (not secrets: none of the values is secret).
   GitHub → Settings → Secrets and variables → Actions → Variables, or with the `gh` CLI:

```bash
gh variable set GCP_PROJECT_ID --repo $REPO --body "$PROJECT_ID"
gh variable set DEPLOY_SA      --repo $REPO --body "$DEPLOY_SA"
gh variable set WIF_PROVIDER   --repo $REPO --body \
  "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/lucerne-temp"
```

4. In your fork, enable workflows in the Actions tab (forks start with Actions disabled).
   Then merge and push the promotion from part C. IAM changes can take a few minutes to arrive.

```bash
git switch main && git merge promote-v2 && git push
gh run watch --repo $REPO
```

5. Open the run. Step 5 compares `/health` on the candidate URL with
   `scripts/champion_version.py`. Make it fail once on purpose: what would you change, and
   which revision keeps serving while the job is red?

## E · Optional: batch job and UI

```bash
uv run python -m lucerne_temp.inference.batch
head -3 predictions/latest.csv
API_URL=$URL uv run --group ui streamlit run ui/app.py
```

The page reads `predictions/latest.csv` for the chart and calls `$URL/predict` for the button.
Both show the model version. After part C they can disagree (file from v2, service rolled back
to v1): that is the "both paths must report the same version" check from the lecture.
`.github/workflows/predict.yml` runs the same batch job every hour on GitHub's runner.

## F · Clean up

Did part G? Run its last step first: an Agent Platform endpoint bills every hour it exists
with a model deployed.

Cloud Run scales to zero and bills per request, so an idle service costs close to nothing;
stored images are billed per GB and month beyond a small free allowance. The exercise should
cost a few cents at most. Delete everything anyway, the credits are for your project:

```bash
gcloud run services delete lucerne-temp --region $REGION --quiet
gcloud artifacts repositories delete ml --location $REGION --quiet     # deletes all images in it
```

If you did part D:

```bash
gcloud iam workload-identity-pools delete github --location global --quiet
gcloud iam service-accounts delete $DEPLOY_SA --quiet
gh variable delete WIF_PROVIDER --repo $REPO      # deploy.yml is skipped again
```

A deleted pool is kept for 30 days and blocks its name during that time. For your project,
pick another pool name or restore it with `gcloud iam workload-identity-pools undelete github --location global`.

Check that nothing is left:

```bash
gcloud run services list
gcloud artifacts repositories list
gcloud ai endpoints list --region $REGION      # part G: no lucerne-temp-ep
gcloud ai models list --region $REGION         # part G: Listed 0 items.
```

Prices: [Cloud Run](https://cloud.google.com/run/pricing) ·
[Artifact Registry](https://cloud.google.com/artifact-registry/pricing)

## G · Optional: the same model on Agent Platform

Cloud Run runs any container. Agent Platform (that is what the Google Cloud console calls it;
formerly **Vertex AI**, and the API `aiplatform.googleapis.com`, the `gcloud ai` commands and
the Python SDK still use the old names) is a managed model-serving platform: you register the
container as a *model* with *versions*, deploy versions to an *endpoint*, and split traffic
between deployed models. Same ideas as part C, different nouns.

| Cloud Run (part C) | Agent Platform (part G) |
|---|---|
| image `:v1`, `:v2` | model `lucerne-temp`, versions 1 and 2 |
| service URL | endpoint `lucerne-temp-ep` |
| revision | deployed model (its own ID) |
| `update-traffic --to-tags candidate=10` | `deploy-model ... --traffic-split <v1-id>=90,0=10` |
| `update-traffic --to-revisions <v1>=100` | `endpoints update --traffic-split <v1-id>=100` |

The container contract is a little different. Agent Platform posts
`{"instances": [{"hours_ahead": 6}]}` to the predict route and expects
`{"predictions": [...]}` back, one prediction per instance, in the same order. That is
`POST /v1/predict` in `src/lucerne_temp/inference/api.py`: a thin adapter around `/predict`.
The health route must answer 200, and the container listens on port 8080 like on Cloud Run.

> **Cost.** Cloud Run scales to zero; an Agent Platform endpoint does not. Every deployed
> model keeps at least one `n1-standard-2` node running (`--min-replica-count 1`) and bills
> node-hours **even when nobody calls it**: in the order of USD 0.10 to 0.20 per node-hour
> (check the pricing page for your region), twice that while two versions are deployed. A
> forgotten endpoint costs a few dollars a day. Do the whole part in one sitting and run
> step 9 at the end. Our test run (about 25 minutes of endpoint time) cost well under a dollar. [Pricing](https://cloud.google.com/vertex-ai/pricing#prediction-prices)

You need the images `$IMAGE:v1` and `$IMAGE:v2` from parts B and C. If you already ran part F,
recreate the repository and push both again (B2, B3, C3).

1. Enable the API. Keep `$PROJECT_ID`, `$REGION` and `$IMAGE` from part B.

```bash
gcloud services enable aiplatform.googleapis.com
```

2. Upload v1 as a model. "Upload" registers the image and its routes; nothing runs yet. It
   takes about two minutes and prints no ID, so read it back with `models list`.

```bash
gcloud ai models upload --region $REGION --display-name lucerne-temp \
  --container-image-uri $IMAGE:v1 --container-ports 8080 \
  --container-health-route /health --container-predict-route /v1/predict
export MODEL_ID=$(gcloud ai models list --region $REGION \
  --filter display_name=lucerne-temp --format 'value(name)')
echo $MODEL_ID                                  # one long number
```

   Two numbers? You uploaded twice. Delete the extra one (`gcloud ai models delete <id> --region $REGION`).

3. Create an endpoint. It is only an address and costs nothing until a model is deployed.

```bash
gcloud ai endpoints create --region $REGION --display-name lucerne-temp-ep
export ENDPOINT_ID=$(gcloud ai endpoints list --region $REGION \
  --filter display_name=lucerne-temp-ep --format 'value(name)')
```

4. Deploy v1 to the endpoint with all traffic. **This took 15 minutes** in our test: Agent
   Platform starts a VM, pulls the image and waits for `/health`. Billing starts now.

```bash
gcloud ai endpoints deploy-model $ENDPOINT_ID --region $REGION --model $MODEL_ID \
  --display-name v1 --machine-type n1-standard-2 \
  --min-replica-count 1 --max-replica-count 3 --traffic-split 0=100
```

   You should see `Deployed a model to the endpoint ... Id of the deployed model: <number>`.
   That ID is not the model ID: it names this *deployment* of v1 on this endpoint, like a
   revision name. `0` in `--traffic-split` means "the model being deployed right now".

5. Predict. The request body is the Agent Platform format, not the one of `/predict`.

```bash
echo '{"instances": [{"hours_ahead": 6}]}' > instances.json
gcloud ai endpoints predict $ENDPOINT_ID --region $REGION \
  --json-request instances.json --format json
```

   You should see (numbers differ):

```json
{
  "deployedModelId": "5307406398949163008",
  "model": "projects/<project-number>/locations/europe-west6/models/<model-id>",
  "modelDisplayName": "lucerne-temp",
  "modelVersionId": "1",
  "predictions": [
    {"feature_source": "live", "feature_time": "2026-10-21T14:00:00Z", "model_version": "1",
     "temperature_c": 9.78, "valid_time": "2026-10-21T20:00:00Z"}
  ]
}
```

   Without `--format json`, gcloud prints only the predictions, as a Python list rather than
   JSON, which `jq` cannot read. Two version fields now: `modelVersionId` is the platform's
   registry, `model_version` is ours. They agree only because we uploaded v1 as version 1.

6. Upload v2 as **version 2 of the same model**, then deploy it next to v1 with 10 % of the
   traffic. `--parent-model` needs the full resource name; the bare ID fails with
   "Location ID is not provided". The new version is `$MODEL_ID@2`. This deploy took about
   7 minutes; v1 keeps serving all traffic until it is done.

```bash
gcloud ai models upload --region $REGION --display-name lucerne-temp \
  --parent-model projects/$PROJECT_ID/locations/$REGION/models/$MODEL_ID \
  --container-image-uri $IMAGE:v2 --container-ports 8080 \
  --container-health-route /health --container-predict-route /v1/predict
gcloud ai models list-version $MODEL_ID --region $REGION    # versions 1 and 2

export V1_DEPLOYED_ID=$(gcloud ai endpoints describe $ENDPOINT_ID --region $REGION \
  --format json | jq -r '.deployedModels[] | select(.displayName=="v1") | .id')
gcloud ai endpoints deploy-model $ENDPOINT_ID --region $REGION --model $MODEL_ID@2 \
  --display-name v2 --machine-type n1-standard-2 \
  --min-replica-count 1 --max-replica-count 3 --traffic-split $V1_DEPLOYED_ID=90,0=10
```

7. Count which version answers. Each call takes about a second.

```bash
gcloud ai endpoints describe $ENDPOINT_ID --region $REGION --format 'yaml(trafficSplit)'
for i in $(seq 40); do
  gcloud ai endpoints predict $ENDPOINT_ID --region $REGION --json-request instances.json \
    --format json 2>/dev/null | jq -r '.predictions[0].model_version'
done | sort | uniq -c
```

   You should see two deployed-model IDs at 90 and 10, and about 36 × `1` and 4 × `2`. We got
   38 and 2; anything from 1 to 7 answers from v2 is normal for 40 random draws.

8. Roll back: all traffic to v1, then undeploy v2 so its node stops billing. No rebuild, no
   new upload: the version stays in the registry.

```bash
export V2_DEPLOYED_ID=$(gcloud ai endpoints describe $ENDPOINT_ID --region $REGION \
  --format json | jq -r '.deployedModels[] | select(.displayName=="v2") | .id')
gcloud ai endpoints update $ENDPOINT_ID --region $REGION --traffic-split $V1_DEPLOYED_ID=100
gcloud ai endpoints undeploy-model $ENDPOINT_ID --region $REGION --deployed-model-id $V2_DEPLOYED_ID
gcloud ai endpoints describe $ENDPOINT_ID --region $REGION --format 'yaml(trafficSplit)'
```

   As on Cloud Run, the new split takes a few seconds to arrive: right after the update we
   still saw 2 of 20 answers from v2. Move the traffic before you undeploy, so no request is
   sent to a model that is going away.

   The prediction log lines arrive in Cloud Logging too, but as a string in
   `jsonPayload.message`, not as separate fields like on Cloud Run:

```bash
gcloud logging read "resource.type=\"aiplatform.googleapis.com/Endpoint\" AND resource.labels.endpoint_id=\"$ENDPOINT_ID\" AND jsonPayload.message:\"prediction\"" \
  --limit 3 --freshness 1h --format 'value(jsonPayload.message)'
```

9. Clean up, **now**. Undeploy v1, delete the endpoint and the model (all versions). Check
   that `lucerne-temp-ep` is gone and no model is left.

```bash
gcloud ai endpoints undeploy-model $ENDPOINT_ID --region $REGION --deployed-model-id $V1_DEPLOYED_ID
gcloud ai endpoints delete $ENDPOINT_ID --region $REGION --quiet
gcloud ai models delete $MODEL_ID --region $REGION --quiet
gcloud ai endpoints list --region $REGION
gcloud ai models list --region $REGION          # Listed 0 items.
```

   The images stay in Artifact Registry; part F deletes them.

10. Think: Cloud Run or Agent Platform for your MS4 service? Compare what you just did with
    part C: time to deploy, cost when idle, the request format, what the platform knows about
    your model. There is no right answer; write your choice and the reason in your README.

### Optional: batch prediction

The same model can score a file without an endpoint: Agent Platform starts nodes, sends the
file through `/v1/predict` in chunks, writes the results to Cloud Storage and stops. In our
test, 5 rows took 11 minutes, almost all of it start-up. Run it before step 9 (it needs the
model), from Python, with your user credentials:

```bash
gcloud auth application-default login
export BUCKET=gs://$PROJECT_ID-lucerne-batch
gcloud storage buckets create $BUCKET --location $REGION
for h in 1 3 6 12 24; do echo "{\"hours_ahead\": $h}"; done > batch.jsonl
gcloud storage cp batch.jsonl $BUCKET/input/batch.jsonl

uv run --no-project --with google-cloud-aiplatform python - <<EOF
from google.cloud import aiplatform

aiplatform.init(project="$PROJECT_ID", location="$REGION")
job = aiplatform.Model("$MODEL_ID").batch_predict(
    job_display_name="lucerne-temp-batch",
    gcs_source="$BUCKET/input/batch.jsonl",
    gcs_destination_prefix="$BUCKET/output",
    instances_format="jsonl",
    machine_type="n1-standard-2",
    starting_replica_count=1,
    max_replica_count=1,
)  # waits until the job is done
print(job.state, job.output_info.gcs_output_directory)
EOF

gcloud storage cat "$BUCKET/output/*/prediction.results-*"
```

The script prints the job state (`JOB_STATE_SUCCEEDED`) and the output folder; the results
file holds one line per input row, e.g.
`{"instance": {"hours_ahead": 1}, "prediction": {"valid_time": "...", "temperature_c": 10.34, "model_version": "1", ...}}`.
Then delete the bucket: `gcloud storage rm -r $BUCKET`.

## What to take to your project

This repository is an inspiration, not a template: it shows one way to do each step, and your
MS4 service should not be a copy of it. Your project has its own data, model and decisions.
The ideas worth carrying over: a `/health` that reports the model version, responses that say
which model and which data produced them, a prediction log, a smoke test before traffic moves,
a promotion gate, and two rollback commands in your README. Build them your way, and explain
your choices in the README.
