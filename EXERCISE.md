# Exercise: deploy, update and roll back lucerne-temp

About 60 minutes. Parts A to C and F are the core; D and E are optional.

| Part | What | Time |
|---|---|---|
| 0 | Prerequisites, a GCP project with billing | before class, 5 min |
| A | Run it locally, with uv and with Docker | 10 min |
| B | Deploy to Cloud Run from your laptop or Cloud Shell | 15 min |
| C | Promote v8, deploy a candidate, split traffic, roll back | 20 min |
| D | Optional: deploy from GitHub Actions without keys | 15 min |
| E | Optional: batch job and UI | 5 min |
| F | Clean up | 5 min |

> **Status of this document.** Part A and the local commands of part E were executed while
> building this repo. Parts B and C were executed on 17.09.2026 against Cloud Run in
> `europe-west6`, and the notes in those parts describe what was seen. Parts D and F, and
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
docker build --platform linux/amd64 -t $IMAGE:v7 .
docker push $IMAGE:v7
```

   No Docker? Let Cloud Build do both steps:
   `gcloud services enable cloudbuild.googleapis.com && gcloud builds submit --tag $IMAGE:v7`

4. Deploy. `--allow-unauthenticated` makes the URL public, which is what "live" means for MS4.

```bash
gcloud run deploy lucerne-temp --image $IMAGE:v7 --region $REGION --allow-unauthenticated
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

## C · Promote v8, canary, roll back

1. Run the promotion gate. It scores v8 and v7 on the same last 7 days and moves the alias
   only if v8 is better. Look at what changed.

```bash
uv run python scripts/promote.py 8
git diff registry/registry.json
uv run python scripts/champion_version.py      # 8
```

2. Commit the promotion on a branch (in part D a push to main deploys by itself).

```bash
git switch -c promote-v8
git commit -am "Promote temp-forecaster v8 to champion"
```

3. Build the new image and deploy it as a second revision **without traffic**, tagged `candidate`.

```bash
docker build --platform linux/amd64 -t $IMAGE:v8 .
docker push $IMAGE:v8
gcloud run deploy lucerne-temp --image $IMAGE:v8 --region $REGION --no-traffic --tag candidate
```

4. Smoke test the candidate on its own URL. Users still get v7. The tag URL is the service URL
   from `services describe` (part B5) with `candidate---` in front. The deploy output in step 3
   prints it too: "The revision can be reached directly at https://candidate---…".

```bash
export CANDIDATE_URL="https://candidate---${URL#https://}"
curl -s $CANDIDATE_URL/health | jq .model_version      # "8"
curl -s $URL/health | jq .model_version                # "7"
```

5. Canary: 10 % of the traffic to the candidate. Count which version answers.

```bash
gcloud run services update-traffic lucerne-temp --region $REGION --to-tags candidate=10
sleep 10                                               # let the new split arrive
for i in $(seq 40); do curl -s $URL/health | jq -r .model_version; done | sort | uniq -c
```

Expect about 36 × `7` and 4 × `8`; the split is random, so 2 or 6 is fine too. Counting right
after the traffic update can show 40 × `7` even at 10 %: the new split takes a few seconds to
reach every front end. Wait and count again.

6. All good: everything to the newest revision.

```bash
gcloud run services update-traffic lucerne-temp --region $REGION --to-latest
curl -s $URL/health | jq .model_version                # "8"
```

   The output shows `candidate (Deleting)` next to the tag URL. Ignore it: the tag is still
   there afterwards, as `gcloud run services describe lucerne-temp --region $REGION` shows.

7. Roll back. List the revisions, then send all traffic to the previous one. No rebuild.

```bash
gcloud run revisions list --service lucerne-temp --region $REGION
gcloud run services update-traffic lucerne-temp --region $REGION \
  --to-revisions <name-of-the-v7-revision>=100
curl -s $URL/health | jq .model_version                # "7"
```

   Traffic is now pinned to that revision: a later `gcloud run deploy` will not receive traffic
   until you run `update-traffic --to-latest` again.

8. Think: step 7 rolled back the *service*. The alias `champion` still says 8. When is that
   the right rollback, and when would you run `uv run python scripts/promote.py 7 --force`,
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
git switch main && git merge promote-v8 && git push
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
Both show the model version. After part C they can disagree (file from v8, service rolled back
to v7): that is the "both paths must report the same version" check from the lecture.
`.github/workflows/predict.yml` runs the same batch job every hour on GitHub's runner.

## F · Clean up

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
```

Prices: [Cloud Run](https://cloud.google.com/run/pricing) ·
[Artifact Registry](https://cloud.google.com/artifact-registry/pricing)

## What to take to your project

This repository is an inspiration, not a template: it shows one way to do each step, and your
MS4 service should not be a copy of it. Your project has its own data, model and decisions.
The ideas worth carrying over: a `/health` that reports the model version, responses that say
which model and which data produced them, a prediction log, a smoke test before traffic moves,
a promotion gate, and two rollback commands in your README. Build them your way, and explain
your choices in the README.
