#!/usr/bin/env bash
# Lecture demo driver for lucerne-temp. One function per live step; see DEMO.md.
#
#   ./demo/demo.sh health|predict|logs|gate|candidate|canary|rollback|status
#   ./demo/demo.sh prepare          # build, push, deploy v1 (before class)
#   ./demo/demo.sh all              # rehearsal: every live step in order
#   ./demo/demo.sh --dry-run all    # print the commands, touch nothing
#
# Environment: PROJECT_ID (required for prepare/candidate), REGION, SERVICE, IMAGE, URL.
# Everything it can derive, it derives. It never deletes anything.

set -euo pipefail

REGION="${REGION:-europe-west6}"
SERVICE="${SERVICE:-lucerne-temp}"
DRY_RUN=0

BOLD=""
DIM=""
RESET=""
if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"
    DIM="$(printf '\033[2m')"
    RESET="$(printf '\033[0m')"
fi

say() { printf '\n%s# %s%s\n' "$DIM" "$*" "$RESET"; }
die() { printf '\n%s! %s%s\n' "$BOLD" "$*" "$RESET" >&2; exit 1; }

# Echo the command so the audience sees it, then run it (unless --dry-run).
run() {
    printf '\n%s$ %s%s\n' "$BOLD" "$1" "$RESET"
    [ "$DRY_RUN" -eq 1 ] && return 0
    eval "$1"
}

# Same, but capture the output instead of showing it; used to derive URLs and names.
# The banner goes to stderr so it stays out of the captured value.
capture() {
    printf '\n%s$ %s%s\n' "$BOLD" "$1" "$RESET" >&2
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '%s' "${2:-DRY_RUN}"
        return 0
    fi
    eval "$1"
}

need_project() {
    if [ -z "${PROJECT_ID:-}" ]; then
        [ "$DRY_RUN" -eq 1 ] || die "PROJECT_ID is not set. export PROJECT_ID=<your-project-id>"
        PROJECT_ID="PROJECT_ID"
    fi
    IMAGE="${IMAGE:-$REGION-docker.pkg.dev/$PROJECT_ID/ml/$SERVICE}"
}

# jq makes the output readable; without it the raw JSON still tells the story.
JQ() {
    if command -v jq >/dev/null 2>&1; then
        printf 'jq %s' "$1"
    else
        printf 'cat'
    fi
}

resolve_url() {
    [ -n "${URL:-}" ] && return 0
    URL="$(capture "gcloud run services describe $SERVICE --region $REGION \
--format 'value(status.url)'" "https://$SERVICE-000000000000.$REGION.run.app")"
    [ -n "$URL" ] || die "no URL for service $SERVICE in $REGION. Deployed? Right gcloud config?"
    CANDIDATE_URL="https://candidate---${URL#https://}"
}

# --- steps ---------------------------------------------------------------

step_prepare() { # before class: build, push, deploy v1
    need_project
    say "build both images and deploy v1 (about 4 minutes in total)"
    run "docker build --platform linux/amd64 -t $IMAGE:v1 ."
    run "docker push $IMAGE:v1"
    say "v2 image: promote locally first, build, then put the alias back"
    run "uv run python scripts/promote.py 2"
    run "docker build --platform linux/amd64 -t $IMAGE:v2 ."
    run "docker push $IMAGE:v2"
    run "git checkout registry/registry.json"
    say "deploy v1 and warm the container"
    run "gcloud run deploy $SERVICE --image $IMAGE:v1 --region $REGION --allow-unauthenticated"
    resolve_url
    run "curl -s $URL/health | $(JQ .)"
}

step_health() { # 1 · the contract
    resolve_url
    run "curl -s $URL/health | $(JQ .)"
}

step_predict() { # 1 · the contract
    resolve_url
    run "curl -s -X POST $URL/predict -H 'Content-Type: application/json' \
-d '{\"hours_ahead\": 6}' | $(JQ .)"
    say "and the contract rejects what the model cannot answer"
    run "curl -s -o /dev/null -w 'HTTP %{http_code}\\n' -X POST $URL/predict \
-H 'Content-Type: application/json' -d '{\"hours_ahead\": 48}'"
}

step_logs() { # 2 · one JSON line per prediction
    run "gcloud logging read 'resource.type=\"cloud_run_revision\" AND \
resource.labels.service_name=\"$SERVICE\" AND jsonPayload.event=\"prediction\"' \
--limit 5 --freshness 1h --format 'value(jsonPayload)'"
}

step_gate() { # 3 · promotion is a decision, not a deploy
    say "changes registry/registry.json; reset with: git checkout registry/registry.json"
    run "uv run python scripts/promote.py 2"
    run "uv run python scripts/champion_version.py"
}

step_candidate() { # 4+5 · a revision nobody sees yet
    need_project
    resolve_url
    run "gcloud run deploy $SERVICE --image $IMAGE:v2 --region $REGION \
--no-traffic --tag candidate"
    say "the candidate answers 2, the service still answers 1"
    run "curl -s $CANDIDATE_URL/health | $(JQ -r\ .model_version)"
    run "curl -s $URL/health | $(JQ -r\ .model_version)"
}

step_canary() { # 6 · 10 % of the traffic
    resolve_url
    run "gcloud run services update-traffic $SERVICE --region $REGION --to-tags candidate=10"
    # The new split can take a few seconds to reach every front end; counting at once can
    # show 40 × "1" even at 10 %.
    say "give the new split a moment to arrive"
    run "sleep ${CANARY_WAIT:-10}"
    say "40 requests, counted by model version"
    run "for i in \$(seq 40); do curl -s $URL/health | $(JQ -r\ .model_version); done \
| sort | uniq -c"
}

step_rollback() { # 7 · back to the previous revision
    resolve_url
    run "gcloud run revisions list --service $SERVICE --region $REGION"
    # Set PREVIOUS_REVISION yourself to skip the lookup and paste the name from the list above.
    previous="${PREVIOUS_REVISION:-}"
    if [ -z "$previous" ]; then
        # The revision that still serves the share the canary did not take. Not "the second
        # newest revision": an old candidate from an earlier run would win that.
        previous="$(capture "gcloud run services describe $SERVICE --region $REGION --format json \
| jq -r '[.status.traffic[] | select((.percent // 0) > 0 and .tag == null)] | max_by(.percent) | .revisionName // empty'" \
"$SERVICE-00001-abc")"
    fi
    [ -n "$previous" ] || die "no revision serves traffic besides the candidate. Read the name from the list above, then:
  PREVIOUS_REVISION=<name> ./demo/demo.sh rollback"
    say "all traffic back to $previous"
    run "gcloud run services update-traffic $SERVICE --region $REGION \
--to-revisions $previous=100"
    run "curl -s $URL/health | $(JQ -r\ .model_version)"
}

step_status() { # where are we right now?
    resolve_url
    run "gcloud run revisions list --service $SERVICE --region $REGION"
    say "service URL, then candidate tag URL"
    run "curl -s $URL/health | $(JQ -r\ .model_version)"
    run "curl -s $CANDIDATE_URL/health | $(JQ -r\ .model_version) || true"
}

step_all() { # rehearsal
    for s in health predict logs gate candidate canary rollback status; do
        printf '\n%s================ %s ================%s\n' "$BOLD" "$s" "$RESET"
        "step_$s"
    done
}

usage() {
    cat <<'EOF'
usage: ./demo/demo.sh [--dry-run] <step>

steps
  prepare    build + push v1 and v2, deploy v1, warm it        (before class)
  health     GET /health                                       (live 1)
  predict    POST /predict, and a 422                          (live 1)
  logs       the structured prediction log                     (live 2)
  gate       the promotion gate, locally                       (live 3)
  candidate  deploy v2 with --no-traffic --tag candidate       (live 4+5)
  canary     10 % to the candidate, 40 requests counted        (live 6)
  rollback   all traffic back to the previous revision         (live 7)
  status     current traffic split and what each URL answers
  all        every live step in order (rehearsal)

env  PROJECT_ID (prepare/candidate)  REGION=europe-west6  SERVICE=lucerne-temp
     IMAGE, URL are derived when not set
EOF
}

main() {
    while [ $# -gt 0 ] && [ "${1#--}" != "$1" ]; do
        case "$1" in
            --dry-run) DRY_RUN=1; shift ;;
            --help) usage; exit 0 ;;
            *) die "unknown option $1" ;;
        esac
    done
    [ $# -eq 1 ] || { usage; exit 1; }
    case "$1" in
        prepare|health|predict|logs|gate|candidate|canary|rollback|status|all)
            [ "$DRY_RUN" -eq 1 ] && say "dry run: printing commands, running none"
            "step_$1"
            ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
