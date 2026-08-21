#!/usr/bin/env bash
# Create the isolated OpenClaw agent after installing/configuring OpenClaw.
# This script intentionally does not download audio or bind a chat channel.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$PROJECT_ROOT/openclaw/vehicle-audio-curator"
OPENCLAW_CLI="$(command -v openclaw || true)"
if [[ -z "$OPENCLAW_CLI" && -x "$PROJECT_ROOT/.artifacts/openclaw-cli/node_modules/.bin/openclaw" ]]; then
  OPENCLAW_CLI="$PROJECT_ROOT/.artifacts/openclaw-cli/node_modules/.bin/openclaw"
fi
if [[ -z "$OPENCLAW_CLI" ]]; then
  echo "openclaw is not installed or not on PATH." >&2
  echo "Install/configure OpenClaw, then rerun: $0" >&2
  exit 127
fi

if "$OPENCLAW_CLI" agents list --json | grep -q '"id"[[:space:]]*:[[:space:]]*"vehicle-audio-curator"'; then
  echo "vehicle-audio-curator already exists; leaving its config unchanged."
else
  "$OPENCLAW_CLI" agents add vehicle-audio-curator \
    --workspace "$WORKSPACE" \
    --non-interactive \
    --json
fi

"$OPENCLAW_CLI" agents set-identity \
  --agent vehicle-audio-curator \
  --workspace "$WORKSPACE" \
  --from-identity \
  --json

# Keep a local Gateway authenticated and prevent approval bypass through elevated exec.
"$OPENCLAW_CLI" config set gateway.bind loopback
"$OPENCLAW_CLI" config set gateway.auth.mode token
if ! "$OPENCLAW_CLI" config get gateway.auth.token >/dev/null 2>&1; then
  "$OPENCLAW_CLI" config set gateway.auth.token "$(openssl rand -hex 32)"
fi
"$OPENCLAW_CLI" config set tools.elevated.enabled false
"$OPENCLAW_CLI" config set agents.defaults.model.primary deepseek/deepseek-v4-flash

# Ask on every host command; no prompt fallback means unattended scraping cannot run.
"$OPENCLAW_CLI" exec-policy set \
  --host gateway --security allowlist --ask always --ask-fallback deny

echo
cat <<EOF
Created the private agent workspace:
  $WORKSPACE

Set the project root for its sessions:
  export VEHICLE_AUDIO_PROJECT_ROOT="$PROJECT_ROOT"

Recommended safety policy (run on the OpenClaw host):
  "$OPENCLAW_CLI" exec-policy set --host gateway --security allowlist --ask always --ask-fallback deny

Also configure the per-agent host approval scope for vehicle-audio-curator with
security=allowlist, ask=always, askFallback=deny. Do not use full/YOLO mode.
The agent's workspace instructions separately require an explicit
"approve download: <source-id>" message before collection.
EOF
