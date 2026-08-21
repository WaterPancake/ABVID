# OpenClaw vehicle-audio-curator

This is a private, approval-gated OpenClaw agent workspace template. It discovers
and evaluates audio sources, writes review proposals, and runs the repository's
license-aware collector only after an operator approves a source ID.

The official OpenClaw CLI is installed locally under `.artifacts/` in this
checkout, and the curator agent is registered in the project-local state directory
`.artifacts/openclaw-state/`. To repeat setup on another machine:

```bash
bash openclaw/setup_vehicle_audio_curator.sh
export VEHICLE_AUDIO_PROJECT_ROOT="$PWD"
```

The curator is configured for `deepseek/deepseek-v4-flash`. The official DeepSeek
provider plugin is installed, but the CLI still needs `DEEPSEEK_API_KEY` before it
can execute an agent turn. On another machine, install it with:

```bash
openclaw plugins install @openclaw/deepseek-provider
```

No channel or daemon was configured.

Start it only as a private/local agent. Do not add a messaging binding until the
operator understands that exec approvals are host guardrails, not user identity
boundaries.

## Safety configuration

Use OpenClaw's host approval policy and keep the agent out of full/YOLO mode. The
following sets a conservative local default; configure the same values for the
`vehicle-audio-curator` agent scope in the Control UI or approvals CLI:

```bash
openclaw exec-policy set \
  --host gateway \
  --security allowlist \
  --ask always \
  --ask-fallback deny
```

The desired per-agent approvals document is:

```json
{
  "version": 1,
  "agents": {
    "vehicle-audio-curator": {
      "security": "allowlist",
      "ask": "always",
      "askFallback": "deny",
      "allowlist": []
    }
  },
  "defaults": {
    "security": "deny",
    "ask": "off",
    "askFallback": "deny"
  }
}
```

An empty allowlist plus `ask: always` ensures host commands require an operator
approval. The workspace instructions add a second semantic gate: the agent must
receive `approve download: <source-id>` before it invokes the collector without
`--dry-run`.

## Workflow

1. Ask the agent to inspect `configs/audio_sources.yaml` and find a missing source.
2. It searches and writes an evidence-backed proposal under
   `openclaw/vehicle-audio-curator/proposals/`.
3. Review the license, attribution, source/session identity, and label yourself.
4. Send `approve download: <source-id>` only for a source already in the catalog.
5. Approve the resulting host command in OpenClaw, then inspect the sidecar and
   `data/collection_manifest.jsonl`.

The initial discovery prompt is in `vehicle-audio-curator/DISCOVERY_PROMPT.md`.
Run it after configuring model authentication, for example:

```bash
export DEEPSEEK_API_KEY="..."
export OPENCLAW_STATE_DIR="$PWD/.artifacts/openclaw-state"
export VEHICLE_AUDIO_PROJECT_ROOT="$PWD"
.artifacts/openclaw-cli/node_modules/.bin/openclaw agent --local \
  --agent vehicle-audio-curator \
  --model deepseek/deepseek-v4-flash \
  --message-file openclaw/vehicle-audio-curator/DISCOVERY_PROMPT.md
```

The agent does not schedule itself and does not download automatically. If future
scheduled curation is desired, add a cron job only after this interactive workflow
has been tested.
