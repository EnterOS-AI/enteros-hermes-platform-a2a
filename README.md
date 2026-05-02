# hermes-platform-molecule-a2a

A [hermes-agent](https://github.com/NousResearch/hermes-agent) platform
adapter that delivers Molecule A2A peer-agent messages to a running
hermes daemon over HTTP, and POSTs agent replies back to the
molecule-runtime callback URL.

## Why this exists

Every Molecule workspace runtime needs MCP-style "push" parity: a
single long-lived agent session that receives new peer messages
mid-thread instead of cold-starting one process per message.

- **claude-code**: native MCP `notifications/claude/channel`
- **codex**: persistent `codex app-server` JSON-RPC stdio
- **hermes**: this plugin (HTTP listener inside the hermes gateway)

Without this, hermes workspaces lose conversation continuity (every
A2A message hits stateless `/v1/chat/completions`) and pay subprocess
startup cost on every peer message.

## Requires

This plugin depends on three small additions to hermes-agent that
are not yet upstream:

1. `PluginContext.register_platform_adapter(name, adapter_class, requirements_check=None)`
2. `GatewayConfig.plugin_platforms: Dict[str, PlatformConfig]`, populated
   by `from_dict` for any platform name claimed by a discovered plugin.
3. `GatewayRunner._create_plugin_adapter(name, config)` boot path.

Until they land, install the plugin against a hermes fork carrying
[`feat/platform-adapter-plugins`](https://github.com/NousResearch/hermes-agent).
See `docs/integrations/hermes-platform-plugins-upstream-pr.md` in
molecule-core for the upstream proposal.

## Install

```bash
pip install hermes-platform-molecule-a2a
```

The plugin auto-registers via the `hermes_agent.plugins` entry point.
No imports or wiring on the hermes side are needed beyond enabling the
platform in config.

## Configure

In your hermes `config.yaml`:

```yaml
platforms:
  molecule-a2a:
    enabled: true
    extra:
      host: "127.0.0.1"             # default
      port: 8645                     # default
      callback_url: "http://runtime:9999/a2a/reply"
      shared_secret: "..."           # required in production; empty
                                     # = open mode for localhost only
```

When the molecule-runtime POSTs an inbound A2A message and includes
its own `callback_url` in the payload, that overrides the default for
that chat — letting the runtime route different peer threads to
different callback endpoints.

## Wire shape

### Inbound (runtime → hermes)

```http
POST /a2a/inbound
X-Molecule-A2A-Secret: <shared_secret>
Content-Type: application/json

{
  "chat_id": "peer-uuid",
  "peer_id": "peer-uuid",
  "peer_name": "ops-agent",
  "peer_role": "sre",
  "content": "the message text",
  "message_id": "uuid-or-monotonic",
  "callback_url": "http://runtime:9999/a2a/reply",
  "thread_id": "optional"
}

→ 200 {"ok": true, "queued": true}
```

The handler returns immediately; hermes processes the message in a
background task and delivers the reply through `send()`.

### Outbound (hermes → runtime)

```http
POST <callback_url>
X-Molecule-A2A-Secret: <shared_secret>
Content-Type: application/json

{
  "chat_id": "peer-uuid",
  "content": "agent reply text",
  "reply_to": "msg-id-of-inbound",
  "metadata": {...}
}
```

## Validate

Unit tests (no hermes runtime required):

```bash
pip install -e '.[test]'
pytest tests/ -q
```

End-to-end against a hermes-agent install with the patch:

```bash
HERMES_REPO=/path/to/hermes-agent python scripts/e2e_validate.py
```

The E2E script confirms the entry-points discovery → platform registry
→ GatewayConfig routing → adapter boot → HTTP roundtrip pipeline.

## License

MIT
