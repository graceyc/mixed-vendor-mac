# mac_mixed/vendor_clients/claude.py
import os
import anthropic

def call_claude(system_message: str, messages: list[str], model: str = None, temperature: float = 1.0, max_tokens: int = 2048):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    mdl = model or os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    # Claude: system + user content (history concatenated)
    user_payload = "\n\n---\n\n".join(messages)
    resp = client.messages.create(
        model=mdl,
        temperature=float(temperature),
        max_tokens=max_tokens,
        system=system_message,
        messages=[{"role": "user", "content": user_payload}],
    )
    # combine blocks
    parts = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
