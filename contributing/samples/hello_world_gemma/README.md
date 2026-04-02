# Hello World — Gemma 3

This sample demonstrates using **Gemma 3** models with ADK via the `Gemma`
class. The `Gemma` class provides workarounds for Gemma 3's lack of native
function calling and system instruction support.

## When to use this

Use this approach for **Gemma 3 models only**. For Gemma 4 and later, use the
standard `Gemini` class directly — see the
[`hello_world_gemma4/`](../hello_world_gemma4/) sample.

## Running this sample

```bash
# From the repository root
adk run contributing/samples/hello_world_gemma

# Or via the web UI
adk web contributing/samples
```

## Related samples

- [`hello_world_gemma4/`](../hello_world_gemma4/) — Gemma 4 via standard
  Gemini class (recommended for Gemma 4+)
- [`hello_world_gemma3_ollama/`](../hello_world_gemma3_ollama/) — Gemma 3 via
  Ollama
