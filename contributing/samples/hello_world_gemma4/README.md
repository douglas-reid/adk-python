# Hello World — Gemma 4

This sample demonstrates using **Gemma 4** with ADK via the standard `Gemini`
class. Gemma 4 supports native function calling and system instructions, so no
special workaround classes are needed.

### Gemma 4 (this sample)

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini

root_agent = Agent(
    model=Gemini(model="gemma-4-31b-it"),  # gemma-4-26b-a4b-it or gemma-4-31b-it
    ...
)
```

### Gemma 3

```python
from google.adk.agents.llm_agent import Agent
from google.adk.models.gemma_llm import Gemma

root_agent = Agent(
    model=Gemma(model="gemma-3-27b-it"),
    ...
)
```

See the [`hello_world_gemma/`](../hello_world_gemma/) sample for the full
Gemma 3 example.

## Why separate classes?

The `Gemma` and `Gemma3Ollama` classes exist because Gemma 3 lacks native
function calling and system instruction support. They provide workarounds by:

- Injecting tool declarations into text prompts
- Parsing function calls from model text responses
- Converting system instructions to user-role messages

Gemma 4 doesn't need any of this — it works natively with the standard
`Gemini` class (via Gemini API) and `LiteLlm` class (via other providers like
Ollama).

## Running this sample

```bash
# From the repository root
adk run contributing/samples/hello_world_gemma4
```

## Related samples

- [`hello_world_gemma/`](../hello_world_gemma/) — Gemma 3 via Gemini API
- [`hello_world_gemma3_ollama/`](../hello_world_gemma3_ollama/) — Gemma 3 via
  Ollama
