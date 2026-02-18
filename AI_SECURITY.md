# AI Security 

This repository implements the following safeguards:

1.  **LLM01: Prompt Injection:** We use XML delimiters in system prompts to separate User data from System instructions.
2.  **LLM02: Insecure Output Handling:** All Agent outputs are validated via Pydantic schemas before execution. We never use `eval()` or `exec()`.
3.  **LLM04: Model Denial of Service:** We implement strict token limits (`max_tokens=500`) and request timeouts (`timeout=10s`) to prevent simulation loops.

## Risk Acceptance

Users running this arena acknowledge that LLMs are non-deterministic. While we implement guardrails, agents may occasionally generate offensive or illogical content.
