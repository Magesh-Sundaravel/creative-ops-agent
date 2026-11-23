# Code checks the facts, the LLM judges

Review and Handoff do their client-stack reads and every check that can be decided from metadata in code: Asset status, approved channels, the channels in the Usage Rights, expiry against the deadline, and dimensions against the brand's `channel_dimensions`. Code adds those Findings to the result directly, so the model can't drop or soften them. The model gets the facts in its prompt and judges only what needs judgement, such as territory against the brief's free-text Usage Rights, tags against the dos and don'ts, fit to the objective, and which Assets cover which deliverable. Code computes the Review Verdict from the Findings. As a result, Review's LLM has no DAM tools at all, only `submit_review`. Please don't give it read tools "in case": a tool the model doesn't need is privilege it doesn't need.

## Considered Options

- **The LLM judges everything from DAM tool results.** Simpler, but it can miss a deterministic flaw (such as an expired date or a 1920x1080 TikTok), and nothing would catch the miss.
- **Pure code.** It can't judge tone, territory against free text, or how deliverables map to Assets without regex-driven control flow, which CLAUDE.md forbids.
