# PT IQ Architecture

## Purpose
PT IQ is a reverse-search tender intelligence engine. Instead of comparing a business against every tender at click time, the system pre-processes each tender into an Ideal Business Profile (IBP), stores that semantic representation in Qdrant, then matches incoming Business DNA vectors against the tender library.

The current codebase has completed the Monday foundation:
- Groq request routing and key rotation
- Qdrant bootstrap and vector store wrapper
- Shared embedder
- Unit and integration tests for the most critical control paths

## Current Runtime Model
The current runtime model is single-process first.

- All Groq traffic should pass through [ptiq/core/groq_client.py](/D:/APPS/PT IQ/src/ptiq/core/groq_client.py).
- All Groq key health and rate-limit state should be managed by [ptiq/core/keymaster.py](/D:/APPS/PT IQ/src/ptiq/core/keymaster.py).
- All embeddings should be generated through [ptiq/core/embedder.py](/D:/APPS/PT IQ/src/ptiq/core/embedder.py).
- All Qdrant collection creation, upserts, retrieval, and similarity search should go through [ptiq/core/qdrant_setup.py](/D:/APPS/PT IQ/src/ptiq/core/qdrant_setup.py).

This design keeps agents and workflows thin. Agent files should focus on extraction or reasoning logic, not vendor SDK details.

## Core Components
### Groq Layer
[ptiq/core/keymaster.py](/D:/APPS/PT IQ/src/ptiq/core/keymaster.py) is the in-memory state manager for Groq credentials. It loads `keys.json`, tracks request and token budgets, records cooldowns, and chooses the best available key for each request.

[ptiq/core/groq_client.py](/D:/APPS/PT IQ/src/ptiq/core/groq_client.py) is the only Groq transport wrapper the rest of the project should use. It performs token estimation before sending, asks `KeyMaster` for an eligible key, classifies failures, retries across keys on rate limits and transient errors, and normalizes text or JSON responses.

Important design rule:
- A rate-limited key problem and an oversized-request problem are not the same thing.
- The client must shrink or chunk oversized requests before rotating keys.

### Vector Layer
[ptiq/core/embedder.py](/D:/APPS/PT IQ/src/ptiq/core/embedder.py) loads `BAAI/bge-small-en-v1.5` once and exposes a single `embed(text: str) -> list[float]` function. This is the vector contract for both tenders and business profiles.

[ptiq/core/qdrant_setup.py](/D:/APPS/PT IQ/src/ptiq/core/qdrant_setup.py) currently plays two roles:
- bootstrap both core collections if they do not exist
- act as the reusable Qdrant store wrapper

The two core collections are:
- `tender_library`: stores one vector per tender IBP
- `business_dna`: stores business profile vectors and later winner-reference vectors

Both collections currently use:
- vector size `384`
- distance metric `Cosine`

## Why This Structure Works Right Now
This architecture is intentionally centralized around shared gateways.

- It follows DRY in the high-risk areas: one Groq client, one key manager, one embedder, one vector-store wrapper.
- It lowers cognitive load for future agent files because agents can call stable interfaces instead of raw SDKs.
- It allows model choice, rate-limit strategy, and vector-store behavior to evolve without rewriting extraction and matching logic everywhere.

## Known Constraints
The current system is a solid Week 1 base, but it is not yet production-hard.

- `KeyMaster` is process-local. If multiple processes run at once, each process will maintain its own view of Groq budget and cooldown state.
- `qdrant_setup.py` is doing both setup and store-wrapper work. This is acceptable in Phase 1, but should later be split into a thin setup entrypoint plus a dedicated store module.
- Several scaffold files still exist as placeholders and may confuse new contributors if they are not either implemented or removed.

## Current Tests
[tests/unit/test_groq_client.py](/D:/APPS/PT IQ/src/tests/unit/test_groq_client.py) validates the most important Groq control paths:
- oversized requests shrink before failover
- `429` responses trigger cooldown and key rotation
- success responses update live header-driven budget state

[tests/integration/test_qdrant_setup.py](/D:/APPS/PT IQ/src/tests/integration/test_qdrant_setup.py) validates the live Qdrant path:
- collections exist
- dummy vectors can be upserted
- similarity search returns the inserted records

## Next Architecture Cleanup
Before the Tuesday pipeline grows further, the following cleanup is recommended:
- Consolidate duplicate helper logic in `qdrant_setup.py`
- Separate placeholder files from real entry-point files
- Add a lightweight settings module so collection names, vector size, and timeout defaults have a single source of truth
- Later split `qdrant_setup.py` into `qdrant_store.py` plus a thin script entrypoint
