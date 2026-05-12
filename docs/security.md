# PT IQ Security Notes

## Scope
These notes cover the Monday foundation only:
- Groq credential handling
- Groq transport behavior
- Qdrant storage behavior
- test-data safety

This is an early-stage security review, not a full audit.

## Current Strengths
- [.gitignore](/D:/APPS/PT IQ/src/.gitignore) excludes `.env` and `keys.json`, reducing the chance of accidental secret commits.
- Groq requests are sent over HTTPS through the official API base by default.
- The project already distinguishes between operational failures and request-shape failures, which reduces unsafe retry behavior.

## Confirmed Risks
### 1. API keys can leak through object repr or debug output
[ptiq/core/keymaster.py](/D:/APPS/PT IQ/src/ptiq/core/keymaster.py) stores `api_key` directly on the `KeyState` dataclass. Because dataclasses generate a default repr, any debug print or traceback that includes a `KeyState` object can expose a live Groq key.

Recommended fix:
- mark the `api_key` field with `repr=False`
- avoid returning or logging raw key-bearing objects

### 2. Raw business payloads can be stored in hosted Qdrant
[ptiq/core/qdrant_setup.py](/D:/APPS/PT IQ/src/ptiq/core/qdrant_setup.py) currently supports storing full `dna_json`, `dna_summary_text`, and company identifiers inside the `business_dna` collection payload.

Risk:
- if real customer uploads later contain sensitive company or financial details, those details could be stored in plaintext payload form on a hosted vector database

Recommended fix:
- minimize payloads before upsert
- keep only fields that are required for retrieval and workflow continuity
- avoid storing raw sensitive source material in Qdrant

### 3. Caller-supplied content is forwarded to Groq without redaction
[ptiq/core/groq_client.py](/D:/APPS/PT IQ/src/ptiq/core/groq_client.py) accepts message payloads and forwards them directly to Groq. Once PDF, DOCX, and business-document parsing are connected, private content will leave the local machine unless we add pre-send redaction or minimization.

Recommended fix:
- add a sanitization/minimization layer before Groq calls
- define which document fields are allowed to leave the machine
- keep financial and identity-sensitive content out of prompts unless explicitly required

### 4. Groq endpoint can be overridden by environment configuration
[ptiq/core/groq_client.py](/D:/APPS/PT IQ/src/ptiq/core/groq_client.py) allows `GROQ_BASE_URL` to be set from the environment. That is useful for testing, but it also means a bad environment can redirect bearer-auth requests to an unintended host.

Recommended fix:
- allowlist trusted Groq hosts in production mode
- treat non-default Groq base URLs as test-only configuration

### 5. Integration tests write into live collections
[tests/integration/test_qdrant_setup.py](/D:/APPS/PT IQ/src/tests/integration/test_qdrant_setup.py) writes directly into the configured `tender_library` and `business_dna` collections.

Risk:
- test artifacts can pollute production search results
- test payloads can persist in shared infrastructure

Recommended fix:
- use dedicated test collection names or a dedicated test Qdrant cluster
- at minimum, mark test payloads explicitly and keep them isolated from live data paths

## Risks Not Yet Fully Addressed
- `KeyMaster` is not multi-process safe, so separate processes can make conflicting assumptions about available Groq budget.
- Existing Qdrant collections are created if missing, but not yet validated for drift in vector size or distance metric.
- `Retry-After` handling currently assumes numeric seconds and does not yet parse HTTP-date forms.
- The future business-document pipeline does not yet enforce data-classification or field-level redaction rules.

## Security Priorities For Phase 1
1. Prevent Groq key leakage through repr or logging.
2. Do not store raw sensitive business data in Qdrant payloads.
3. Add a sanitization policy before document content is sent to Groq.
4. Separate integration-test data from real collections.
5. Later, introduce safer production configuration validation for external endpoints.
