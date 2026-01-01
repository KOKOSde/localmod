# LocalMod - Awesome List Submissions

This document contains prepared entries for submitting LocalMod to various GitHub awesome lists.

---

## 🥇 TIER 1: Priority Submissions

---

### 1. awesome-selfhosted/awesome-selfhosted-data (258K+ stars)

**Repository:** https://github.com/awesome-selfhosted/awesome-selfhosted-data

**Submission Method:** Create a Pull Request with a new YAML file

**File to Create:** `software/localmod.yml`

```yaml
name: LocalMod
website_url: https://github.com/KOKOSde/localmod
source_code_url: https://github.com/KOKOSde/localmod
description: Content moderation API with toxicity, PII, prompt injection, spam, and NSFW detection for text and images (alternative to Amazon Comprehend, Perspective API).
licenses:
  - MIT
platforms:
  - Python
  - Docker
tags:
  - Generative Artificial Intelligence (GenAI)
```

**PR Title:** `Add LocalMod - Self-hosted content moderation API`

**PR Description:**

```markdown
## Add LocalMod - Self-hosted Content Moderation API

**Project:** [LocalMod](https://github.com/KOKOSde/localmod)

**What it is:** Fully offline, self-hosted content moderation API with 6 classifiers:
- Toxicity detection (weighted ensemble of 4 models)
- PII detection (emails, phones, SSNs, credit cards)  
- Prompt injection detection (for LLM applications)
- Spam detection
- NSFW text detection
- NSFW image detection

**Why it belongs here:**
- Self-hosted alternative to expensive cloud moderation services (Amazon Comprehend, Perspective API, OpenAI Moderation)
- 100% offline — data never leaves your infrastructure (GDPR/HIPAA friendly)
- Outperforms Amazon Comprehend (0.75 vs 0.74 balanced accuracy on CHI 2025 benchmark)
- Text + image moderation in a single API
- Single Docker deployment
- MIT licensed, actively maintained

**Key differentiator:** Only self-hosted solution combining all 6 classifiers (including image moderation) with prompt injection detection for LLM apps.

**Checklist:**
- [x] Read CONTRIBUTING.md
- [x] Entry follows required format
- [x] Project is open source (MIT)
- [x] Project is actively maintained
- [x] Project has working installation instructions
```

**⚠️ Important Note:** awesome-selfhosted requires the first release to be **4+ months old**. Check your release history before submitting.

---

### 2. selfh.st/apps (500K+ monthly views)

**Website:** https://selfh.st/apps

**Submission Method:** Use their submission form at https://selfh.st/apps/submit/ or email

**Suggested Entry Text:**

```
Name: LocalMod
URL: https://github.com/KOKOSde/localmod
Category: AI/Machine Learning, Security
License: MIT

Description: Self-hosted content moderation API with 6 classifiers: toxicity, PII, prompt injection, spam, NSFW text, and NSFW image detection. Outperforms Amazon Comprehend (0.75 vs 0.74 balanced accuracy). 100% offline — data never leaves your server. Built with FastAPI, Docker-ready. Self-hosted alternative to Amazon Comprehend and Perspective API.
```

---

### 3. mjhea0/awesome-fastapi (8K+ stars)

**Repository:** https://github.com/mjhea0/awesome-fastapi

**Submission Method:** Pull Request to add entry in README.md

**Section:** `### Open Source Projects`

**Entry to Add (alphabetically placed):**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted content moderation API with 6 classifiers (toxicity, PII, prompt injection, spam, NSFW text, NSFW image). Outperforms Amazon Comprehend.
```

**File Path:** `README.md` → Under `### Open Source Projects` section (insert alphabetically between "JSON-RPC Server" and "Mailer")

**PR Title:** `Add LocalMod to Open Source Projects`

**PR Description:**

```markdown
## Add LocalMod - Content Moderation API

**Project:** [LocalMod](https://github.com/KOKOSde/localmod)

**What it is:** Production-ready FastAPI content moderation API with 6 classifiers:
- Toxicity (weighted ensemble, benchmarked against commercial APIs)
- PII detection (emails, phones, SSNs, credit cards)
- Prompt injection (for LLM applications)
- Spam, NSFW text, NSFW image detection

**Why it belongs in awesome-fastapi:**
- Built entirely with FastAPI
- Async endpoints with batch processing
- Pydantic models for request/response validation
- Docker-ready deployment
- Well-documented API with OpenAPI spec
- Actively maintained, MIT licensed

**Key Features:**
- Self-hosted alternative to cloud moderation APIs
- Outperforms Amazon Comprehend (0.75 vs 0.74 balanced accuracy)
- 100% offline capable after model download
- Text + image moderation in single API
```

---

### 4. Shubhamsaboo/awesome-llm-apps (15K+ stars)

**Repository:** https://github.com/Shubhamsaboo/awesome-llm-apps

**Submission Method:** Pull Request or Issue

**Suggested Section:** Create new section "🛡️ LLM Safety & Guardrails" or add to existing AI Agents section

**Entry to Add:**

```markdown
### 🛡️ LLM Safety & Guardrails

*   [🛡️ LocalMod - Content Moderation API](https://github.com/KOKOSde/localmod) - Self-hosted prompt injection detector and content moderation API with 6 classifiers (toxicity, PII, prompt injection, spam, NSFW text, NSFW image). Runs 100% offline. Outperforms Amazon Comprehend.
```

**PR Title:** `Add LocalMod - LLM Safety Guardrails`

**PR Description:**

```markdown
## Add LocalMod - LLM Safety & Content Moderation

**Project:** [LocalMod](https://github.com/KOKOSde/localmod)

**What it is:** Self-hosted content moderation API designed for LLM applications with:
- **Prompt injection detection** using DeBERTa model
- Toxicity detection (weighted ensemble of 4 models)
- PII detection (emails, phones, SSNs, credit cards)
- Spam, NSFW text, and NSFW image detection

**Why it belongs in awesome-llm-apps:**
- Specifically designed as a guardrail for LLM applications
- Prompt injection detector blocks jailbreak attempts
- Runs 100% offline — no data sent to cloud
- Outperforms Amazon Comprehend on benchmarks
- FastAPI-based, easy to integrate with any LLM app
- MIT licensed, actively maintained

This fills a gap in the current list — there's no self-hosted LLM safety/guardrails solution listed.
```

---

## 🥈 TIER 2: LLM Security Lists

---

### 5. corca-ai/awesome-llm-security

**Repository:** https://github.com/corca-ai/awesome-llm-security

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted prompt injection detector and content moderation API. Runs 100% offline with 6 classifiers including toxicity, PII, and NSFW detection.
```

**Section:** "Defensive Tools" or "Guardrails"

---

### 6. wearetyomsmnv/Awesome-LLMSecOps

**Repository:** https://github.com/wearetyomsmnv/Awesome-LLMSecOps

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted LLM guardrails with prompt injection detection, toxicity filtering, PII detection, and image moderation. Outperforms Amazon Comprehend.
```

---

### 7. brinhosa/awesome-ai-security

**Repository:** https://github.com/brinhosa/awesome-ai-security

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Open-source content moderation API with prompt injection detection. Self-hosted alternative to cloud moderation services.
```

---

### 8. ottosulin/awesome-ai-security

**Repository:** https://github.com/ottosulin/awesome-ai-security

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted AI content moderation with 6 classifiers (toxicity, PII, prompt injection, spam, NSFW text/image). MIT licensed.
```

---

### 9. tldrsec/prompt-injection-defenses

**Repository:** https://github.com/tldrsec/prompt-injection-defenses

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted prompt injection detector using DeBERTa model. Includes additional classifiers for toxicity, PII, spam, and NSFW content.
```

**Section:** "Detection Tools" or "Open Source Defenses"

---

### 10. ThuCCSLab/Awesome-LM-SSP

**Repository:** https://github.com/ThuCCSLab/Awesome-LM-SSP

**Entry Format:**

```markdown
- [LocalMod](https://github.com/KOKOSde/localmod) - Self-hosted content safety API with prompt injection detection, toxicity filtering, PII detection, and NSFW moderation for text and images.
```

---

## 📋 PR Checklist for All Submissions

Before submitting to each list:

- [ ] Fork the repository
- [ ] Create a new branch (e.g., `add-localmod`)
- [ ] Check their CONTRIBUTING.md for specific requirements
- [ ] Verify the entry format matches existing entries
- [ ] Place entry alphabetically if required
- [ ] Use their exact markdown/YAML format
- [ ] Submit PR with descriptive title and body
- [ ] Be responsive to reviewer feedback

---

## 🎯 Submission Priority Order

1. **awesome-selfhosted-data** — Highest impact, but requires 4-month-old release
2. **awesome-fastapi** — Good fit, likely quick approval
3. **awesome-llm-apps** — Great visibility for LLM use case
4. **selfh.st/apps** — Additional exposure via newsletter
5. **LLM security lists** — Targeted audience for prompt injection feature

---

## 📝 Notes

- **awesome-selfhosted** requires first release to be 4+ months old
- Most lists prefer alphabetical ordering
- Keep descriptions concise (<250 characters for some lists)
- Avoid terms like "open-source" or "free" (implied by the list context)
- Mention "alternative to X" format where appropriate

