# LocalMod API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required (designed for internal/private deployment).

---

## Endpoints

### GET /

Get API information.

**Response:**
```json
{
  "name": "LocalMod",
  "version": "0.1.0",
  "description": "Fully offline content moderation API",
  "docs": "/docs"
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": ["pii", "toxicity"],
  "device": "cpu"
}
```

---

### GET /classifiers

List available classifiers with descriptions.

**Response:**
```json
{
  "toxicity": {
    "name": "toxicity",
    "version": "1.0.0",
    "description": "Detects toxic content including hate speech, harassment, threats, and profanity."
  },
  "pii": {
    "name": "pii",
    "version": "1.0.0",
    "description": "Detects personally identifiable information using regex patterns and validation."
  },
  "prompt_injection": {
    "name": "prompt_injection",
    "version": "1.0.0",
    "description": "Detects prompt injection attempts and LLM jailbreaks."
  },
  "spam": {
    "name": "spam",
    "version": "1.0.0",
    "description": "Detects spam, promotional content, and scam messages."
  },
  "nsfw": {
    "name": "nsfw",
    "version": "1.0.0",
    "description": "Detects sexually explicit or adult content in text."
  }
}
```

---

### POST /analyze

Analyze text content for safety issues.

**Request Body:**
```json
{
  "text": "string (required, 1-10000 chars)",
  "classifiers": ["toxicity", "pii", "all"],  // optional, default: ["all"]
  "include_explanation": false  // optional
}
```

**Response:**
```json
{
  "flagged": true,
  "results": [
    {
      "classifier": "pii",
      "flagged": true,
      "confidence": 1.0,
      "severity": "medium",
      "categories": ["email"],
      "metadata": {
        "detections": [
          {
            "type": "email",
            "start": 12,
            "end": 32,
            "redacted_preview": "[EMAIL]"
          }
        ],
        "total_count": 1
      },
      "explanation": "Content flagged by pii (confidence: 100.00%, severity: medium)"
    }
  ],
  "summary": "Content flagged for: pii (medium): email",
  "processing_time_ms": 1.23
}
```

**Errors:**
- `422` - Validation error (empty text, text too long)
- `400` - Text exceeds maximum length

---

### POST /analyze/batch

Analyze multiple texts in a single request.

**Request Body:**
```json
{
  "texts": ["text1", "text2", "text3"],  // required, 1-32 items
  "classifiers": ["pii"],  // optional
  "include_explanation": false  // optional
}
```

**Response:**
```json
{
  "results": [
    {
      "flagged": false,
      "results": [...],
      "summary": "Content passed all safety checks.",
      "processing_time_ms": 0.5
    },
    {
      "flagged": true,
      "results": [...],
      "summary": "Content flagged for: pii (medium): email",
      "processing_time_ms": 0.6
    }
  ],
  "total_flagged": 1,
  "processing_time_ms": 1.2
}
```

**Errors:**
- `400` - Batch size exceeds maximum (32)
- `400` - Individual text exceeds maximum length

---

### POST /redact

Redact PII from text.

**Request Body:**
```json
{
  "text": "Contact john@example.com for help",  // required
  "replacement": "[REDACTED]"  // optional, default: type-specific labels
}
```

**Response:**
```json
{
  "original_length": 33,
  "redacted_text": "Contact [EMAIL] for help",
  "redactions": [
    {
      "type": "email",
      "start": 8,
      "end": 24,
      "replacement": "[EMAIL]"
    }
  ],
  "processing_time_ms": 0.45
}
```

**Default Replacement Labels:**
- Email: `[EMAIL]`
- Phone: `[PHONE]`
- SSN: `[SSN]`
- Credit Card: `[CREDIT_CARD]`
- IP Address: `[IP_ADDRESS]`
- Date of Birth: `[DOB]`

---

## Classifier Types

Use these values in the `classifiers` array:

| Value | Description |
|-------|-------------|
| `toxicity` | Hate speech, harassment, threats |
| `pii` | Personally identifiable information |
| `prompt_injection` | LLM jailbreak attempts |
| `spam` | Promotional/scam content |
| `nsfw` | Adult content |
| `all` | Run all classifiers |

---

## Severity Levels

| Level | Value | Description |
|-------|-------|-------------|
| None | `none` | Content passed checks |
| Low | `low` | Minor issues (confidence 50-60%) |
| Medium | `medium` | Moderate issues (confidence 60-75%) |
| High | `high` | Serious issues (confidence 75-90%) |
| Critical | `critical` | Severe issues (SSN, credit cards, 90%+) |

---

## Error Responses

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "Text cannot be empty or whitespace only",
      "type": "value_error"
    }
  ]
}
```

**Bad Request (400):**
```json
{
  "detail": "Text exceeds maximum length of 10000 characters"
}
```

---

## Rate Limits

Default: 100 requests per minute (configurable via `LOCALMOD_RATE_LIMIT`).

---

## Examples

### cURL

```bash
# Analyze text
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Analyze with specific classifiers
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "My SSN is 123-45-6789", "classifiers": ["pii"]}'

# Batch analyze
curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "Email: test@example.com"]}'

# Redact PII
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "Call me at 555-123-4567"}'
```

### Python (requests)

```python
import requests

# Analyze
response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "Hello world", "classifiers": ["pii", "toxicity"]}
)
result = response.json()
print(f"Flagged: {result['flagged']}")

# Batch analyze
response = requests.post(
    "http://localhost:8000/analyze/batch",
    json={"texts": ["Text 1", "Text 2", "Text 3"]}
)
results = response.json()
print(f"Total flagged: {results['total_flagged']}")

# Redact
response = requests.post(
    "http://localhost:8000/redact",
    json={"text": "Email: john@example.com"}
)
print(response.json()["redacted_text"])
```

### JavaScript (fetch)

```javascript
// Analyze
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello world' })
});
const result = await response.json();
console.log(`Flagged: ${result.flagged}`);
```


