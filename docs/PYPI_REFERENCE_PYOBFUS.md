# PyPI Publishing Reference - pyobfus Project

**Created**: 2026-01-02
**Purpose**: Reference for cardiac-shared PyPI release based on pyobfus experience

---

## 1. Reference Project Info

| Item | Value |
|------|-------|
| **Project** | pyobfus |
| **PyPI URL** | https://pypi.org/project/pyobfus/ |
| **GitHub** | https://github.com/zhurong2020/pyobfus |
| **Local Path** | `C:\onedrive\msft\OneDrive - MSFT\rong\3-job\program\pyobfus` |
| **Version** | 0.3.2 |
| **License** | Apache-2.0 |

---

## 2. pyproject.toml Template

Key sections from pyobfus that should be adapted for cardiac-shared:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cardiac-shared"
version = "0.2.0"
description = "Shared utilities for cardiac imaging analysis"
readme = {file = "README.md", content-type = "text/markdown"}
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Rong Zhu", email = "zhurong0525@gmail.com"}
]
maintainers = [
    {name = "Rong Zhu", email = "zhurong0525@gmail.com"}
]
keywords = ["cardiac", "medical-imaging", "dicom", "nifti", "ct-scan"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]

[project.urls]
Homepage = "https://github.com/zhurong2020/cardiac-shared"
Documentation = "https://github.com/zhurong2020/cardiac-shared/blob/main/README.md"
Repository = "https://github.com/zhurong2020/cardiac-shared"
"Bug Tracker" = "https://github.com/zhurong2020/cardiac-shared/issues"
"Changelog" = "https://github.com/zhurong2020/cardiac-shared/blob/main/CHANGELOG.md"

[tool.setuptools.packages.find]
where = ["."]
include = ["cardiac_shared*"]
exclude = ["tests*", "docs*", "examples*"]
```

---

## 3. PyPI Publishing Process

### 3.1 First-time Setup

```bash
# Install build tools
pip install build twine

# Create PyPI account
# Go to: https://pypi.org/account/register/

# Create API token
# Go to: https://pypi.org/manage/account/token/
# Save token to ~/.pypirc or use environment variable
```

### 3.2 Build Package

```bash
cd /path/to/cardiac-shared

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# This creates:
# dist/cardiac_shared-0.2.0-py3-none-any.whl
# dist/cardiac_shared-0.2.0.tar.gz
```

### 3.3 Upload to PyPI

```bash
# Test on TestPyPI first (recommended)
twine upload --repository testpypi dist/*

# Upload to real PyPI
twine upload dist/*
```

### 3.4 Verify Installation

```bash
# From TestPyPI
pip install --index-url https://test.pypi.org/simple/ cardiac-shared

# From PyPI (after release)
pip install cardiac-shared
```

---

## 4. Stripe Integration (For Future Licensing)

pyobfus uses Stripe + Cloudflare Workers for license management:

### 4.1 Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   User Purchase  │────▶│  Stripe Checkout │────▶│ Webhook to       │
│   (Website)      │     │  (Payment)       │     │ Cloudflare Worker│
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   pyobfus CLI    │◀────│  License Server  │◀────│   KV Storage     │
│   (Verification) │     │  (API)           │     │   (Licenses)     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### 4.2 Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Payment | Stripe Checkout | Process payments |
| Webhook | Cloudflare Worker | Handle payment events, create licenses |
| Storage | Cloudflare KV | Store license data |
| Verification | REST API | Validate license keys |

### 4.3 Configuration Files

```
pyobfus/
├── .env.stripe.example    # Stripe keys template
├── cloudflare-worker/
│   ├── src/index.js       # Worker code (webhook handler)
│   ├── wrangler.toml      # Cloudflare config
│   └── README.md          # Deployment guide
└── pyobfus_pro/
    └── license.py         # License validation client
```

### 4.4 .env.stripe Template

```bash
# Test Mode Keys (development)
STRIPE_TEST_PUBLISHABLE_KEY=pk_test_...
STRIPE_TEST_SECRET_KEY=sk_test_...

# Live Mode Keys (production)
STRIPE_LIVE_PUBLISHABLE_KEY=pk_live_...
STRIPE_LIVE_SECRET_KEY=sk_live_...

# Product Configuration
STRIPE_PRODUCT_ID=prod_...
STRIPE_PRICE_ID=price_...

# Webhook Secret
STRIPE_WEBHOOK_SECRET=whsec_...

# Environment
STRIPE_ENVIRONMENT=test  # or 'live'
```

---

## 5. License Data Schema

From pyobfus (can be adapted for cardiac projects):

```json
{
  "license_key": "CARDIAC-XXXX-XXXX-XXXX-XXXX",
  "email": "customer@hospital.com",
  "status": "active",
  "created_at": "2026-01-02T00:00:00Z",
  "stripe_session_id": "cs_xxx",
  "stripe_customer_id": "cus_xxx",
  "devices": ["device-id-1", "device-id-2"],
  "expires_at": "2027-01-02T00:00:00Z",
  "product": "vbca",
  "tier": "professional"
}
```

**Status Values**:
- `active`: License is valid
- `suspended`: Temporarily disabled
- `revoked`: Permanently disabled
- `expired`: Past expiration date

---

## 6. Cloudflare Worker Deployment

### 6.1 Setup

```bash
# Install Wrangler
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy worker
cd cloudflare-worker
wrangler deploy
```

### 6.2 Configure Secrets

```bash
# Add Stripe keys
echo "sk_live_..." | wrangler secret put STRIPE_SECRET_KEY
echo "whsec_..." | wrangler secret put STRIPE_WEBHOOK_SECRET
```

### 6.3 KV Management

```bash
# Create KV namespace
wrangler kv namespace create "LICENSES"

# Add test license
wrangler kv key put --remote \
  --namespace-id=YOUR_NAMESPACE_ID \
  "CARDIAC-TEST-1234" \
  '{"license_key":"CARDIAC-TEST-1234","status":"active"}'
```

---

## 7. API Endpoints Reference

### 7.1 Health Check
```
GET /api/health
Response: {"status": "ok", "service": "license-server"}
```

### 7.2 Verify License
```
POST /api/verify
Body: {"license_key": "CARDIAC-XXXX", "device_id": "unique-id"}
Response: {"valid": true, "features": {...}}
```

### 7.3 Stripe Webhook
```
POST /api/webhook/stripe
Headers: stripe-signature
Body: Stripe event payload
```

---

## 8. Project Structure Reference

pyobfus structure for reference:

```
pyobfus/
├── pyproject.toml          # PyPI configuration
├── README.md               # Package documentation
├── CHANGELOG.md            # Version history
├── LICENSE                 # Apache-2.0
├── .gitignore
├── .env.stripe.example     # Stripe config template
│
├── pyobfus/                # Main package (public, PyPI)
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   └── ...
│
├── pyobfus_pro/            # Pro features (private, not on PyPI)
│   ├── __init__.py
│   ├── license.py          # License validation
│   └── ...
│
├── cloudflare-worker/      # License server
│   ├── src/index.js
│   ├── wrangler.toml
│   └── README.md
│
├── docs/                   # Documentation
├── tests/                  # Unit tests
└── examples/               # Usage examples
```

---

## 9. Next Steps for cardiac-shared

1. [ ] Update pyproject.toml with correct metadata
2. [ ] Create CHANGELOG.md
3. [ ] Test build locally
4. [ ] Upload to TestPyPI
5. [ ] Test installation from TestPyPI
6. [ ] Upload to PyPI
7. [ ] (Future) Set up Stripe + Cloudflare for paid features
