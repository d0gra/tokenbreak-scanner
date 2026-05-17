# GitHub Repository Setup Guide

Copy these values when creating the repo on GitHub.

## Repository Settings

| Field | Value |
|---|---|
| **Repository name** | `tokenbreak-scanner` |
| **Description** | 🔐 Detect TokenBreak adversarial vulnerabilities in LLMs, classifiers, and encoders. Scan HuggingFace and custom model artifacts for BPE/WordPiece tokenization risks before production deployment. Crucial for AI supply chain scanning. |
| **Visibility** | **Public** |
| **Add README** | ❌ No (we already have one) |
| **Add .gitignore** | ❌ No (we already have one) |
| **Choose a license** | ❌ No (we already have LICENSE file) |

## Topics / Tags

Add these topics on the repo main page (click the gear icon next to "About"):

```
tokenbreak, adversarial-machine-learning, nlp-security, vulnerability-scanner, tokenizer-security, bpe, wordpiece, unigram, huggingface, transformers, llm-security, model-auditing, red-teaming, ai-safety, python-cli, agpl
```

## Social Preview Image

GitHub → Settings → Social preview

Recommended: generate a 1280×640px image with text:
- **Title**: TokenBreak Scanner
- **Subtitle**: Detect adversarial tokenizer vulnerabilities in NLP models
- **Colors**: dark background (#0d1117), green accent (#2ea043), red accent (#da3633)

## Secrets Setup (Required for PyPI Publishing)

Navigate to: **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|---|---|
| `PYPI_API_TOKEN` | Your PyPI API token (from https://pypi.org/manage/account/token/) |

### How to create a PyPI token

1. Register/Login at https://pypi.org
2. Go to **Account settings → API tokens → Add API token**
3. Name: `tokenbreak-scanner-github-actions`
4. Scope: **Entire account** (or scoped to `tokenbreak-scanner` project)
5. Copy the token and paste it as the GitHub secret value

## Release Workflow

After pushing to GitHub:

```bash
# Normal development
make changes
git add .
git commit -m "feat: something"
git push origin main

# When ready to release:
git tag v0.1.0
git push origin v0.1.0
```

The `publish.yml` workflow automatically:
1. Runs tests
2. Builds wheel + sdist
3. Uploads to PyPI
4. Creates a GitHub Release with asset downloads

## Branch Protection (Recommended)

Settings → Branches → Add rule:
- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging
- Status checks: `test (3.9)`, `test (3.10)`, `test (3.11)`, `test (3.12)`
- ✅ Require branches to be up to date before merging
