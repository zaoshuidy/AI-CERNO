# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities privately via email to the project
maintainers. Do **not** open public issues for security-sensitive bugs.

Expected response time: within 7 days.

## API Key Handling

**Never commit API keys to this repository.**

- Use a `.env` file (see `.env.example` for the template).
- Ensure `.env` is ignored by Git (already listed in `.gitignore`).
- Rotate keys immediately if accidentally committed.
- CI jobs run against mocked endpoints; no real keys are required for tests.

## What This Project Does NOT Guarantee

`veritas-core` performs retrieval-augmented fact verification using external
LLMs and search APIs. It does **not** guarantee 100% accuracy. Output should
be treated as an assistive signal, not a definitive legal or medical
judgment. See the [Disclaimer](#disclaimer) section in the README.
