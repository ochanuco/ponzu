# Security and Privacy Policy

## Repository Policy

The Ponzu repository is intended to be safe for public visibility.

Only generic source code, documentation, examples, and non-sensitive test data may be committed.

## Never Commit

- `.env` files
- API keys
- OAuth credentials
- Access tokens
- Private keys
- Audio recordings
- Speech transcripts
- Conversation history
- Long-term memory
- Personal prompts
- Email content
- Calendar content
- Home network addresses
- Device identifiers
- Local database files
- Logs containing user content
- Wake-word training recordings

## Recommended `.gitignore`

```gitignore
.env
.env.*
!.env.example

config.local.*
*.db
*.sqlite
*.sqlite3

audio/
recordings/
logs/
models/
data/private/
runtime/

*.wav
*.mp3
*.m4a
*.pcm

.DS_Store
```

## GitHub Settings

Recommended for the public repository:

- Secret scanning
- Push protection
- Dependabot alerts
- Dependabot security updates
- Code scanning
- Read-only default GitHub Actions permissions
- Protected default branch or repository ruleset

## Incident Response

If a secret is committed:

1. Revoke or rotate the secret immediately.
2. Remove it from the current tree.
3. Rewrite Git history where appropriate.
4. Assume forks, caches, and clones may retain the original value.
5. Review logs for unauthorized use.

Deleting a commit does not make an exposed secret safe again.
