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

# Anchored with a leading slash on purpose. An unanchored `audio/` matches a
# directory of that name at ANY depth, including the source package
# `src/ponzu/audio/`, which is then silently never committed -- the clone
# installs and fails at import. These entries exist to catch runtime data
# accidentally written into the checkout; per ADR-006 it belongs in the user
# data directory, so matching only the repository root is what is wanted.
/audio/
/recordings/
/logs/
/models/
/runtime/
/data/private/

*.wav
*.mp3
*.m4a
*.pcm

.DS_Store
```

Directory patterns for runtime data must stay anchored. Source directories are
named after what they do, so a bare `logs/` or `models/` will eventually
collide with one.

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
