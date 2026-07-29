"""Local LLM adapters (DESIGN section 4.5, ADR-005).

Only one backend exists so far (`ponzu.llm.ollama`); this module stays a thin
package marker rather than a registry so adding a second backend doesn't force
an abstraction decision before it's needed.
"""

from __future__ import annotations
