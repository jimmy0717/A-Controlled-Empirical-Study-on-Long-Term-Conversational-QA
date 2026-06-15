"""Safe prompt-template filling.

Our prompt templates intentionally contain *literal* JSON braces (e.g.
``{"op": "ADD"}``) as few-shot output specifications. Using Python's
``str.format`` on such templates raises ``KeyError`` / ``ValueError``
because it treats every ``{...}`` as a replacement field. We therefore
fill placeholders by **explicit literal substitution**, which leaves all
other braces untouched.

Placeholders are written as ``{name}`` in the template; only the exact
keys passed to :func:`fill_template` are replaced.
"""
from __future__ import annotations


def fill_template(template: str, **kwargs) -> str:
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value))
    return out
