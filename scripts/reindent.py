"""Heuristic re-indenter for files whose multi-space indentation was
collapsed to single-space. Uses ``:`` opening blocks and ``)/]/}`` /
``return``/``raise``/``pass``/``break``/``continue``/``elif``/``else``/
``except``/``finally`` keywords plus current-line bracket balance to
restore 4-space indentation. Best-effort, not a parser.
"""

from __future__ import annotations

import io
import sys
import tokenize
from pathlib import Path


def _trim(s: str) -> str:
    """Remove the leading single space the regex left in place, plus any
    other leading whitespace, since indentation must be reconstructed."""
    return s.lstrip(" \t")


def reindent(src: str) -> str:
    """Reconstruct 4-space indentation from a damaged source file.

    Strategy: walk lines, maintain a stack of expected indent depths.
    Increase depth after a line ending with ``:`` (block opener).
    Decrease depth based on dedent-keywords or empty lines that close a
    suite. Bracket balance prevents counting ``:`` inside type hints
    (``dict[str, int]:``) twice — only counts the trailing ``:`` if
    bracket balance is zero at end of line.
    """
    out: list[str] = []
    indent = 0
    pending_open = False
    bracket = 0  # net (a ([{ count - )]} count
    block_keywords_close = {"return", "raise", "pass", "break", "continue"}
    block_keywords_continue = {"elif", "else", "except", "finally"}

    for raw in src.splitlines():
        body = _trim(raw)

        # Pure blank lines: keep blank.
        if body == "":
            out.append("")
            continue

        # Comments / strings: keep at current indent.
        # Adjust dedent if the body itself is a closing-bracket-only line.
        first_token = body.split(maxsplit=1)[0] if body else ""

        # If the body starts with ``elif``/``else:``/``except``/``finally``,
        # outdent by one before emitting.
        local_indent = indent
        if first_token.rstrip(":") in block_keywords_continue and indent > 0:
            local_indent = indent - 1

        # If the body is just a closing bracket / ``])` etc., outdent.
        if body in {")", "]", "}", "):", "],", "})", "));", "],"}:
            local_indent = max(indent - 1, 0)

        out.append("    " * local_indent + body)

        # Update bracket balance from this line (ignoring strings/comments
        # is handled by tokenize for accuracy; quick heuristic: count
        # outside string literals via tokenize tokens).
        try:
            toks = list(
                tokenize.generate_tokens(io.StringIO(body).readline)
            )
        except (tokenize.TokenizeError, IndentationError, SyntaxError):
            toks = []

        opens = sum(1 for t in toks if t.type == tokenize.OP and t.string in "([{")
        closes = sum(1 for t in toks if t.type == tokenize.OP and t.string in ")]}")
        bracket += opens - closes
        if bracket < 0:
            bracket = 0

        # Decide whether the next line is more deeply indented.
        stripped = body.rstrip()
        # Only treat trailing colon as block-opener when bracket-balanced.
        if bracket == 0 and stripped.endswith(":") and not stripped.endswith("\\:"):
            indent = local_indent + 1
        else:
            indent = local_indent

        # If first_token is a "close-block" keyword (return / raise /
        # pass / break / continue), the next statement should outdent
        # back to the parent level — emulate by reducing depth by 1 if
        # we are inside a function body.
        if first_token in block_keywords_close and bracket == 0 and indent > 0:
            indent = max(indent - 1, 0)

        del pending_open  # unused but kept for clarity
        pending_open = False  # reset

    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python reindent.py <file.py> [file2.py ...]", file=sys.stderr)
        sys.exit(2)
    for arg in sys.argv[1:]:
        p = Path(arg)
        src = p.read_text()
        new = reindent(src)
        p.write_text(new)


if __name__ == "__main__":
    main()
