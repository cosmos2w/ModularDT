"""Keep the public README's equations safe for GitHub Markdown math."""
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _math_fragments(text):
    """Extract GitHub's code-protected inline math and standalone displays."""
    fragments = []
    in_code = False
    display = None
    for line in text.splitlines():
        if line.startswith("```"):
            assert display is None, "code fence inside a math display"
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.strip() == "$$":
            if display is None:
                display = []
            else:
                fragments.append("\n".join(display))
                display = None
            continue
        if display is not None:
            display.append(line)
            continue
        inline = re.findall(r"\$`([^`\n]+)`\$", line)
        fragments.extend(inline)
        remainder = re.sub(r"\$`([^`\n]+)`\$", "", line)
        assert "$" not in remainder, f"unprotected or unmatched math delimiter: {line}"
    assert not in_code, "unclosed code fence"
    assert display is None, "unclosed math display"
    return fragments


def test_readme_math_uses_github_safe_delimiters_and_commands():
    fragments = _math_fragments(README.read_text())
    assert fragments, "README should retain its model equations"
    for fragment in fragments:
        assert r"\operatorname" not in fragment
        assert r"\!" not in fragment
        assert "fallbackrawblock" not in fragment.lower()
        # Escaped set delimiters are TeX symbols, not brace groups.
        groups = re.sub(r"\\[{}]", "", fragment)
        depth = 0
        for character in groups:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                assert depth >= 0, fragment
        assert depth == 0, fragment
        assert re.findall(r"\\begin\{([^}]+)\}", fragment) == re.findall(
            r"\\end\{([^}]+)\}", fragment
        )
