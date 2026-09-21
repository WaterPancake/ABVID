"""Cross-check every Obsidian wikilink and image embed in the condensed guide."""
import re
import sys
from pathlib import Path

GUIDE = Path("/Users/magi/workspace/ABVID/docs/notes/condensed-guide")
files = sorted(GUIDE.glob("*.md"))

# Collect note names (Obsidian wikilinks match by filename, spaces included).
note_names = {p.stem for p in GUIDE.glob("*.md")}

wikilink_re = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
image_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

problems = 0
for f in files:
    text = f.read_text()
    for target in wikilink_re.findall(text):
        target = target.strip()
        if target not in note_names:
            print(f"BROKEN WIKILINK in {f.name}: [[{target}]]")
            problems += 1
    for img in image_re.findall(text):
        # image paths are relative to the note's folder
        p = (GUIDE / img).resolve()
        if not p.exists():
            print(f"MISSING IMAGE in {f.name}: {img}")
            problems += 1

# Check every note is reachable from README via a wikilink.
readme = (GUIDE / "README.md").read_text()
unreachable = note_names - {"README"} - set(wikilink_re.findall(readme))
for n in sorted(unreachable):
    print(f"UNREACHABLE note (not linked from README): {n}")
    problems += 1

print(f"\n{len(files)} files, {problems} problems")
sys.exit(1 if problems else 0)
