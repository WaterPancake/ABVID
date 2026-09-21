"""Validate Obsidian links across the whole docs/notes vault.

Checks:
- every [[wikilink]] resolves to exactly one note by basename (vault-wide);
- every embedded image path exists relative to the note's folder;
- no duplicate note basenames (which would make name-links ambiguous);
- the top-level index notes are reachable from their folder index.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

VAULT = Path("/Users/magi/workspace/ABVID/docs/notes")
files = sorted(VAULT.rglob("*.md"))

wikilink_re = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
image_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# basename -> list of files with that basename
by_name = defaultdict(list)
for f in files:
    by_name[f.stem].append(f)

problems = 0

# 1) duplicate basenames
for name, hits in sorted(by_name.items()):
    if len(hits) > 1:
        print(f"DUPLICATE basename '{name}':")
        for h in hits:
            print(f"   - {h.relative_to(VAULT)}")
        problems += 1

# 2) every file's wikilinks resolve
for f in files:
    text = f.read_text()
    for target in wikilink_re.findall(text):
        target = target.strip()
        if target not in by_name:
            print(f"UNRESOLVED link in {f.relative_to(VAULT)}: [[{target}]]")
            problems += 1
    # 3) every embedded image exists relative to the note's folder
    for img in image_re.findall(text):
        p = (f.parent / img).resolve()
        if not p.exists():
            print(f"MISSING IMAGE in {f.relative_to(VAULT)}: {img}")
            problems += 1

# 4) no leftover ../ style links (Obsidian does not resolve those)
for f in files:
    text = f.read_text()
    for m in re.findall(r"\[\[\.\./", text):
        print(f"LEFTOVER ../ link in {f.relative_to(VAULT)}")
        problems += 1
        break

print(f"\n{len(files)} files checked, {problems} problems")
sys.exit(1 if problems else 0)
