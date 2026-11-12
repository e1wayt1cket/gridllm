"""Regenerate hash_manifest.json for the M1 deliverables.

Records only the files this audit created or modified. Nothing under src/,
config/, tests/, policies/, runs/ or snapshots/ appears because none was
touched; a missing entry is the evidence of that, not an omission.
"""
import hashlib
import io
import json
import os

M1_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(M1_DIR, "..", "..", ".."))


def collect():
    targets = []
    for name in sorted(os.listdir(os.path.join(ROOT, "docs", "project_context"))):
        path = os.path.join("docs", "project_context", name)
        if os.path.isfile(os.path.join(ROOT, path)):
            targets.append(path)
    targets.append(os.path.join("tasks", "M1_RULE_AUDIT.md"))
    for name in sorted(os.listdir(M1_DIR)):
        if name in ("hash_manifest.json", os.path.basename(__file__)):
            continue
        targets.append(os.path.join(
            "reports", "market_mechanism_audit",
            "M1_jiangsu_da_rule_audit_2026-11-12", name))
    targets.append("CLAUDE.md")
    return targets


def main():
    entries = []
    for rel in collect():
        full = os.path.join(ROOT, rel)
        digest = hashlib.sha256(open(full, "rb").read()).hexdigest()
        entries.append({
            "path": rel.replace(os.sep, "/"),
            "sha256": digest,
            "bytes": os.path.getsize(full),
        })
    doc = {
        "manifest_id": "m1_jiangsu_da_rule_audit_2026-11-12",
        "generated": "2026-11-12",
        "note": ("Hashes of the files this audit created or modified. No file "
                 "under src/, config/, tests/, policies/, runs/ or snapshots/ "
                 "is listed because none was touched."),
        "file_count": len(entries),
        "files": entries,
    }
    out = os.path.join(M1_DIR, "hash_manifest.json")
    with io.open(out, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print("wrote %s with %d entries" % (out, len(entries)))


if __name__ == "__main__":
    main()
