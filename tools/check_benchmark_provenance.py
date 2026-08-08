#!/usr/bin/env python3
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
index=json.loads((ROOT/'benchmarks/INDEX.json').read_text())
errors=[]
for item in index['artifacts']:
    artifact=ROOT/'benchmarks'/item['path']
    if not artifact.exists():
        errors.append(f"missing artifact: {item['path']}")
        continue
    status=item['status']
    gen=item.get('generator')
    if status.startswith('current'):
        if not gen:
            errors.append(f"current artifact lacks generator: {item['path']}")
        elif not (ROOT/gen).exists():
            errors.append(f"current artifact generator missing: {item['path']} -> {gen}")
    if status.startswith('superseded') and 'run2_' not in item['path'] and 'run3_' not in item['path']:
        errors.append(f"unexpected superseded artifact naming: {item['path']}")
if errors:
    raise SystemExit('\n'.join(errors))
print(f"provenance index OK: {len(index['artifacts'])} artifacts registered")
