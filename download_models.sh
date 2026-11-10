#!/bin/bash
PY=${PY:-python}
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
for m in ["microsoft/Phi-3.5-mini-instruct", "meta-llama/Llama-3.2-3B-Instruct"]:
    print(f"=== {m} ===", flush=True)
    try:
        p = snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
        print("  ->", p, flush=True)
    except Exception as e:
        print("  FAILED:", type(e).__name__, str(e)[:300], flush=True)
PYEOF
echo "=== DOWNLOADS DONE ==="
