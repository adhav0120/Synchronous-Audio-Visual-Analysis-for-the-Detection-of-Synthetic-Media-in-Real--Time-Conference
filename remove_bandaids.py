"""
remove_bandaids.py
==================
Run this AFTER retraining is complete and you've confirmed the calibration
report shows the new model scores genuine speech correctly.

What it does:
  1. Removes output[0][1] -= fake_logit_penalty  from predict_audio_from_chunk()
  2. Removes score *= system_audio_penalty_multiplier from system_audio_capture_thread()
  3. Removes those two keys from config.json
  4. Updates README.md to move item 3 to Completed

Run from project root:
    .venv\\Scripts\\python.exe remove_bandaids.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC          = PROJECT_ROOT / "src" / "desktop_live_capture.py"
CFG          = PROJECT_ROOT / "config.json"
README       = PROJECT_ROOT / "README.md"

# ── 1. Patch desktop_live_capture.py ───────────────────────────────────────
print("Patching desktop_live_capture.py…")
code = SRC.read_text(encoding="utf-8")

# Remove the logit penalty line inside predict_audio_from_chunk
LOGIT_PENALTY_LINE = (
    "        # Apply a mathematical penalty to the 'Fake' logit to reduce artificial sensitivity\n"
    "        output[0][1] -= AUDIO_CONFIG.get('fake_logit_penalty', 2.0)\n"
)
LOGIT_PENALTY_REPLACEMENT = (
    "        # No post-hoc logit adjustment needed — model is natively calibrated\n"
)

if LOGIT_PENALTY_LINE in code:
    code = code.replace(LOGIT_PENALTY_LINE, LOGIT_PENALTY_REPLACEMENT)
    print("  ✓ Removed fake_logit_penalty subtraction")
else:
    print("  ! fake_logit_penalty line not found (may already be removed)")

# Remove the score multiplier line inside system_audio_capture_thread
MULTIPLIER_LINE = (
    "            # Artificially cap the system audio score to 60% of its raw output\n"
    "            score = score * AUDIO_CONFIG.get('system_audio_penalty_multiplier', 0.60)\n"
)
MULTIPLIER_REPLACEMENT = (
    "            # Score used directly — model is natively calibrated\n"
)

if MULTIPLIER_LINE in code:
    code = code.replace(MULTIPLIER_LINE, MULTIPLIER_REPLACEMENT)
    print("  ✓ Removed system_audio_penalty_multiplier scaling")
else:
    print("  ! system_audio_penalty_multiplier line not found (may already be removed)")

SRC.write_text(code, encoding="utf-8")
print("  ✓ desktop_live_capture.py saved")

# ── 2. Patch config.json ────────────────────────────────────────────────────
print("\nPatching config.json…")
with open(CFG, "r") as f:
    cfg = json.load(f)

audio_cfg = cfg.get("audio", {})
removed = []
for key in ["fake_logit_penalty", "system_audio_penalty_multiplier"]:
    if key in audio_cfg:
        del audio_cfg[key]
        removed.append(key)

cfg["audio"] = audio_cfg
with open(CFG, "w") as f:
    json.dump(cfg, f, indent=4)

if removed:
    print(f"  ✓ Removed keys: {', '.join(removed)}")
else:
    print("  ! Keys already absent from config.json")

# ── 3. Update README.md ─────────────────────────────────────────────────────
print("\nUpdating README.md…")
readme = README.read_text(encoding="utf-8")

OLD_REMAINING = (
    "### 3. Core AI Retraining\n"
    "*   **True Model Calibration:** Right now, there is a literal mathematical subtraction "
    "(`score -= 2.0`) being forced onto the AI model outputs to reduce false positive rates. "
    "This works, but is technically a band-aid limit. **Improvement:** Gather a comprehensive "
    "dataset of \"negative examples\" (non-deepfakes) and completely retrain the actual `.pth` "
    "PyTorch model to naturally understand these boundaries without needing hardcoded "
    "post-processing limits!"
)

NEW_COMPLETED = (
    "*   ✅ **True Model Calibration:** Fine-tuned the `transformer_voice_detector.pth` "
    "Transformer on a balanced dataset (ASVspoof 2019 LA + LibriSpeech dev-clean) using "
    "WeightedRandomSampler and label-smoothed CrossEntropyLoss. The model now naturally "
    "scores genuine speech as bonafide — no more hardcoded `score -= 2.0` or `0.60×` "
    "penalty multiplier post-processing."
)

if "### 3. Core AI Retraining" in readme:
    # Add to completed list
    readme = readme.replace(
        "*   ✅ **Type Hinting & Testing:**",
        "*   ✅ **Type Hinting & Testing:**"  # keep this line
    )
    # Remove the Remaining section 3 entirely
    readme = re.sub(
        r"\n### 3\. Core AI Retraining\n.*",
        "",
        readme,
        flags=re.DOTALL,
    )
    # Append to completed list (before the --- divider)
    readme = readme.replace(
        "*   ✅ **Proper Logging System:**",
        f"*   ✅ **Proper Logging System:**",
    )
    # Insert new completed item after logging entry
    readme = readme.replace(
        "\n---\n",
        f"\n{NEW_COMPLETED}\n\n---\n",
        1,  # only first occurrence
    )
    README.write_text(readme, encoding="utf-8")
    print("  ✓ README.md updated — item moved to Completed")
else:
    print("  ! README.md section not found (may already be updated)")

print()
print("=" * 60)
print("  ✅ Band-aids removed successfully!")
print("  The application now uses the natively calibrated model.")
print("=" * 60)
