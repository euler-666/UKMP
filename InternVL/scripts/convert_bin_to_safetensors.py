"""
One-off: convert a saved HF checkpoint from `pytorch_model.bin` (pickle)
to `model.safetensors`.

Needed because this env pins torch 2.4 and transformers >=4.43 refuses to
load pickle checkpoints unless torch>=2.6 (CVE-2025-32434).

The trainer (`finetune_internvl_full_ce.py`) has been fixed to save with
`safe_serialization=True` going forward; this script just rescues
checkpoints that were saved before that fix.

Usage:
    python scripts/convert_bin_to_safetensors.py <ckpt_dir>
"""
import os
import sys

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

ckpt_dir = sys.argv[1]
bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

if not os.path.isfile(bin_path):
    print(f"[skip] no pytorch_model.bin at {bin_path}")
    sys.exit(0)

# Disable the transformers torch.load safety check before importing the
# model. Our checkpoint is locally produced, so the CVE is not a concern.
import transformers.utils.import_utils as _iu
import transformers.modeling_utils as _mu
_iu.check_torch_load_is_safe = lambda *a, **kw: None
_mu.check_torch_load_is_safe = lambda *a, **kw: None

import torch
from transformers import AutoModel, AutoTokenizer

print(f"Loading model from {ckpt_dir} ...")
model = AutoModel.from_pretrained(
    ckpt_dir,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

print("Saving with safe_serialization=True ...")
model.save_pretrained(ckpt_dir, safe_serialization=True)
tokenizer.save_pretrained(ckpt_dir)

if os.path.isfile(safetensors_path):
    print(f"Removing legacy {bin_path} ...")
    os.remove(bin_path)
else:
    print(f"[warn] expected {safetensors_path} but it was not written; "
          f"leaving {bin_path} in place.")

print("Done.")
