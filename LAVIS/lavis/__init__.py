"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys

from omegaconf import OmegaConf

from lavis.common.registry import registry

# Wildcard imports for LAVIS registry (BLIP-2 models, datasets, etc.).
# Wrapped in try/except because newer transformers versions removed APIs
# that some LAVIS model definitions depend on (e.g. apply_chunking_to_forward).
# The InternVL pruning pipeline does not require these imports.
try:
    from lavis.datasets.builders import *
    from lavis.models import *
    from lavis.processors import *
    from lavis.tasks import *
except ImportError:
    pass


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
