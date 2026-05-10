"""Shared utilities for scripts_v2: YAML config loading + quiet Logger wrapper.

Design:
- load_config(path) -> dict; supports nested dicts. Numbers like 1_500_000 work.
- merge_with_args(config, args, key_map) -> Namespace; CLI overrides config.
- silence_logger_stdout(): monkey-patch draco.utils.logger.Logger.dump so it still
  writes to .txt and .jsonl but does not print to stdout. Returns the
  monkey-patched original dump so tqdm can hook it for progress display.
- patch_logger_for_tqdm(pbar, postfix_keys): wraps Logger.dump to also update tqdm.
"""
from __future__ import annotations
import argparse
import io
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file. Returns {} if path is empty/None."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a dict, got {type(cfg)}")
    return cfg


def cfg_get(cfg: Dict[str, Any], key: str, default=None):
    """Read key from config; supports both 'foo_bar' and 'foo-bar' (CLI style)."""
    if key in cfg:
        return cfg[key]
    if key.replace("_", "-") in cfg:
        return cfg[key.replace("_", "-")]
    if key.replace("-", "_") in cfg:
        return cfg[key.replace("-", "_")]
    return default


def merge_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Apply config values to args ONLY where args still holds the parser default
    (i.e., user did not pass that flag on CLI). This makes CLI > config > parser-default.

    Detects "user-passed" by comparing args to a dict of parser defaults that the
    caller must precompute (since argparse doesn't expose that distinction natively).
    """
    raise NotImplementedError("Use merge_args_with_defaults instead.")


def merge_args_with_defaults(
    args: argparse.Namespace,
    parser_defaults: Dict[str, Any],
    cfg: Dict[str, Any],
) -> argparse.Namespace:
    """For each attribute in args: if it equals the parser default AND a config
    value exists, use the config value. Otherwise keep args."""
    for key, default in parser_defaults.items():
        cur = getattr(args, key, default)
        if cur == default:
            cfg_val = cfg_get(cfg, key, None)
            if cfg_val is not None:
                setattr(args, key, cfg_val)
    return args


@contextmanager
def silence_stdout():
    """Context manager that redirects stdout to a string buffer."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def make_quiet_dump(pbar=None, postfix_keys=("EpRet/mean", "EpCost/mean", "Lambda/mean")):
    """Returns a replacement Logger.dump method that:
    - silences print() inside dump() (still writes .txt and .jsonl)
    - optionally updates a tqdm pbar with epoch postfix.
    Use as: Logger.dump = make_quiet_dump(pbar)
    """
    from draco.utils.logger import Logger
    orig_dump = Logger.dump

    def quiet_dump(self, epoch, extra=None):
        with silence_stdout():
            agg = orig_dump(self, epoch, extra=extra)
        if pbar is not None:
            postfix = {}
            for k in postfix_keys:
                if k in agg:
                    short = k.split("/")[0]
                    postfix[short] = f"{agg[k]:.3f}" if abs(agg[k]) < 1000 else f"{agg[k]:.1e}"
            pbar.set_postfix(postfix)
            pbar.update(1)
        return agg

    return quiet_dump


__all__ = [
    "load_config", "cfg_get", "merge_args_with_defaults",
    "silence_stdout", "make_quiet_dump",
]
