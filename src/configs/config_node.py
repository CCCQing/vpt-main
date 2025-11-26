#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

from fvcore.common.config import CfgNode as _CfgNode
from ..utils.file_io import PathManager


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    support manifold path
    """

    @classmethod
    def _open_cfg(cls, filename):
        # Explicitly use UTF-8 to avoid locale-dependent decoding errors when
        # reading YAML configs on platforms with non-UTF-8 defaults (e.g.,
        # Windows CP936/GBK).
        return PathManager.open(filename, "r", encoding="utf-8")

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)
