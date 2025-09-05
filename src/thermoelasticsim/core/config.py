"""配置加载模块（MVP）

提供轻量的 YAML 配置加载与工具函数，满足教学与示例场景：

- 递归合并多份 YAML（后者覆盖前者）
- 点路径访问（如 ``md.timestep``）
- 统一设置随机种子（numpy/random）
- 基于模板创建输出目录并保存配置快照

Notes
-----
本模块刻意不引入 Hydra，以保持依赖简单与行为透明；后续若需要
更复杂的组合与命令行覆盖，可在不破坏兼容性的前提下扩展。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _get_by_path(d: dict, path: str, default: Any = None) -> Any:
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class _Resolved:
    data: dict
    sources: list[str]


class ConfigManager:
    """配置管理器

    加载一组 YAML 配置文件并进行递归合并，提供点路径访问与常用工具。

    Parameters
    ----------
    files : Iterable[str] | None, optional
        需要加载的 YAML 文件列表，后者覆盖前者；若为 ``None`` 则只构建空配置。

    Attributes
    ----------
    data : dict
        合并后的配置数据（只读属性 ``.data`` 暴露内部字典）。
    """

    def __init__(self, files: Iterable[str] | None = None) -> None:
        self._resolved = self._load_all(files)

    # --------- 加载与解析 ---------
    def _load_all(self, files: Iterable[str] | None) -> _Resolved:
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "config" / "default.yaml"
        data: dict[str, Any] = {}
        sources: list[str] = []
        if default_path.exists():
            with open(default_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            sources.append(str(default_path))
        # 用户覆盖
        if files:
            for p in files:
                path = Path(p)
                if not path.exists():
                    continue
                with open(path, encoding="utf-8") as f:
                    ov = yaml.safe_load(f) or {}
                data = _deep_update(data, ov)
                sources.append(str(path))
        return _Resolved(data=data, sources=sources)

    @property
    def data(self) -> dict:
        """获取合并后的配置数据字典。"""
        return self._resolved.data

    # --------- 访问接口 ---------
    def get(self, path: str, default: Any | None = None) -> Any:
        """获取配置值（点路径）

        使用 ``a.b.c`` 形式访问嵌套字典，若不存在则返回 ``default``。

        Parameters
        ----------
        path : str
            点路径键名，例如 ``"md.timestep"``。
        default : Any, optional
            当键不存在时返回的默认值。

        Returns
        -------
        Any
            对应的配置值或 ``default``。
        """
        return _get_by_path(self._resolved.data, path, default)

    # --------- 实用工具 ---------
    def set_global_seed(self, seed: int | None = None) -> int:
        """统一设置随机种子

        同时设置 ``numpy.random`` 与 Python ``random`` 的种子，以增强可复现性。

        Parameters
        ----------
        seed : int | None, optional
            若为 ``None``，则读取 ``rng.global_seed``（默认 42）。

        Returns
        -------
        int
            实际使用的种子值。
        """
        if seed is None:
            seed = int(self.get("rng.global_seed", 42))
        try:
            import numpy as _np

            _np.random.seed(seed)
        except Exception:
            pass
        _random.seed(seed)
        return seed

    def make_output_dir(self, name: str | None = None) -> str:
        """创建输出目录

        依据模板 ``run.output_dir`` 创建目录，支持 ``{name}`` 与 ``{timestamp}`` 占位符。
        若未配置，默认使用 ``examples/logs/{name}_{timestamp}``。

        Parameters
        ----------
        name : str | None, optional
            运行名；若为 ``None``，则读取 ``run.name``（默认 ``"run"``）。

        Returns
        -------
        str
            创建的输出目录路径。
        """
        # 默认与项目示例保持一致放在 examples/logs/
        pattern = str(self.get("run.output_dir", "examples/logs/{name}_{timestamp}"))
        name = name or str(self.get("run.name", "run"))
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = pattern.format(name=name, timestamp=ts)
        os.makedirs(out, exist_ok=True)
        return out

    def snapshot(self, output_dir: str) -> None:
        """保存配置快照

        在输出目录写入 ``resolved_config.yaml`` 与轻量 ``manifest.json``，帮助记录
        本次运行所使用的配置来源与时间戳，便于复现与审计。

        Parameters
        ----------
        output_dir : str
            输出目录路径。
        """
        try:
            path = Path(output_dir) / "resolved_config.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self._resolved.data, f, allow_unicode=True, sort_keys=True
                )
            # 轻量 manifest
            manifest = {
                "timestamp": _dt.datetime.now().isoformat(),
                "sources": self._resolved.sources,
            }
            with open(Path(output_dir) / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception:
            # 快照失败不阻断主流程
            pass
