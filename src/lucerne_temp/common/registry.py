"""A model registry small enough to read in one sitting.

registry/registry.json holds metadata only: versions, their config and metrics, and the
alias champion -> version. Model binaries are never committed: they are rebuilt from the
committed snapshot by the training pipeline and cached in artifacts/.

In a real project MLflow plays this role (see README: "From the file registry to MLflow").
"""

import json
from pathlib import Path
from typing import Any, Protocol

import joblib

from lucerne_temp.common import config


class ModelStore(Protocol):
    """What the pipelines need from a registry. Swap the backend, keep the callers."""

    def version_of(self, alias: str) -> str: ...
    def set_alias(self, alias: str, version: str) -> None: ...
    def load(self, version: str) -> Any: ...


class FileModelStore:
    def __init__(self, registry_path: Path | None = None, artifact_dir: Path | None = None):
        self.registry_path = registry_path or config.REGISTRY_PATH
        self.artifact_dir = artifact_dir or config.ARTIFACT_DIR

    def read(self) -> dict:
        return json.loads(self.registry_path.read_text())

    def _write(self, registry: dict) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2) + "\n")

    def version_of(self, alias: str = config.ALIAS) -> str:
        return self.read()["aliases"][alias]

    def set_alias(self, alias: str, version: str) -> None:
        registry = self.read()
        if version not in registry["versions"]:
            raise KeyError(f"unknown version {version!r}")
        registry["aliases"][alias] = version
        self._write(registry)

    def version_config(self, version: str) -> dict:
        return self.read()["versions"][version]

    def record(self, version: str, **info: Any) -> None:
        """Store training window and metrics next to the version's config."""
        registry = self.read()
        registry["versions"][version].update(info)
        self._write(registry)

    def artifact_path(self, version: str) -> Path:
        return self.artifact_dir / f"{config.MODEL_NAME}-v{version}.joblib"

    def save(self, version: str, model: Any) -> Path:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.artifact_path(version))
        return self.artifact_path(version)

    def load(self, version: str) -> Any:
        # joblib is pickle: only load artifacts you built yourself
        return joblib.load(self.artifact_path(version))
