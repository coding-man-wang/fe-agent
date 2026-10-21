from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.benchmark.workspace_dependencies import (
    NodeModulesWorkspaceProvisioner,
    WorkspaceDependencyError,
)


def _make_bin(node_modules: Path, name: str) -> None:
    bin_root = node_modules / ".bin"
    bin_root.mkdir(parents=True, exist_ok=True)
    (bin_root / name).write_text("#!/bin/sh\n", encoding="utf-8")


def test_preflight_rejects_existing_node_modules_without_verification_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "repo"
    node_modules = source / "node_modules"
    node_modules.mkdir(parents=True)
    (node_modules / ".bin").mkdir()

    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")

    provisioner = NodeModulesWorkspaceProvisioner(
        dependency_source_root=source,
    )

    with pytest.raises(WorkspaceDependencyError) as exc_info:
        provisioner.require_source_ready()

    message = str(exc_info.value)
    assert "eslint" in message
    assert "vue-tsc" in message
    assert "vitest" in message
    assert "vite" in message
    assert "npm ci" in message


def test_preflight_accepts_complete_dev_dependency_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "repo"
    node_modules = source / "node_modules"
    node_modules.mkdir(parents=True)

    for name in NodeModulesWorkspaceProvisioner.DEFAULT_REQUIRED_BINS:
        _make_bin(node_modules, name)

    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")

    provisioner = NodeModulesWorkspaceProvisioner(
        dependency_source_root=source,
    )
    provisioner.require_source_ready()
    assert provisioner.missing_bins() == []


def test_provision_validates_target_after_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "repo"
    node_modules = source / "node_modules"
    node_modules.mkdir(parents=True)
    for name in NodeModulesWorkspaceProvisioner.DEFAULT_REQUIRED_BINS:
        _make_bin(node_modules, name)

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")

    provisioner = NodeModulesWorkspaceProvisioner(
        dependency_source_root=source,
    )
    provisioner.provision(workspace)

    assert (workspace / "node_modules").exists()
    assert provisioner.missing_bins(workspace / "node_modules") == []
