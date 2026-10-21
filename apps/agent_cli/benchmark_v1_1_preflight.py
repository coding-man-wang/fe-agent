from pathlib import Path

from agent_core.benchmark.workspace_dependencies import (
    NodeModulesWorkspaceProvisioner,
    WorkspaceDependencyError,
)


def main() -> None:
    benchmark_repository = Path(r"D:\project\fe-agent-benchmark-web")
    provisioner = NodeModulesWorkspaceProvisioner(
        dependency_source_root=benchmark_repository,
    )

    try:
        provisioner.require_source_ready()
    except WorkspaceDependencyError as exc:
        print("=== Full Execution Environment Preflight FAILED ===")
        print(str(exc))
        raise SystemExit(1) from exc

    print("=== Full Execution Environment Preflight PASSED ===")
    print(f"repository={benchmark_repository}")
    print("node/npm=ready")
    print(
        "required_bins="
        + ", ".join(provisioner.required_bins)
    )


if __name__ == "__main__":
    main()
