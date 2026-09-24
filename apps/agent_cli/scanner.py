from agent_core.codebase.scanner import (
    RepositoryScanner,
)

scanner = RepositoryScanner()

profile = scanner.scan(r"D:\project\v3-admin-vite-main")

print(profile.model_dump_json(indent=2))
