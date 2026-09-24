from agent_core.codebase.file_index import (
    RepositoryFileIndexer,
)

indexer = RepositoryFileIndexer()

index = indexer.build(r"D:\project\v3-admin-vite-main")

print(
    "total_files =",
    index.total_files,
)

print(
    "indexable_files =",
    index.indexable_files,
)

print(
    "category_counts =",
    index.category_counts,
)

for file in index.files[:30]:
    print(
        file.relative_path,
        file.category,
        file.is_indexable,
    )
