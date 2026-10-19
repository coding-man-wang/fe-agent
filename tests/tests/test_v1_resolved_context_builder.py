from agent_core.artifacts.resolver import TaskArtifactResolver
from agent_core.artifacts.store import JsonTaskArtifactStore
from agent_core.change_surface.context_builder import ResolvedTaskContextBuilder
from agent_core.change_surface.model import ResolvedChangeSurface
from agent_core.codebase.model import (
    FileRecord, HybridRelevantFile, HybridRelevantFileSelection,
    RepositoryFileIndex, RepositoryProfile,
)
from agent_core.sdd.model import RequirementSpec, TaskDefinition, TechnicalSpec


def test_surface_can_pull_file_missing_from_original_selection(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.ts").write_text("export const a = 1", encoding="utf-8")
    (tmp_path / "src/b.ts").write_text("export const b = 2", encoding="utf-8")
    records=[]
    for path in ["src/a.ts","src/b.ts"]:
        f=tmp_path/path
        records.append(FileRecord(relative_path=path,name=f.name,suffix=".ts",size_bytes=f.stat().st_size,category="source",language="TypeScript",is_indexable=True))
    index=RepositoryFileIndex(root_path=str(tmp_path),total_files=2,indexable_files=2,files=records,category_counts={"source":2})
    selected=HybridRelevantFile(relative_path="src/a.ts",category="source",language="TypeScript",size_bytes=10,retrieval_score=1,retrieval_rank=1,policy_score=0,selection_score=1,rerank_rank=1,rerank_score=.9)
    selection=HybridRelevantFileSelection(candidate_count=1,eligible_count=1,selected_count=1,total_selected_bytes=10,selected_files=[selected])
    surface=ResolvedChangeSurface(task_id="TASK-002",implementation_targets=["src/a.ts","src/b.ts"])
    task=TaskDefinition(task_id="TASK-002",task_type="code_change",title="x",objective="x",source_step_order=1)
    store=JsonTaskArtifactStore(root=tmp_path/"artifacts")
    builder=ResolvedTaskContextBuilder(artifact_resolver=TaskArtifactResolver(store=store))
    bundle=builder.build(
        repository_root=tmp_path,
        requirement=RequirementSpec(title="x",summary="x"),
        technical_spec=TechnicalSpec(summary="x"),
        task=task,
        repository=RepositoryProfile(project_name="x",root_path=str(tmp_path),framework="Vue3",language="TypeScript"),
        index=index,selection=selection,surface=surface,
    )
    assert {x.relative_path for x in bundle.code_context} == {"src/a.ts","src/b.ts"}
    assert bundle.change_surface["implementation_targets"] == ["src/a.ts","src/b.ts"]
