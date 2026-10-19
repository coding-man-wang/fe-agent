# fe-agent Benchmark Cases v0.1

Base commit: `f5c29aac0d6dee053fdf10f6118e82b5adbd17af`

## 使用说明
- `runnable/CASE-001.json` ~ `CASE-012.json`：当前 Planning/Retrieval Harness 可以读取。
- `review_drafts/CASE-013.json` ~ `CASE-015.json`：先保存，不建议放入当前 `benchmark/cases/` 自动 run-all。
  真正的 Review Revision 测试还需要“初始 Patch/PR + Review 意见”作为前置夹具，当前 Case Schema 还没表达这一层。

## 当前评估模型的一个已知限制
`expected_files` 目前同时承担了“应该检索到的上下文文件”和“最终应该修改的文件”两种含义。
后续建议拆为：
- required_context_files
- useful_context_files
- expected_change_files
- allowed_change_files
- forbidden_change_files
