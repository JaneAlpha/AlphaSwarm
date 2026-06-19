# Project Rules

## Checkpoints Directory Constraints

- Do not create any new files under `checkpoints/`.
- Do not delete any files under `checkpoints/`.
- Do not rename any files under `checkpoints/`.
- Do not change the directory structure under `checkpoints/`.
- Only modify an existing file under `checkpoints/` when the user explicitly names that file as writable in the current request.
- If a task appears to require creating, deleting, renaming, or moving any file under `checkpoints/`, stop and ask the user for explicit confirmation before taking action.

## Codex Operating Rule

Before performing any task that involves `checkpoints/`, Codex must respect the constraints above. A general request to inspect, complete, fix, or improve the project is not permission to add, delete, rename, or move files in `checkpoints/`.
