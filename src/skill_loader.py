from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SkillLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path


class SkillLoader:
    """Read-only loader for project-local SKILL.md files."""

    def __init__(self, skill_root: str | Path) -> None:
        self.skill_root = Path(skill_root).resolve()
        self._registry: dict[str, SkillMetadata] | None = None

    def list_skills(self) -> dict[str, Any]:
        registry = self._load_registry()
        skills = [
            {
                "name": item.name,
                "description": item.description,
                "source_path": item.path.as_posix(),
            }
            for item in sorted(registry.values(), key=lambda value: value.name)
        ]
        return {
            "valid": True,
            "skill_root": self.skill_root.as_posix(),
            "skill_count": len(skills),
            "skills": skills,
        }

    def load_skill(self, name: str) -> dict[str, Any]:
        registry = self._load_registry()
        if name not in registry:
            raise SkillLoaderError(f"Unknown skill: {name}")
        metadata = registry[name]
        content = self._read_allowed_skill_file(metadata.path)
        return {
            "valid": True,
            "name": metadata.name,
            "description": metadata.description,
            "content": content,
            "source_path": metadata.path.as_posix(),
        }

    def _load_registry(self) -> dict[str, SkillMetadata]:
        if self._registry is not None:
            return self._registry
        if not self.skill_root.exists():
            raise SkillLoaderError(f"Skill root does not exist: {self.skill_root}")
        if not self.skill_root.is_dir():
            raise SkillLoaderError(f"Skill root is not a directory: {self.skill_root}")

        registry: dict[str, SkillMetadata] = {}
        for path in sorted(self.skill_root.glob("*/SKILL.md")):
            content = self._read_allowed_skill_file(path)
            metadata = self._parse_frontmatter(content, path)
            if metadata.name in registry:
                raise SkillLoaderError(f"Duplicate skill name: {metadata.name}")
            registry[metadata.name] = metadata
        self._registry = registry
        return registry

    def _read_allowed_skill_file(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.skill_root)
        except ValueError as exc:
            raise SkillLoaderError(f"Skill path escapes skill root: {resolved}") from exc
        if resolved.name != "SKILL.md":
            raise SkillLoaderError(f"Only SKILL.md can be loaded: {resolved}")
        if resolved.parent.parent != self.skill_root:
            raise SkillLoaderError(f"Skill file must be directly under a skill folder: {resolved}")
        return resolved.read_text(encoding="utf-8")

    @staticmethod
    def _parse_frontmatter(content: str, path: Path) -> SkillMetadata:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillLoaderError(f"Missing YAML frontmatter: {path}")
        end_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = index
                break
        if end_index is None:
            raise SkillLoaderError(f"Unclosed YAML frontmatter: {path}")

        fields: dict[str, str] = {}
        for line in lines[1:end_index]:
            stripped = line.strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise SkillLoaderError(f"Invalid frontmatter line in {path}: {line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            fields[key] = value

        extra_keys = set(fields) - {"name", "description"}
        if extra_keys:
            raise SkillLoaderError(f"Unsupported frontmatter keys in {path}: {sorted(extra_keys)}")
        name = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        if not name:
            raise SkillLoaderError(f"Missing skill name in {path}")
        if not description:
            raise SkillLoaderError(f"Missing skill description in {path}")
        return SkillMetadata(name=name, description=description, path=path.resolve())
