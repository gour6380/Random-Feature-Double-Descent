"""Standard-library static checks only: never import or execute src/notebook cells."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = []
    files = [
        *root.joinpath("src").glob("*.py"),
        *root.joinpath("tests").glob("*.py"),
        *root.joinpath("tools").glob("*.py"),
    ]
    for path in files:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
    notebooks = sorted(root.joinpath("notebooks").glob("*.ipynb"))
    if not notebooks:
        errors.append("No workflow notebook found")
    notebook_cells = {}
    saved_output_cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        notebook_cells[path.name] = len(code_cells)
        for index, cell in enumerate(code_cells):
            source = "".join(cell["source"])
            # Saved outputs are intentional evidence, not a reason to rewrite the notebook.
            saved_output_cells += bool(cell.get("outputs"))
            # Optional TensorBoard magic is deliberately a separate notebook action.
            lines = [line for line in source.splitlines() if not line.lstrip().startswith("%")]
            try:
                ast.parse("\n".join(lines), filename=f"{path.name}-cell-{index}")
            except SyntaxError as exc:
                errors.append(str(exc))
            if re.search(r"/Users/|/home/|[A-Z]:\\\\Users\\\\", source):
                errors.append(f"{path.name} cell {index} contains a personal absolute path")
        language = notebook.get("metadata", {}).get("kernelspec", {}).get("language", "python")
        if language != "python":
            errors.append(f"Expected a Python notebook in {path.name}")
    requirements = (root / "requirements.txt").read_text()
    blocks = re.split(r"\n(?=[A-Za-z0-9])", requirements)
    pins = []
    for block in blocks:
        first = block.splitlines()[0]
        if not first or first.startswith("#"):
            continue
        if not re.match(r"^[A-Za-z0-9_.-]+==[^\s;]+", first) or "--hash=sha256:" not in block:
            errors.append(f"Dependency lacks exact pin/hash: {first}")
        pins.append(first)
    local_links = 0
    documents = [
        root / "README.md",
        *root.joinpath("docs").glob("*.md"),
        *root.joinpath("results").rglob("*.md"),
    ]
    for path in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            resolved = path.parent / unquote(target.split("#", 1)[0])
            if not resolved.exists():
                errors.append(f"Broken local link in {path.name}: {target}")
            local_links += 1
    if errors:
        raise SystemExit("\n".join(errors))
    result_files = 0
    for manifest_path in sorted(root.joinpath("results").glob("*/manifest.json")):
        base = manifest_path.parent.resolve()
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["files"].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base):
                errors.append(f"Result path escapes its package: {relative}")
            elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Result checksum mismatch: {relative}")
            result_files += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        json.dumps(
            {
                "status": "passed_static_only",
                "python_files": len(files),
                "notebook_code_cells": notebook_cells,
                "retained_output_cells": saved_output_cells,
                "hashed_dependency_pins": len(pins),
                "local_links": local_links,
                "verified_result_files": result_files,
            }
        )
    )


if __name__ == "__main__":
    main()
