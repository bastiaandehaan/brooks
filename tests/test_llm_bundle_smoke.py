import subprocess
from pathlib import Path


def test_make_llm_bundle_creates_outputs():
    root = Path.cwd()
    script = root / ""bundle_repo.py"
    assert script.exists(), "bundle_repo.py missing"

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Root",
        str(root),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"PowerShell failed:\n{result.stderr}"

    # REQUIRED outputs
    dump = root / "llm_dump.txt"
    export_dir = root / "_llm_export"

    assert dump.exists(), "llm_dump.txt not created"
    assert export_dir.exists(), "_llm_export not created"

    content = dump.read_text(encoding="utf-8", errors="ignore")
    assert "### FILE:" in content
    assert len(content) > 100, "llm_dump.txt seems empty"

    # OPTIONAL output (best effort)
    zip_path = root / "llm_bundle.zip"
    if not zip_path.exists():
        print("NOTE: llm_bundle.zip not created (allowed on some PS versions)")
