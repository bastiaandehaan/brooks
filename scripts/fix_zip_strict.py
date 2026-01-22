#!/usr/bin/env python3
"""Fix duplicate strict=True in zip() calls"""

import re
from pathlib import Path


def fix_zip_strict(filepath: Path) -> bool:
    """Remove duplicate strict=True from zip calls"""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Fix pattern: zip(..., strict=True)
    # Replace with: zip(..., strict=True)
    pattern = r"zip\(([^)]+?)(?:,\s*strict=True)+\)"
    replacement = r"zip(\1, strict=True)"
    content = re.sub(pattern, replacement, content)

    # Fix broken zip(*grid.items(), strict=True...
    # Should be: zip(*grid.items(), strict=True)
    content = re.sub(
        r"zip\(\*grid\.items\(\s*,\s*strict=True", "zip(*grid.items(), strict=True", content
    )

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def fix_other_issues():
    """Fix remaining issues"""

    # Fix 1: strategies/h2l2.py - i should be idx
    file = Path("strategies/h2l2.py")
    content = file.read_text()
    content = content.replace(
        "window = m5.iloc[idx - p.pullback_bars + 1 : i + 1]",
        "window = m5.iloc[idx - p.pullback_bars + 1 : idx + 1]",
    )
    # Fix ambiguous 'l' -> 'low'
    content = content.replace(
        "def _is_rejection_bar(o: float, h: float, l: float, c: float, side: Side, frac: float) -> bool:",
        "def _is_rejection_bar(o: float, h: float, low: float, c: float, side: Side, frac: float) -> bool:",
    )
    content = content.replace('    l = float(bar["low"])', '    low = float(bar["low"])')
    # Replace l usage with low
    content = re.sub(r"\bl\b(?!\w)", "low", content)
    file.write_text(content)
    print("✅ Fixed h2l2.py")

    # Fix 2: test_selection_stability.py - i should be idx
    file = Path("tests/test_selection_stability.py")
    content = file.read_text()
    content = content.replace(
        'signal_ts=_utc(f"2026-01-05 14:{50+i}:00"),',
        'signal_ts=_utc(f"2026-01-05 14:{50+idx}:00"),',
    )
    content = content.replace("stop=4799.0 - i * 0.1,", "stop=4799.0 - idx * 0.1,")
    file.write_text(content)
    print("✅ Fixed test_selection_stability.py")

    # Fix 3: test_debug_system.py - uncomment and fix
    file = Path("tests/test_debug_system.py")
    content = file.read_text()
    content = content.replace(
        '    # recent_errors = debug.get_recent_logs(n=5) # Method missing\n    print(f"✅ Found {len(recent_errors)} recent error logs")',
        '    recent_errors = debug.get_recent_errors(count=5)\n    print(f"✅ Found {len(recent_errors)} recent error logs")',
    )
    file.write_text(content)
    print("✅ Fixed test_debug_system.py")

    # Fix 4: execution/emergency_stop.py - specific exception
    file = Path("execution/emergency_stop.py")
    content = file.read_text()
    content = content.replace(
        "            try:\n                reason = self.stop_file.read_text().strip()\n            except:",
        "            try:\n                reason = self.stop_file.read_text().strip()\n            except (OSError, UnicodeDecodeError):",
    )
    file.write_text(content)
    print("✅ Fixed emergency_stop.py")

    # Fix 5: Remove unused variables
    file = Path("scripts/run_all_tests.py")
    content = file.read_text()
    content = content.replace(
        "        result = subprocess.run(cmd, check=True, capture_output=False)",
        "        subprocess.run(cmd, check=True, capture_output=False)",
    )
    file.write_text(content)
    print("✅ Fixed run_all_tests.py")

    file = Path("scripts/test_ftmo_data_limits.py")
    content = file.read_text()
    content = content.replace(
        '    m1_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M1, "M1", 1440)',
        '    _ = find_exact_limit(client, symbol, mt5.TIMEFRAME_M1, "M1", 1440)  # noqa: F841',
    )
    content = content.replace(
        '    m5_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M5, "M5", 288)',
        '    _ = find_exact_limit(client, symbol, mt5.TIMEFRAME_M5, "M5", 288)  # noqa: F841',
    )
    content = content.replace(
        '    m15_limit = find_exact_limit(client, symbol, mt5.TIMEFRAME_M15, "M15", 96)',
        '    _ = find_exact_limit(client, symbol, mt5.TIMEFRAME_M15, "M15", 96)  # noqa: F841',
    )
    file.write_text(content)
    print("✅ Fixed test_ftmo_data_limits.py")

    # Fix 6: utils/mt5_client.py - B008
    file = Path("utils/mt5_client.py")
    content = file.read_text()
    content = content.replace(
        "    def __init__(self, mt5_module, params: Mt5ConnectionParams = Mt5ConnectionParams()):",
        "    def __init__(self, mt5_module, params: Mt5ConnectionParams | None = None):",
    )
    # Add check at start of __init__
    content = content.replace(
        "    def __init__(self, mt5_module, params: Mt5ConnectionParams | None = None):\n        self._mt5 = mt5_module",
        "    def __init__(self, mt5_module, params: Mt5ConnectionParams | None = None):\n        if params is None:\n            params = Mt5ConnectionParams()\n        self._mt5 = mt5_module",
    )
    file.write_text(content)
    print("✅ Fixed mt5_client.py")


if __name__ == "__main__":
    print("🔧 Fixing duplicate strict=True and other issues\n")

    # Fix all zip() issues
    fixed = 0
    for file in Path(".").rglob("*.py"):
        if any(ex in file.parts for ex in [".venv", "venv", "__pycache__"]):
            continue
        if fix_zip_strict(file):
            print(f"✅ Fixed zip in: {file}")
            fixed += 1

    print(f"\n✅ Fixed {fixed} files with zip issues")
    print("\n🔧 Fixing other issues...")
    fix_other_issues()
    print("\n✅ All fixes complete!")
