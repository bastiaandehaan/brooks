import os
from pathlib import Path


def bundle_repository(output_filename="full_repository_bundle.txt"):
    # Mappen die we willen overslaan (om de file niet onnodig groot te maken)
    ignore_dirs = {
        '.git', '__pycache__', '.venv', 'venv', '.idea',
        '.pytest_cache', '.mypy_cache', 'logs', 'backtest_png'
    }

    # Bestands extensies die we willen meenemen
    include_extensions = {'.py', '.yaml', '.toml', '.json', '.md', '.ps1', '.txt'}

    # Het pad waar het script staat (de root van je project)
    root_dir = Path(__file__).parent

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f"### REPOSITORY BUNDLE GENERATED ON {os.getpid()} ###\n")
        outfile.write(f"### ROOT: {root_dir} ###\n\n")

        for root, dirs, files in os.walk(root_dir):
            # Filter de negeer-mappen
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                file_path = Path(root) / file

                # Sla het output bestand zelf over en check de extensie
                if file == output_filename:
                    continue

                if file_path.suffix in include_extensions:
                    try:
                        relative_path = file_path.relative_to(root_dir)
                        outfile.write(f"\n\n### FILE: {relative_path}\n")
                        outfile.write("-" * 40 + "\n")

                        with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                            outfile.write(infile.read())

                        outfile.write("\n" + "=" * 60 + "\n")
                        print(f"Toegevoegd: {relative_path}")
                    except Exception as e:
                        print(f"Kon {file} niet lezen: {e}")

    print(f"\n✅ Klaar! Alles staat in: {output_filename}")


if __name__ == "__main__":
    bundle_repository()