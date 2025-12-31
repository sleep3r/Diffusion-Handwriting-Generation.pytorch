#!/usr/bin/env python3
"""
Script to prepare IAM dataset by splitting lines.txt into individual form text files.
"""

from collections import defaultdict
from pathlib import Path


def parse_lines_file(lines_file: Path) -> dict[str, dict[str, str]]:
    """
    Parse lines.txt and group lines by form ID.

    Args:
        lines_file: Path to lines.txt

    Returns:
        Dictionary mapping form IDs to line texts
        e.g., {"a01-000u": {"a01-000u-00": "A MOVE to stop...", ...}}
    """
    forms = defaultdict(dict)

    with lines_file.open("r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split(" ")
            if len(parts) < 9:
                continue

            line_id = parts[0]  # e.g., "a01-000u-00"

            # Extract text (everything after the 8th space, replace | with spaces)
            text = " ".join(parts[8:]).replace("|", " ")

            # Extract form ID (everything before the last hyphen and digits)
            # e.g., "a01-000u-00" -> "a01-000u"
            form_id = "-".join(line_id.split("-")[:-1])

            forms[form_id][line_id] = text

    return dict(forms)


def create_form_files(forms: dict[str, dict[str, str]], ascii_dir: Path) -> None:
    """
    Create individual form text files with CSR sections.

    Args:
        forms: Dictionary mapping form IDs to line texts
        ascii_dir: Path to ascii directory
    """
    created_count = 0

    for form_id, lines in forms.items():
        # Create directory structure: ascii/a01/a01-000u/
        # form_id[:3] gives "a01", form_id[:7] gives "a01-000"
        form_dir = ascii_dir / form_id[:3] / form_id[:7]
        form_dir.mkdir(parents=True, exist_ok=True)

        # Create the form text file
        form_file = form_dir / f"{form_id}.txt"

        with form_file.open("w") as f:
            f.write("CSR:\n")

            # Write each line's text
            for line_id in sorted(lines.keys()):
                text = lines[line_id]
                f.write(f"{text}\n")

        created_count += 1

        if created_count % 100 == 0:
            print(f"Created {created_count} form files...")

    print(f"\nTotal: Created {created_count} form text files")


def main():
    # Setup paths
    data_dir = Path(__file__).parent / "data"
    lines_file = data_dir / "ascii" / "lines.txt"
    ascii_dir = data_dir / "ascii"

    if not lines_file.exists():
        print(f"Error: {lines_file} not found!")
        return

    print("Parsing lines.txt...")
    forms = parse_lines_file(lines_file)
    print(f"Found {len(forms)} forms")

    print("\nCreating form text files...")
    create_form_files(forms, ascii_dir)

    print("\nDone! Data preparation complete.")


if __name__ == "__main__":
    main()
