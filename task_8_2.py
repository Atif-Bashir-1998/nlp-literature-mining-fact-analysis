import os
from lxml import etree
import pandas as pd

# ------------------------------
# CONFIGURATION
# ------------------------------

base_dir = os.path.dirname(__file__)
tei_dir = os.path.join(base_dir, "paper_tei_files")
output_dir = os.path.join(base_dir, "tables_output")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# LOOP THROUGH ALL TEI FILES
# ------------------------------
tei_files = [f for f in os.listdir(tei_dir) if f.lower().endswith("_output.tei.xml")]

if not tei_files:
    print(" No TEI XML files found in the directory.")
else:
    print(f" Found {len(tei_files)} TEI XML files.")

for tei_file in sorted(tei_files):
    tei_path = os.path.join(tei_dir, tei_file)

    # Extract Paper ID (e.g., Paper_1 from Paper_1_output.tei.xml)
    paper_id = tei_file.split("_output.tei.xml")[0]
    paper_folder = os.path.join(output_dir, paper_id)
    os.makedirs(paper_folder, exist_ok=True)

    print(f"\n Processing {tei_file} → saving to {paper_folder}")

    try:
        tree = etree.parse(tei_path)
        root = tree.getroot()
        tables = root.findall(".//{*}table")

        if not tables:
            print(f" No tables found in {tei_file}.")
            continue

        print(f"Found {len(tables)} tables in {paper_id}.")

        for i, table in enumerate(tables, start=1):
            rows_data = []
            for row in table.findall(".//{*}row"):
                cells = [("".join(cell.itertext())).strip() for cell in row.findall(".//{*}cell")]
                rows_data.append(cells)

            # Convert to DataFrame
            df = pd.DataFrame(rows_data)
            csv_path = os.path.join(paper_folder, f"table_{i}.csv")
            df.to_csv(csv_path, index=False, header=False)
            print(f"📄 Saved Table {i} → {csv_path}")

    except Exception as e:
        print(f" Error processing {tei_file}: {e}")

print("\n All TEI files processed and tables extracted!")
