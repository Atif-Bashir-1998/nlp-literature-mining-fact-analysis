import os
import json

folder_path = r"C:\Users\hp830\Desktop\OULU\University\Courses\Natural_Language_Processing_and_Text_Mining_521158S-3006\Project\Analysis\Test_Pdfs"  # change this to your folder path
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

papers_json = {f"Paper_{i+1}": pdf for i, pdf in enumerate(pdf_files)}

# Save to JSON file
with open("papers_list.json", "w") as f:
    json.dump(papers_json, f, indent=4)

print("JSON file created successfully!")