# Document Processing Tools

This package contains two Python scripts for processing JSON documents containing text and table data.

## Scripts:

### json_csv.py

Extracts tables from JSON and saves them as CSV files

### reconstruct_bbox.py

Reconstructs the document with proper layout in HTML format

## Installation:

First install the required packages:

```bash
pip install pandas beautifulsoup4 lxml
```

## Usage:

To extract tables as CSV files:

```bash
python json_csv.py yourfile.json
```

To reconstruct the document layout:

```bash
python reconstruct_bbox.py yourfile.json
``` 

Output Files:

json_csv.py will create:

- A tables_csv directory containing CSV files (table_1.csv, table_2.csv, etc.)

reconstruct_bbox.py will create:

- A [filename]_tables directory with individual HTML tables
- A [filename]_document.html file with the full reconstructed document

## Requirements:
- Python 3.x
- Packages: pandas, beautifulsoup4, lxml

## Notes:
- Input JSON must follow the expected format with 'chunks' and 'blocks'
- Tables must be marked with type 'Table'
- Text content maintains original line breaks