import sys
import json
import os
from bs4 import BeautifulSoup


def process_content(block, output_dir='tables'):
    content = block['content'].strip()
    if block['type'] == 'Table':
        # Create output directory for tables if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process table HTML
        soup = BeautifulSoup(content, 'html.parser')
        for table in soup.find_all('table'):
            table['style'] = 'width: 100%; table-layout: auto;'
        
        # Generate unique filename for the table
        table_filename = f"table_page_{block['bbox']['page']}_pos_{block['bbox']['top']}_{block['bbox']['left']}.html"
        table_path = os.path.join(output_dir, table_filename)
        
        # Save individual table HTML
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(f"<!DOCTYPE html>\n<html>\n<head><title>{table_filename}</title></head>\n<body>\n")
            f.write(str(soup))
            f.write("\n</body>\n</html>")
        
        # Return link to the table for the main document
        return f'<div class="table-wrapper"><a href="{table_path}" target="_blank">View table in separate window</a>{str(soup)}</div>'
    
    html_content = content.replace("\n", "<br>")
    return f'<div class="text-content">{html_content}</div>'


def create_flow_layout_html_document(json_data, output_file='reconstructed_document_flow.html', tables_dir='tables'):
    blocks = []
    for chunk in json_data['result']['chunks']:
        blocks.extend(chunk['blocks'])

    # Sort blocks by page and vertical position
    sorted_blocks = sorted(blocks, key=lambda x: (x['bbox']['page'], x['bbox']['top'], x['bbox']['left']))

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Reconstructed Document (Flow Layout)</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .document {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            padding: 40px;
        }}
        .page {{
            margin-bottom: 50px;
            padding: 20px;
            background: white;
            page-break-after: always;
        }}
        .text-block {{
            margin-bottom: 15px;
            line-height: 1.5;
            white-space: pre-wrap;
        }}
        .table-container {{
            margin: 15px 0;
            overflow-x: auto;
            border: 1px solid #ccc;
            position: relative;
        }}
        .table-wrapper {{
            overflow-x: auto;
            max-width: 100%;
        }}
        .table-link {{
            position: absolute;
            right: 10px;
            top: 10px;
            background: #f2f2f2;
            padding: 2px 5px;
            font-size: 12px;
            border-radius: 3px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
            white-space: nowrap;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="document">
        {content}
    </div>
</body>
</html>"""

    pages = {}
    for block in sorted_blocks:
        page_num = block['bbox']['page']
        if page_num not in pages:
            pages[page_num] = []
        block_class = "text-block" if block['type'] != 'Table' else "table-container"
        block_content = process_content(block, tables_dir)
        block_html = f'<div class="{block_class}">{block_content}</div>'
        pages[page_num].append(block_html)

    content = ""
    for page_num, page_blocks in sorted(pages.items()):
        content += f'<div class="page" id="page-{page_num}">\n'
        content += "\n".join(page_blocks)
        content += '\n</div>\n'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template.format(content=content))

    return output_file


if __name__ == '__main__':
    json_file_path = sys.argv[1]
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output directory name based on input JSON filename
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    tables_dir = f"{base_name}_tables"
    
    create_flow_layout_html_document(data, output_file=f"{base_name}_document.html", tables_dir=tables_dir)