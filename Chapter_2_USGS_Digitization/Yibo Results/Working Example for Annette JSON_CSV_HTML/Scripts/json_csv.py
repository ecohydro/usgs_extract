import json
import pandas as pd
from bs4 import BeautifulSoup
import os
import sys

def html_table_to_csv(html_table, output_filename):
    """
    Convert an HTML table to CSV using pandas.
    """

    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    

    rows = table.find_all('tr')
    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
    

    data = []
    for row in rows[1:]:
        data.append([td.get_text(strip=True) for td in row.find_all('td')])
    

    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

def process_json_file(json_file_path, output_dir='tables_csv'):
    """
    Process the JSON file to extract HTML tables and save them as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    table_counter = 1
    for chunk in data['result']['chunks']:
        for block in chunk['blocks']:
            if block['type'] == 'Table':
                html_table = block['content']
                output_filename = os.path.join(output_dir, f'table_{table_counter}.csv')
                html_table_to_csv(html_table, output_filename)
                table_counter += 1

if __name__ == '__main__':
    json_file_path = sys.argv[1]
    process_json_file(json_file_path)