import pandas as pd
import requests
import json
import os
import sys
import csv
import re
import ast

key = "sk-79866d820a034bd29523cf83ef44386d"

# Chat Completion API Call
def chat_with_model(token, promp, model):
    url = 'https://llm.grit.ucsb.edu/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'response_format': 'json_object'
    }
    
    data = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": promp
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ ==  "__main__": 

    # Declare Root and Target Directories in command line
    out_csv = "/home/waves/data/usgs_extract/metadataResults/metadata.csv"
    root_json = "/home/waves/data/usgs_extract/metadataResults/json"

    model = "phi4:latest" # llama3:latest, deepseek-r1:latest


    # COLUMN SUBSETS

    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "TOWNSHIPS_RANGES_SECTIONS", "WATERSOURCE_NAME", "LOCATION", "COUNTY", "DATE_OF_DATA", "RESOLUTION","UNITS_USED", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "WATERSOURCE_NAME", "COUNTY", "DATE_OF_DATA", "RESOLUTION", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "COUNTY", "DATE_OF_DATA", "WATER_TYPE"


    columns = [
        "ID", "PAGE_NUMBER", "Inferred_Latitude", "Inferred_Longitude", "Actual_Latitude", "Actual_Longitude", "Location", "Townships_Ranges_Sections",
        "Watersource_Name", "Actual_County", "Inferred_County", "Dates_of_Recording", "Temporal_Resolution",
        "Units_Of_Measurement", "Water_Type", "KeyTerms"
    ]

    df = pd.read_csv(out_csv)
    df_init = df.copy()

    # Loop through the JSON files in the Root Directory
    for dirEntry in os.scandir(root_json):

        # Fix the ID and Page number
        filep = dirEntry.path

        try:
            if not dirEntry.is_file() or not dirEntry.name.endswith(".json"):
                continue
            elif dirEntry.name[0] == "r":
                id = "SantaBarbaraDoc"
                page_num = "FULL"
            else:
                id = dirEntry.name.split('_')[0]
                page_num = dirEntry.name.split('_')[-1][:-5]

        except Exception as e:
            continue

        mask = (
            df_init["ID"].astype(str) == str(id)
        ) & (
            df_init["PAGE_NUMBER"].astype(str) == str(page_num)
        )

        if mask.any():
            print(f"{id}, {page_num} already parsed")
            continue
        else:
            print(id, page_num)

        # Parse the JSON for content only i.e. remove bboxes, types, job-ids, and etc
        doc_content = ""
        
        try:
            with open(filep) as json_file:
                parsed = json.load(json_file)['result']['chunks']

            table_exists = False

            table_exists = any(
                block["type"] == "Table"
                for content in parsed
                for block in content["blocks"]
            )           
            
            if parsed == []:
                continue
            elif not table_exists:
                continue
            else:
                # input = parsed[0]['content'] 
                for content in parsed:
                    doc_content += content['content']

        except Exception as e:
            print(e)
            continue
            
        # Terms we want to capture
        v1 = "Groundwater", "Stream Discharge", "Precipitation", "Springs", "Reservoir", "Irrigation", "Water Quality", "Not Water Related"
        v2 = "Groundwater well", "Groundwater recharge", "Stream discharge", "Precipitation", "Springs", "Groundwater water quality", "Spring water quality", "Stream water quality", "Irrigation", "Reservoir", "Other (Choose a water type that best describes the data)"
        v3 = "Groundwater", "Stream", "Precipitation", "Springs", "Reservoir", "Other (Choose a water type that best describes the data)" 
        
        r1_summarize = f"""
            Reply only with valid JSON format, 

            {{
            "Watersource_Name" : ...,
            "Actual_County": ...,
            "Inferred_County": ...,
            "Location": ...,
            "Townships_Ranges_Sections": ... ("[TRS, TRS, ...]"),
            "Temporal_Resolution": ...(time interval between sequential measurements),
            "Units_Of_Measurement": ...,
            "Actual_Latitude": ... (number in decimals),
            "Actual_Longitude": ... (number in decimals),
            "Inferred_Latitude": ... (number in decimals),
            "Inferred_Longitude": ... (number in decimals),
            "Dates_of_Recording": ... (MM/DD/YYYY-MM/DD/YYYY),
            "Water_Type": {v1},
            "KeyTerms" ... (Terms in the document that corresponds to the Water_Type)
            }}

            There may be multiple watersources. In the case that multiple water sources are present, list out each water source individually. In the case that no watersource is present, do not return a JSON output.
            All data is from the State of California. If there is no latitude and longitude data in the document, use inference to locate the watersource and record the results as "Inferred_Latitude" and "Inferred_Longitude"; do not make comments with \\\\.
            Some documents may have tables with column names that are misleading or incorrect. Double check the given categories.
            Double-check your answers. Make sure that it is compatible with json.loads() and there are commas separating each key-value pair and each JSON Object. Do not give examples or comments inside the JSON chunk. 

            Here is the document content:\n{doc_content}\n
        
            """
        # Retrieve Output

        try:
            response = chat_with_model(key, r1_summarize, model)
            doc_summary = response['choices'][0]['message']['content']


            # Retrieve only JSON object from output
            match = re.search(r"```json\s*(.*?)\s*```", doc_summary, re.DOTALL)
            if match:
                cleaned_json = match.group(1)
            else:
                continue


            cleaned = re.sub(r'//.*', '', cleaned_json)
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)

            # Load JSON into Dataframe
            try:
                dicts = json.loads(cleaned) 
                if isinstance(dicts, list):
                    for dic in dicts:
                        dic.update({"ID": id, "PAGE_NUMBER": page_num})
                    df = pd.concat([df, pd.DataFrame(dicts)], ignore_index=True)
                else:
                    dicts.update({"ID": id, "PAGE_NUMBER": page_num})
                    df = pd.concat([df, pd.DataFrame([dicts])], ignore_index=True)
            
            except Exception as e:
                try:
                    cleaned = ''.join(cleaned.rsplit(',', 1)[0]) + "]"
                    dicts = json.loads(cleaned) 
                    if isinstance(dicts, list):
                        for dic in dicts:
                            dic.update({"ID": id, "PAGE_NUMBER": page_num})
                        df = pd.concat([df, pd.DataFrame(dicts)], ignore_index=True)
                    else:
                        dicts.update({"ID": id, "PAGE_NUMBER": page_num})
                        df = pd.concat([df, pd.DataFrame([dicts])], ignore_index=True)
                except Exception as f:
                    with open(os.path.join("/home/waves/data/usgs_extract/metadataResults/error_log",id + "_" + page_num + ".txt"), "w") as text_file:
                        print(doc_summary, file=text_file)

            df.to_csv(out_csv, index=False)
            
        except Exception as e:
            print(e)

