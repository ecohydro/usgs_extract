import pandas as pd
import requests
import json
import os
import sys
import csv
import re
import ast

key = "sk-a4152196f16845598b062a9a1fc82f38"

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
    out_csv = sys.argv[1]
    root_json = sys.argv[2]

    model = "phi4:latest" # llama3:latest, deepseek-r1:latest


    # COLUMN SUBSETS

    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "TOWNSHIPS_RANGES_SECTIONS", "WATERSOURCE_NAME", "LOCATION", "COUNTY", "DATE_OF_DATA", "RESOLUTION","UNITS_USED", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "WATERSOURCE_NAME", "COUNTY", "DATE_OF_DATA", "RESOLUTION", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "COUNTY", "DATE_OF_DATA", "WATER_TYPE"

    columns = [
        "ID", "PAGE_NUMBER", "Latitude", "Longitude", "Location", "Townships_Ranges_Sections",
        "Watersource_Name", "County","Dates_of_Recording", "Temporal_Resolution",
        "Units_Of_Measurement", "Water_Type"
    ]

    # PROOF OF CONCEPT 

    df = pd.DataFrame(columns=columns)

    # Loop through the JSON files in the Root Directory
    for f in os.listdir(root_json):

        # Fix the ID and Page number
        filep = os.path.join(root_json, f)
        id = f.split('_')[0]
        page_num = f.split('_')[-1][:-5]

        # Sanity Check
        print(id, page_num)

        # Parse the JSON for content only i.e. remove bboxes, types, job-ids, and etc
        input = ""
        
        with open(filep) as json_file:
            parsed = json.load(json_file)['result']['chunks']

        table_exists = False

        for content in parsed:
            for block in content["blocks"]:
                if block["type"] == "Table":
                    table_exists = True
                    continue
            if table_exists:
                continue
            
        
        if parsed == []:
            continue
        elif not table_exists:
            continue
        else:
            # input = parsed[0]['content'] 
            for content in parsed:
                input = input + content['content']
        
        print(input)

        # Terms we want to capture
        v1 = "Groundwater", "Stream Discharge", "Precipitation", "Springs", "Reservoir", "Irrigation", "Water Quality"
        v2 = "Groundwater well", "Groundwater recharge", "Stream discharge", "Precipitation", "Springs", "Groundwater water quality", "Spring water quality", "Stream water quality", "Irrigation", "Reservoir", "Other (Choose a water type that best describes the data)"
        v3 = "Groundwater", "Stream", "Precipitation", "Springs", "Reservoir", "Other (Choose a water type that best describes the data)" 
        
        # User Prompt

        Prompt_sum = f""" Summarize the entire Document and tell me about these information; "Latitude", "Longitude", "Townships_Ranges_Sections",
            "Watersource_Name", "County","Dates_of_Recording", "Temporal_Resolution",
            "Units_Of_Measurement", "Water_Type". Where water type is one of the following: {v1}

            Here is the Document: \n{input}\n
            """
        
        r1_summarize = f"""
            Reply only with valid JSON format, 

            {{
            "Watersource_Name" : ...,
            "County": ...,
            "Location": ...,
            "Townships_Ranges_Sections": ... ("[TRS, TRS, ...]"),
            "Temporal_Resolution": ...(time interval between sequential measurements),
            "Units_Of_Measurement": ...,
            "Latitude": ... (number in decimals),
            "Longitude": ... (number in decimals),
            "Dates_of_Recording": ... (MM/DD/YYYY-MM/DD/YYYY),
            "Water_Type": {v2}
            }}

            There may be multiple watersources. In the case that multiple water sources are present, list out each water source individually. In the case that no watersource is present, do not return a JSON output.
            All data is from the State of California. If there is no geospatial data in the document, try to locate the watersource yourself.
            Some documents may have tables with column names that are misleading or incorrect. Double check the given categories.
            Double-check your answers. Make sure that it is compatible with json.loads() and there are commas separating each key-value pair and each JSON Object. Do not give examples or comments inside the JSON chunk. 

            Here is the document content:\n{input}\n
        
            """

        # Retrieve Output
        doc_summary = chat_with_model(key, r1_summarize, model)['choices'][0]['message']['content']
        with open(id + "_" + page_num + ".txt", "w") as text_file:
            print(doc_summary, file=text_file)
    #     # Check
    #     print(doc_summary)

        # Retrieve only JSON object from output
        match = re.search(r"```json\s*(.*?)\s*```", doc_summary, re.DOTALL)
        if match:
            cleaned_json = match.group(1)
        else:
            continue


        print(cleaned_json)

        # Load JSON into Dataframe
        try:
            dicts = json.loads(cleaned_json) 
            if isinstance(dicts, list):
                for dic in dicts:
                    dic.update({"ID": id, "PAGE_NUMBER": page_num})
                df = pd.concat([df, pd.DataFrame(dicts)], ignore_index=True)
            else:
                dicts.update({"ID": id, "PAGE_NUMBER": page_num})
                df = pd.concat([df, pd.DataFrame([dicts])], ignore_index=True)
        
        except Exception as e:
            try:
                cleaned_json = ''.join(cleaned_json.rsplit(',', 1)[0]) + "]"
                dicts = json.loads(cleaned_json) 
                if isinstance(dicts, list):
                    for dic in dicts:
                        dic.update({"ID": id, "PAGE_NUMBER": page_num})
                    df = pd.concat([df, pd.DataFrame(dicts)], ignore_index=True)
                else:
                    dicts.update({"ID": id, "PAGE_NUMBER": page_num})
                    df = pd.concat([df, pd.DataFrame([dicts])], ignore_index=True)
            except Exception as f:
                print(f)

    # Save
    df.to_csv(out_csv)



