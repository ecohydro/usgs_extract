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

    out_csv = sys.argv[1]
    root_json = sys.argv[2]

    model = "llama3:latest"
    model_r1 = "deepseek-r1:latest"

    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "TOWNSHIPS_RANGES_SECTIONS", "WATERSOURCE_NAME", "LOCATION", "COUNTY", "DATE_OF_DATA", "RESOLUTION","UNITS_USED", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "WATERSOURCE_NAME", "COUNTY", "DATE_OF_DATA", "RESOLUTION", "WATER_TYPE"
    # "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "COUNTY", "DATE_OF_DATA", "WATER_TYPE"

    columns = [
        "ID", "PAGE_NUMBER", "LATITUDE", "LONGITUDE", "TOWNSHIPS_RANGES_SECTIONS",
        "WATERSOURCE_NAME", "LOCATION", "COUNTY", "DATE_OF_DATA", "RESOLUTION",
        "UNITS_USED", "WATER_TYPE"
    ]

    df = pd.DataFrame(columns=columns)

    for f in os.listdir(root_json):
        filep = os.path.join(root_json, f)
        id = f.split('_')[0]
        page_num = f.split('_')[-1][:-5]
        print(id, page_num)
        input = ""
        
        with open(filep) as json_file:
            parsed = json.load(json_file)['result']['chunks']

        if parsed == []:
            continue
        else:
            for content in parsed:
                input += content['content']

        v1 = "Groundwater", "Stream Discharge", "Precipitation", "Springs", "Reservoir", "Irrigation", "Water Quality"

        v2 = "Groundwater well", "Groundwater recharge", "Stream discharge", "Precipitation", "Springs", "Groundwater water quality", "Spring water quality", "Stream water quality", "Irrigation", "Reservoir", "Other (Choose a water type that best describes the data)"

        v3 = "Groundwater", "Stream", "Precipitation", "Springs", "Reservoir", "Other (Choose a water type that best describes the data)" 
        
        r1_summarize = f"""
            Summarize the following metadata in the document below:
            For each Water Source in the document, 
            Name of the Water Source,
            Townships, ranges, and sections,
,           Location of the Water Source,
            Temporal Resolution of the data,
            Units of measurements used in the document,
            Latitude/longitude in decimal degrees (EPSG:4326),
            Dates that the Data was taken,
            and Water types from: {v1}. Double-check your answers.
        

            Here is the document:\n{input}\n
        
            """

        doc_summary = chat_with_model(key, r1_summarize, model_r1)['choices'][0]['message']['content'].split("</think>", 1)[-1]
        print(doc_summary)

        prompt = f"""
            You are an assistant that only responds with JSON objects in the following format:
            
            \n\n
            [
            
            \n

            {{
            \n \"LATITUDE\": \"...\", 
            \n \"LONGITUDE\": \"...\",
            \n \"TOWNSHIPS_RANGES_SECTIONS\": \"...\",
            \n \"WATERSOURCE_NAME\": \"...\",
            \n \"LOCATION\": \"...\", 
            \n \"COUNTY\": \"...\", 
            \n \"DATE_OF_DATA\": \"...\",
            \n \"RESOLUTION\": \"...\",  
            \n \"UNITS_USED\": \"...\",  
            \n \"WATER_TYPE\": \"...\"
            \n }},
            ... (repeat the format above for additional locations)
            ]
            \n\n
            Input "NA" with double quotes if information is not available. Do not fabricate values for Latitude and Longitude. RESOLUTION is the frequency of data (daily, monthly, yearly, etc).
            \n
            If there are multiple locations, respond with corresponding number of JSON objects.
            \n
            Select the WATER_TYPE that best fits one of these: [{v1}].
            \n
            Here is the input:\n{r1_summarize}\n
            
            \n
            
            Respond ONLY with valid JSON. Respond only with the Keys included in the format. Do not make up new keys. Make sure all values are double quoted (\"...\"). Do not include markdown or explanations.
            """

        res = chat_with_model(key, prompt, model)
        print(res['choices'][0]['message']['content'])
        print(type(res['choices'][0]['message']['content']))

        json_obj = "{0}".format(res['choices'][0]['message']['content'])
        cleaned_json = json_obj.replace("...", "").replace(" NA,", " \"NA\",")
        print(cleaned_json)
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
            except Exception as ex:
                print(ex)
    df.to_csv(out_csv)



