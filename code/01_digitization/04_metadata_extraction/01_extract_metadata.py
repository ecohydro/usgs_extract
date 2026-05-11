import argparse
import json
import os
import re
import sys

import pandas as pd
import requests


COLUMNS = [
    "ID", "PAGE_NUMBER", "Inferred_Latitude", "Inferred_Longitude",
    "Actual_Latitude", "Actual_Longitude", "Location", "Townships_Ranges_Sections",
    "Watersource_Name", "County", "Dates_of_Recording", "Temporal_Resolution",
    "Units_Of_Measurement", "Water_Type", "KeyTerms",
]

WATER_TYPES = [
    "Groundwater", "Stream Discharge", "Precipitation", "Springs",
    "Reservoir", "Irrigation", "Water Quality", "Not Water Related",
]


def chat_with_model(api_url, token, prompt, model):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def build_prompt(page_text):
    return f"""
        Reply only with valid JSON format,

        {{
        "Watersource_Name": ...,
        "County": ...,
        "Location": ...,
        "Townships_Ranges_Sections": ... ("[TRS, TRS, ...]"),
        "Temporal_Resolution": ... (time interval between sequential measurements),
        "Units_Of_Measurement": ...,
        "Actual_Latitude": ... (number in decimals),
        "Actual_Longitude": ... (number in decimals),
        "Inferred_Latitude": ... (number in decimals),
        "Inferred_Longitude": ... (number in decimals),
        "Dates_of_Recording": ... (MM/DD/YYYY-MM/DD/YYYY),
        "Water_Type": {WATER_TYPES},
        "KeyTerms": ... (terms in the document that correspond to the Water_Type)
        }}

        If the document is not related to water data, DO NOT return a JSON.

        There may be multiple watersources. In the case that multiple water sources are present, list out each water source individually. In the case that no watersource is present, do not return a JSON output.
        All data is from the State of California. If there is no latitude and longitude data in the document, use inference to locate the watersource and record the results as "Inferred_Latitude" and "Inferred_Longitude"; do not make comments with \\\\.
        Some documents may have tables with column names that are misleading or incorrect. Double check the given categories.
        Double-check your answers. Make sure output is compatible with json.loads(), with commas separating each key-value pair and each JSON object. Do not give examples or comments inside the JSON.

        Here is the document content:\n{page_text}\n
        """


def parse_response(raw):
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if not match:
        return None
    cleaned = match.group(1)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = "".join(cleaned.rsplit(",", 1)[0]) + "]"
        return json.loads(cleaned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract metadata from Reducto per-page JSONs using a UCSB HPC LLM."
    )
    parser.add_argument("--json-dir", required=True, help="Directory of per-page Reducto JSON files")
    parser.add_argument("--output-csv", required=True, help="Path to write (or append to) output CSV")
    parser.add_argument("--model", default="phi4:latest", help="Model name (default: phi4:latest)")
    parser.add_argument(
        "--api-url",
        default="https://llm.grit.ucsb.edu/api/chat/completions",
        help="LLM API endpoint (default: UCSB HPC)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("UCSB_LLM_API_KEY")
    if not api_key:
        sys.exit("Error: UCSB_LLM_API_KEY environment variable not set.")

    done = set()
    write_header = not os.path.exists(args.output_csv)
    if not write_header:
        existing = pd.read_csv(args.output_csv, usecols=["ID", "PAGE_NUMBER"])
        done = set(zip(existing["ID"].astype(str), existing["PAGE_NUMBER"].astype(str)))
        print(f"Skipping {len(done)} already-processed pages")

    json_files = sorted(f for f in os.listdir(args.json_dir) if f.endswith(".json"))
    print(f"Found {len(json_files)} JSON files")

    for fname in json_files:
        doc_id = fname.split("_")[0]
        page_num = os.path.splitext(fname)[0].split("_")[-1]

        if (doc_id, page_num) in done:
            continue

        fpath = os.path.join(args.json_dir, fname)
        with open(fpath) as f:
            chunks = json.load(f)["result"]["chunks"]

        if not any(
            block["type"] == "Table"
            for chunk in chunks
            for block in chunk["blocks"]
        ):
            continue

        page_text = "".join(chunk["content"] for chunk in chunks)

        try:
            raw = chat_with_model(args.api_url, api_key, build_prompt(page_text), args.model)
            raw = raw["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API error on {fname}: {e}")
            continue

        try:
            parsed = parse_response(raw)
        except Exception as e:
            print(f"Parse error on {fname}: {e}")
            error_path = os.path.join(args.json_dir, f"{doc_id}_{page_num}_error.txt")
            with open(error_path, "w") as ef:
                ef.write(raw)
            continue

        if parsed is None:
            continue

        rows = parsed if isinstance(parsed, list) else [parsed]
        df = pd.DataFrame(rows)
        df.insert(0, "ID", doc_id)
        df.insert(1, "PAGE_NUMBER", page_num)
        df = df.reindex(columns=COLUMNS)

        df.to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False
        print(f"{fname}: {len(rows)} row(s)")
