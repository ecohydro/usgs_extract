import argparse
import asyncio
import json
import os
import sys

import aiohttp


async def process_file(session, api_key, png_path, json_path):
    try:
        async with session.post(
            "https://platform.reducto.ai/upload",
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        with open(png_path, "rb") as f:
            await session.put(data["presigned_url"], data=f, headers={"Content-Type": ""})

        async with session.post(
            "https://platform.reducto.ai/parse",
            json={"document_url": data["file_id"]},
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved: {json_path}")

    except Exception as e:
        with open("error_log.txt", "a") as log:
            log.write(f"Error processing {png_path}: {e}\n")
        print(f"Error: {png_path}: {e}")


async def run(png_dir, output_dir, api_key, batch_size):
    os.makedirs(output_dir, exist_ok=True)
    png_files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))

    tasks = []
    async with aiohttp.ClientSession() as session:
        for fname in png_files:
            json_name = os.path.splitext(fname)[0] + ".json"
            json_path = os.path.join(output_dir, json_name)
            if os.path.exists(json_path):
                continue
            tasks.append(process_file(session, api_key, os.path.join(png_dir, fname), json_path))

        print(f"Processing {len(tasks)} files (skipping {len(png_files) - len(tasks)} already done)")
        for i in range(0, len(tasks), batch_size):
            await asyncio.gather(*tasks[i:i + batch_size])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send page PNGs to the Reducto.ai API and save per-page JSON output."
    )
    parser.add_argument("--png-dir", required=True, help="Directory of full-page PNGs")
    parser.add_argument("--output-dir", required=True, help="Directory to save Reducto JSON output")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Concurrent requests per batch (default: 50)")
    args = parser.parse_args()

    api_key = os.environ.get("REDUCTO_API_KEY")
    if not api_key:
        sys.exit("Error: REDUCTO_API_KEY environment variable not set.")

    asyncio.run(run(args.png_dir, args.output_dir, api_key, args.batch_size))
