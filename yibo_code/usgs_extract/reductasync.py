import asyncio
import aiohttp
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

key = os.getenv("API_KEY")

if not key:
    print("Error: API_KEY not found in environment variables.")
    exit(1)


async def process_file(session, path, j_path):
    try:
        # Get presigned URL
        async with session.post(
            "https://platform.reducto.ai/upload",
            headers={"Authorization": f"Bearer {key}"}
        ) as response:
            response.raise_for_status()
            response_data = await response.json()
            presigned_url = response_data["presigned_url"]
            file_id = response_data["file_id"]

        # Upload the file
        with open(path, "rb") as file_data:
            await session.put(
                presigned_url,
                data=file_data,
                headers={"Content-Type": ""}
            )

        # Parse the document
        async with session.post(
            "https://platform.reducto.ai/parse",
            json={"document_url": file_id},
            headers={"Authorization": f"Bearer {key}"}
        ) as response:
            response.raise_for_status()
            json_object = await response.json()

            # Save JSON output
            with open(j_path, "w") as f:
                json.dump(json_object, f, indent=4)

    except Exception as e:
        # Log individual file errors
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Error processing file {path}: {str(e)}\n")
        print(f"Error processing file {path}: {e}")


async def process_documents():
    root = "/home/waves/data/usgs_extract/UpdatedDataJan"
    json_root = "/home/waves/data/usgs_extract/ReductJson"
    docs = os.listdir(root)

    tasks = []  # Collect all tasks here

    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        for doc in docs:
            doc_path = os.path.join(root, doc)
            pages = os.listdir(doc_path)
            json_path = os.path.join(json_root, doc)

            if not os.path.exists(json_path):
                os.makedirs(json_path)

            for page in pages:
                page_path = os.path.join(doc_path, page)
                json_page_path = os.path.join(json_path, page[:-4] + ".json")

                # Skip already parsed files
                if not os.path.exists(json_page_path):
                    tasks.append(process_file(session, page_path, json_page_path))

        # Run all tasks in batches
        if tasks:
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                await asyncio.gather(*tasks[i:i+batch_size])


if __name__ == "__main__":
    asyncio.run(process_documents())
