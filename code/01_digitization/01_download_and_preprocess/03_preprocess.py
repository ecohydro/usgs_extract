import argparse
import os
from multiprocessing import Pool, cpu_count

from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm


def convert_page(args):
    pdf_path, output_dir, page_num = args
    pub_id = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300, fmt="png")
    if not images:
        return None
    out_path = os.path.join(output_dir, f"{pub_id}_page_{page_num}.png")
    images[0].convert("L").save(out_path)  # convert to grayscale single-channel
    return out_path


def convert_pdf(pdf_path, output_dir):
    try:
        n_pages = len(PdfReader(pdf_path).pages)
    except Exception as e:
        print(f"Could not read {pdf_path}: {e}")
        return

    tasks = [(pdf_path, output_dir, i) for i in range(1, n_pages + 1)]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(convert_page, tasks), total=len(tasks), desc=os.path.basename(pdf_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert USGS PDFs to grayscale PNGs, one per page.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing input PDFs")
    parser.add_argument("--output-dir", required=True, help="Directory to save output PNGs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_files = [os.path.join(args.pdf_dir, f) for f in os.listdir(args.pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        convert_pdf(pdf_path, args.output_dir)
