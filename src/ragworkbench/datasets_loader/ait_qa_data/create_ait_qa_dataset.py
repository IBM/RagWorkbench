"""Script to download AIT QA dataset PDFs from archive.org.

This script creates the necessary folder structure and downloads annual report PDFs
from various airlines (Alaska, American Airlines, Delta, Southwest, United) for the
years 2017-2019. The PDFs are sourced from archive.org's Wayback Machine.

The script will:
1. Create an 'ait_qa_pdf' folder in the main project directory
2. Create a 'documents' subfolder within it
3. Download all specified PDFs to the documents folder

Usage:
    python -m src.openrag_eval.data_loaders.create_ait_qa_dataset
"""

import logging
import time
import warnings

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ragworkbench.datasets_loader.ait_qa_data.config import (
    get_ait_qa_data_dir,
    get_ait_qa_documents_dir,
)

# Suppress SSL warnings when verify=False is used
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define URL to output filename mapping for all AIT QA dataset PDFs
# This dictionary is used both for downloading and for validation
URL_TO_OUTPUT_FILE: dict[str, str] = {
    "https://web.archive.org/web/20220629112311/https://www.annualreports.com/HostedData/AnnualReportArchive/a/NYSE_ALK_2017.pdf": "Alaska-2017.pdf",
    "https://web.archive.org/web/20220629112336/https://www.annualreports.com/HostedData/AnnualReportArchive/a/NYSE_ALK_2018.pdf": "Alaska-2018.pdf",
    "https://web.archive.org/web/20240626083628/https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAL_2017.pdf": "AmericanAirlines-2017.pdf",
    # "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAL_2017.pdf": "AmericanAirlines-2017.pdf.bck",
    "https://web.archive.org/web/20240802123343/https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAL_2018.pdf": "AmericanAirlines-2018.pdf",
    "https://web.archive.org/web/20240626051553/https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAL_2019.pdf": "AmericanAirlines-2019.pdf",
    "https://web.archive.org/web/20250726023255/https://www.annualreports.com/HostedData/AnnualReportArchive/d/NYSE_DAL_2017.pdf": "Delta-2017.pdf",
    "https://web.archive.org/web/20250913060612/https://www.annualreports.com/HostedData/AnnualReportArchive/d/NYSE_DAL_2018.pdf": "Delta-2018.pdf",
    "https://web.archive.org/web/20250913060615/https://www.annualreports.com/HostedData/AnnualReportArchive/d/NYSE_DAL_2019.pdf": "Delta-2019.pdf",
    "https://web.archive.org/web/20211204111318/https://www.annualreports.com/HostedData/AnnualReportArchive/s/NYSE_LUV_2017.pdf": "Southwest-2017.pdf",
    "https://web.archive.org/web/20260303211647/https://www.annualreports.com/HostedData/AnnualReportArchive/s/NYSE_LUV_2018.pdf": "Southwest-2018.pdf",
    "https://web.archive.org/web/20211204101822/https://www.annualreports.com/HostedData/AnnualReportArchive/s/NYSE_LUV_2019.pdf": "Southwest-2019.pdf",
    "https://web.archive.org/web/20250716011902/https://www.annualreports.com/HostedData/AnnualReportArchive/u/NYSE_UAL_2017.pdf": "United-2017.pdf",
    "https://web.archive.org/web/20250716020222/https://www.annualreports.com/HostedData/AnnualReportArchive/u/NYSE_UAL_2018.pdf": "United-2018.pdf",
    "https://web.archive.org/web/20250717033806/https://www.annualreports.com/HostedData/AnnualReportArchive/u/NYSE_UAL_2019.pdf": "United-2019.pdf",
}


def create_session_with_retries() -> requests.Session:
    """Create a requests session with retry logic and SSL handling.

    Returns:
        A configured requests.Session object with retry logic.
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=2,  # Wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["GET", "HEAD"],  # Methods to retry
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set a reasonable timeout and user agent
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    return session


def create_ait_qa_dataset() -> None:
    """Create folder structure and download AIT QA dataset PDFs.

    This function creates the necessary directories and downloads all annual report
    PDFs from the specified URLs. Each PDF is saved with its corresponding filename
    as defined in the url_to_output_file dictionary.

    The folder structure created:
        ait_qa_pdf/
            documents/
                Alaska-2017.pdf
                Alaska-2018.pdf
                ... (other PDFs)

    Raises:
        Exception: If any download fails, logs the error and continues with remaining files.
    """
    # Step 1: Create folder structure using shared configuration
    # This ensures both the downloader and data loader use the same location
    ait_qa_pdf_folder = get_ait_qa_data_dir()
    ait_qa_pdf_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified folder: {ait_qa_pdf_folder}")

    # Create documents subfolder
    documents_folder = get_ait_qa_documents_dir()
    documents_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified folder: {documents_folder}")

    # Step 2: Download PDFs
    total_files = len(URL_TO_OUTPUT_FILE)
    successful_downloads = 0
    failed_downloads = 0

    logger.info(f"Starting download of {total_files} PDF files...")

    # Create session with retry logic
    session = create_session_with_retries()

    for idx, (url, filename) in enumerate(URL_TO_OUTPUT_FILE.items(), 1):
        output_path = documents_folder / filename

        # Skip if file already exists
        if output_path.exists():
            logger.info(f"[{idx}/{total_files}] Skipping {filename} (already exists)")
            successful_downloads += 1
            continue

        # Try downloading with multiple attempts
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    f"[{idx}/{total_files}] Downloading {filename} (attempt {attempt}/{max_attempts})..."
                )

                # Use session with timeout and verify=False to handle SSL issues
                response = session.get(url, stream=True, timeout=60, verify=False)
                response.raise_for_status()

                # Write to file
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"[{idx}/{total_files}] Successfully downloaded {filename}")
                successful_downloads += 1
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < max_attempts:
                    wait_time = 5 * attempt  # Exponential backoff: 5, 10, 15 seconds
                    logger.warning(
                        f"[{idx}/{total_files}] Attempt {attempt} failed for {filename}: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"[{idx}/{total_files}] All attempts failed for {filename}: {e}"
                    )
                    failed_downloads += 1
                    # Clean up partial download if it exists
                    if output_path.exists():
                        output_path.unlink()

    # Summary
    logger.info("=" * 60)
    logger.info("Download Summary:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Successful: {successful_downloads}")
    logger.info(f"  Failed: {failed_downloads}")
    logger.info(f"  Output directory: {documents_folder}")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_ait_qa_dataset()
