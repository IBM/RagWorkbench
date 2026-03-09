"""
Script to explore watsonxDocsQA dataset URLs and find their archived versions from 2021-2023.
Processes ALL URLs from the dataset with progress saving for resumability.
"""

import argparse
import json
import os
import random
import time
from threading import Lock

import requests
from datasets import load_dataset
from requests.exceptions import ConnectionError, RequestException, Timeout


def clean_url(url):
    """Remove query parameters from URL."""
    if "?" in url:
        return url.split("?")[0]
    return url


def get_wayback_snapshots(url, from_year=2021, to_year=2023, max_retries=2):
    """
    Get available snapshots from Wayback Machine for a given URL and date range.
    Returns the most recent snapshots first.
    Includes retry logic with exponential backoff for handling timeouts.

    Returns:
        tuple: (snapshots_list, status_string)
            - snapshots_list: List of snapshots or empty list
            - status_string: One of:
                - "success": Archive reached and snapshots found
                - "no_snapshots": Archive reached but no snapshots exist
                - "access_failed": Archive could not be reached (timeout/error)
    """
    # Wayback Machine CDX API
    cdx_api = "http://web.archive.org/cdx/search/cdx"

    params = {
        "url": url,
        "from": f"{from_year}0101",
        "to": f"{to_year}1231",
        "output": "json",
        "limit": 1,  # Get up to 1 snapshots from the date range
        "sort": "reverse",  # Get most recent snapshots first
    }

    for attempt in range(max_retries):
        try:
            # Increase timeout for each retry
            timeout = 30 + (attempt * 15)
            print(f"  Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")

            response = requests.get(cdx_api, params=params, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:  # First row is headers
                    return data[1:], "success"  # Return snapshots with success status
                else:
                    # Archive was reached but no snapshots exist
                    return [], "no_snapshots"
            else:
                # Non-200 status code
                return [], "access_failed"

        except (Timeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # Exponential backoff: 2s, 4s
                print(
                    f"  Connection timeout/error. Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {type(e).__name__}")
                return [], "access_failed"

        except RequestException as e:
            print(f"  ✗ Request error: {e}")
            return [], "access_failed"

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return [], "access_failed"

    return [], "access_failed"


def format_wayback_url(timestamp, original_url):
    """Format a Wayback Machine URL."""
    return f"https://web.archive.org/web/{timestamp}/{original_url}"


def process_single_url(idx, entry, print_lock):
    """
    Process a single URL entry. This function is thread-safe.
    Returns a result dictionary.
    """
    original_url = entry["url"]

    with print_lock:
        print(f"\n{'='*80}")
        print(f"Processing dataset index: {idx}")
        print(f"{'='*80}")
        print(f"Original URL: {original_url}")

    # Clean URL
    cleaned_url = clean_url(original_url)
    with print_lock:
        print(f"Cleaned URL:  {cleaned_url}")

    # Get Wayback Machine snapshots from 2021-2023
    with print_lock:
        print("\nSearching Wayback Machine for 2021-2023 snapshots...")

    snapshots, archive_status = get_wayback_snapshots(
        cleaned_url, from_year=2021, to_year=2023
    )

    # Determine which year was actually used from the snapshot timestamp
    year_used = None
    if snapshots:
        timestamp = snapshots[0][1]
        year_used = int(timestamp[:4])

    if snapshots:
        with print_lock:
            print(f"Found {len(snapshots)} snapshot(s) from {year_used}:")

        # Display all snapshots
        for i, snapshot in enumerate(snapshots, 1):
            timestamp = snapshot[1]
            status_code = snapshot[4]

            # Format timestamp for display
            year = timestamp[:4]
            month = timestamp[4:6]
            day = timestamp[6:8]
            hour = timestamp[8:10]
            minute = timestamp[10:12]
            second = timestamp[12:14]

            formatted_date = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            wayback_url = format_wayback_url(timestamp, cleaned_url)

            with print_lock:
                print(f"  {i}. Date: {formatted_date} | Status: {status_code}")
                print(f"     URL: {wayback_url}")

        # Select the first snapshot (most recent due to reverse sort)
        selected_snapshot = snapshots[0]
        selected_timestamp = selected_snapshot[1]
        selected_wayback_url = format_wayback_url(selected_timestamp, cleaned_url)

        with print_lock:
            print(f"\n✓ Selected snapshot from {year_used}: {selected_wayback_url}")

        return {
            "index": idx,
            "original_url": original_url,
            "cleaned_url": cleaned_url,
            "selected_wayback_url": selected_wayback_url,
            "timestamp": selected_timestamp,
            "total_snapshots": len(snapshots),
            "year": year_used,
            "archive_status": archive_status,
        }
    else:
        # Distinguish between no snapshots and access failure
        if archive_status == "no_snapshots":
            with print_lock:
                print("✗ Archive reached but no snapshots found from 2021-2023")
        else:  # access_failed
            with print_lock:
                print("✗ Failed to access web.archive.org (timeout/connection error)")

        return {
            "index": idx,
            "original_url": original_url,
            "cleaned_url": cleaned_url,
            "selected_wayback_url": None,
            "timestamp": None,
            "total_snapshots": 0,
            "year": None,
            "archive_status": archive_status,
        }


def load_progress(progress_file):
    """Load progress from JSON file if it exists."""
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            data = json.load(f)
            return set(data["processed_indices"]), data["results"]
    return set(), []


def save_progress(progress_file, processed_indices, results):
    """Save progress to JSON file."""
    with open(progress_file, "w") as f:
        json.dump(
            {"processed_indices": list(processed_indices), "results": results},
            f,
            indent=2,
        )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Explore watsonxDocsQA dataset URLs and find their archived versions from 2021-2023."
    )
    parser.add_argument(
        "--retry_access_failed",
        action="store_true",
        help="Retry URLs that previously failed to access web.archive.org",
    )
    args = parser.parse_args()

    progress_file = "watsonx_urls_progress.json"

    print("Loading watsonxDocsQA dataset (streaming mode)...")

    # Load the dataset in streaming mode to avoid memory issues
    dataset = load_dataset(
        "ibm-research/watsonxDocsQA", "corpus", split="train", streaming=True
    )

    # Load previous progress if exists
    processed_indices, results = load_progress(progress_file)

    if processed_indices:
        print(
            f"Resuming from previous run. Already processed {len(processed_indices)} URLs."
        )

    # Handle retry_access_failed flag
    if args.retry_access_failed:
        # Find indices with access_failed status
        failed_indices = {
            result["index"]
            for result in results
            if result["archive_status"] == "access_failed"
        }

        if failed_indices:
            print(
                f"\n--retry_access_failed flag detected: Will retry {len(failed_indices)} URLs that previously failed to access web.archive.org"
            )
            # Remove failed indices from processed set so they'll be retried
            processed_indices -= failed_indices
            # Remove failed results from results list
            results = [
                result
                for result in results
                if result["archive_status"] != "access_failed"
            ]
        else:
            print(
                "\n--retry_access_failed flag detected but no failed URLs found to retry"
            )

    # Collect all entries
    print("Collecting all dataset entries...")
    all_entries = []
    total_count = 0

    for entry in dataset:
        all_entries.append(entry)
        total_count += 1
        if total_count % 1000 == 0:
            print(f"  Collected {total_count} entries...")

    print(f"\nTotal entries in dataset: {len(all_entries)}")
    print(f"Already processed: {len(processed_indices)}")
    print(f"Remaining: {len(all_entries) - len(processed_indices)}\n")

    # Create shuffled index list for random processing order
    shuffled_indices = list(range(len(all_entries)))
    random.shuffle(shuffled_indices)
    print("Entries will be processed in random order\n")

    print_lock = Lock()  # For consistency with the function signature

    # Process all URLs in random order
    for idx in shuffled_indices:
        if idx in processed_indices:
            continue  # Skip already processed

        entry = all_entries[idx]

        print(
            f"\n[{idx+1}/{len(all_entries)}] (Completed: {len(processed_indices)}) ",
            end="",
        )
        result = process_single_url(idx, entry, print_lock)
        results.append(result)
        processed_indices.add(idx)

        # Save progress every 5 URLs
        if len(processed_indices) % 5 == 0:
            save_progress(progress_file, processed_indices, results)
            print(
                f"\n  ✓ Progress saved ({len(processed_indices)}/{len(all_entries)} completed)"
            )

        # Add delay between requests to be respectful to the API
        if idx < len(all_entries) - 1:
            wait_time = 2
            print(f"\nWaiting {wait_time} seconds before next request...")
            time.sleep(wait_time)

    # Final save
    save_progress(progress_file, processed_indices, results)
    print(f"\n\n✓ All URLs processed! Final progress saved to {progress_file}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for result in results:
        print(f"Entry (Dataset index {result['index']}):")
        print(f"  Original: {result['original_url']}")
        print(f"  Cleaned:  {result['cleaned_url']}")
        if result["selected_wayback_url"]:
            print(f"  Archive:  {result['selected_wayback_url']}")
            print(f"  Year:     {result['year']}")
            print(
                f"  (Found {result['total_snapshots']} snapshots from {result['year']})"
            )
            print(f"  Status:   {result['archive_status']}")
        else:
            print("  Archive:  Not found")
            print(f"  Status:   {result['archive_status']}")
            if result["archive_status"] == "no_snapshots":
                print("  Reason:   Archive reached but no snapshots exist")
            elif result["archive_status"] == "access_failed":
                print("  Reason:   Failed to access web.archive.org")
        print()


if __name__ == "__main__":
    main()

# Made with Bob
