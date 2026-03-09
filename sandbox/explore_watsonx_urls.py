"""
Script to explore watsonxDocsQA dataset URLs and find their 2023 archived versions.
Processes ALL URLs from the dataset with progress saving for resumability.
"""

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


def get_wayback_snapshots(url, year=2023, max_retries=3):
    """
    Get available snapshots from Wayback Machine for a given URL and year.
    Includes retry logic with exponential backoff for handling timeouts.
    """
    # Wayback Machine CDX API
    cdx_api = "http://web.archive.org/cdx/search/cdx"

    params = {
        "url": url,
        "from": f"{year}0101",
        "to": f"{year}1231",
        "output": "json",
        "limit": 1,  # Get up to 1 snapshot from the year
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
                    return data[1:]  # Return all snapshots except header
            return []

        except (Timeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(
                    f"  Connection timeout/error. Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {type(e).__name__}")
                return []

        except RequestException as e:
            print(f"  ✗ Request error: {e}")
            return []

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return []

    return []


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

    # Get Wayback Machine snapshots from 2023, fallback to 2022
    with print_lock:
        print("\nSearching Wayback Machine for 2023 snapshots...")

    snapshots = get_wayback_snapshots(cleaned_url, year=2023)
    year_used = 2023

    if not snapshots:
        with print_lock:
            print("No 2023 snapshots found. Trying 2022...")
        snapshots = get_wayback_snapshots(cleaned_url, year=2022)
        year_used = 2022

    if not snapshots:
        with print_lock:
            print("No 2022-2023 snapshots found. Trying 2021...")
        snapshots = get_wayback_snapshots(cleaned_url, year=2021)
        year_used = 2021

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

        # Select the first snapshot (or middle one for variety)
        selected_snapshot = (
            snapshots[len(snapshots) // 2] if len(snapshots) > 1 else snapshots[0]
        )
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
        }
    else:
        with print_lock:
            print("✗ No snapshots found from 2023 or 2022")

        return {
            "index": idx,
            "original_url": original_url,
            "cleaned_url": cleaned_url,
            "selected_wayback_url": None,
            "timestamp": None,
            "total_snapshots": 0,
            "year": None,
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
    progress_file = "sandbox/watsonx_urls_progress.json"

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

        # Save progress every 10 URLs
        if len(processed_indices) % 10 == 0:
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
        else:
            print("  Archive:  Not found")
        print()


if __name__ == "__main__":
    main()

# Made with Bob
