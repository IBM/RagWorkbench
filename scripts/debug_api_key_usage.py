#!/usr/bin/env python3
"""
Debug tool: Print usage data for a specific virtual API key.

This script queries the LiteLLM proxy and prints detailed usage statistics
for a given API key.

Usage:
    python scripts/debug_api_key_usage.py

Configuration:
    1. Set LITELLM_MASTER_KEY environment variable
    2. Update the API_KEY and LITELLM_PROXY_URL variables below
"""

import asyncio
import logging
import os
import sys

# Add src to path to import ragworkbench modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragworkbench.eval.cost_tracking import CostTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# CONFIGURATION: Update these with your values
# ============================================================
API_KEY = "your-api-key-goes-here"
LITELLM_PROXY_URL = "http://localhost:4000"
# ============================================================


async def main():
    """Main function to query and print API key usage."""
    # Verify master key is set
    master_key = os.getenv("LITELLM_MASTER_KEY")
    if not master_key:
        print("ERROR: LITELLM_MASTER_KEY environment variable must be set.")
        print("Please set it with your LiteLLM proxy master key.")
        sys.exit(1)

    # Create tracker and manually set the API key
    tracker = CostTracker(enabled=True, litellm_proxy_url=LITELLM_PROXY_URL)
    tracker.api_key = API_KEY

    print(f"\n{'='*70}")
    print(f"Querying usage data for API key: {API_KEY[:30]}...")
    print(f"LiteLLM Proxy URL: {LITELLM_PROXY_URL}")
    print(f"{'='*70}\n")

    try:
        # Retrieve usage data
        usage_data = await tracker.get_usage_data()

        # Print detailed usage information
        print("USAGE DATA SUMMARY")
        print(f"{'='*70}")
        print(f"API Key:            {usage_data.api_key[:30]}...")
        print(f"Total Cost:         ${usage_data.total_cost:.6f}")
        print(f"Total Tokens:       {usage_data.total_tokens:,}")
        print(f"  - Prompt Tokens:  {usage_data.prompt_tokens:,}")
        print(f"  - Completion:     {usage_data.completion_tokens:,}")
        print(f"Total Requests:     {usage_data.requests}")
        print(
            f"Models Used:        {', '.join(usage_data.models_used) if usage_data.models_used else 'None'}"
        )
        print(f"{'='*70}\n")

        # Additional statistics
        if usage_data.requests > 0:
            avg_cost_per_request = usage_data.total_cost / usage_data.requests
            avg_tokens_per_request = usage_data.total_tokens / usage_data.requests
            print("AVERAGES PER REQUEST")
            print(f"{'='*70}")
            print(f"Avg Cost:           ${avg_cost_per_request:.6f}")
            print(f"Avg Tokens:         {avg_tokens_per_request:.2f}")
            print(f"{'='*70}\n")

        # Success message
        print("✓ Usage data retrieved successfully")

    except Exception as e:
        print(f"\n✗ Error retrieving usage data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
