"""
Asset downloader for PolyVision v2.0

Automatically downloads shared folders from Google Drive.
Configuration is read from assets_config.json.

Usage:
    python download_assets.py                            # Download all configured assets
    python download_assets.py Models/SEAMaP-Binary-Full  # Download specific asset
"""

import gdown
import json
import os
import sys


def load_config(config_file="assets_config.json"):
    """Load asset configuration from JSON file."""
    if not os.path.exists(config_file):
        print(f"✗ Configuration file not found: {config_file}")
        sys.exit(1)

    with open(config_file) as f:
        return json.load(f)


def is_configured(folder_url):
    """Check if a folder URL has a real Google Drive ID (not a placeholder)."""
    return not ("YOUR_" in folder_url)


def download_assets(asset_names=None):
    """Download asset folders from Google Drive.

    Args:
        asset_names: List of specific assets to download. If None, download all.
    """
    config = load_config()
    assets = config.get("assets", {})

    if not assets:
        print("✗ No assets defined in assets_config.json")
        return False

    if asset_names:
        assets = {k: v for k, v in assets.items() if k in asset_names}
        if not assets:
            print(f"✗ No matching assets found for: {asset_names}")
            return False

    skipped = []
    failed = []

    for dest, asset_info in assets.items():
        folder_url = asset_info.get("folder_url", "")
        description = asset_info.get("description", "")

        if not is_configured(folder_url):
            skipped.append((dest, description))
            continue

        print(f"\n📦 Downloading {dest}")
        if description:
            print(f"   ({description})")

        try:
            gdown.download_folder(folder_url, output=dest, quiet=False, use_cookies=False)
            print(f"✓ {dest} ready")

        except Exception as e:
            print(f"✗ Error downloading {dest}: {e}")
            failed.append(dest)

    # Summary
    print("\n" + "=" * 60)
    if failed:
        print(f"✗ Failed: {len(failed)} asset(s)")
        for asset in failed:
            print(f"  - {asset}")

    if skipped:
        print(f"⚠️  Skipped: {len(skipped)} asset(s) (not configured)")
        for asset, desc in skipped:
            print(f"  - {asset}")
            if desc:
                print(f"    {desc}")
        print(f"\n→ Edit assets_config.json and replace YOUR_* placeholders with folder IDs")

    if not failed and not skipped:
        print("✓ All assets downloaded successfully!")
        return True

    return len(failed) == 0


if __name__ == "__main__":
    asset_names = sys.argv[1:] if len(sys.argv) > 1 else None
    success = download_assets(asset_names)
    sys.exit(0 if success else 1)
