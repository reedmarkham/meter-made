#!/usr/bin/env python3
"""
Script to fetch DC parking violations data from the DC GIS REST service.
Downloads all parking violations from 2010 to 2025 using the DC GIS MapServer endpoints
for the respective years.
"""

from pathlib import Path
from typing import List
import logging, json

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DC GIS REST service base URL pattern
DC_GIS_BASE = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA"

# Mapping of years to their MapServer service names and layer IDs
# These follow the pattern: Violations_Parking_[YEAR]/MapServer/[LAYER_ID]
PARKING_VIOLATIONS_SERVICES = {
    2018: {"service": "Violations_Parking_2018", "layer": 10},
    2019: {"service": "Violations_Parking_2019", "layer": 10},
    2020: {"service": "Violations_Parking_2020", "layer": 10},
    2021: {"service": "Violations_Parking_2021", "layer": 10},
    2022: {"service": "Violations_Parking_2022", "layer": 10},
    2023: {"service": "Violations_Parking_2023", "layer": 10},
    2024: {"service": "Violations_Parking_2024", "layer": 10},
    2025: {"service": "Violations_Parking_2025", "layer": 10},
    # For years before 2018, services may not exist or have different names
    # We'll attempt to query them with default patterns
}

# Years to attempt fetching
YEARS = list(range(2010, 2026))


def fetch_from_rest_service(service: str, layer: int, year: int, limit: int = 10000) -> List[dict]:
    """
    Fetch parking violations from a DC GIS REST service.

    Args:
        service: MapServer service name
        layer: Layer ID within the service
        year: Year being fetched (for logging)
        limit: Maximum number of records to fetch (REST services have limits)

    Returns:
        List of violation records
    """
    try:
        url = f"{DC_GIS_BASE}/{service}/MapServer/{layer}/query"

        params = {
            "where": "1=1",  # Get all records
            "outFields": "*",  # Return all fields
            "outSR": 4326,  # Output spatial reference (WGS84 lat/lng)
            "f": "json",  # Response format
            "resultRecordCount": limit
        }

        logger.info(f"Fetching {year} violations from {service}/MapServer/{layer}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for errors in the response
        if "error" in data:
            logger.error(f"API Error for {year}: {data['error'].get('message', 'Unknown error')}")
            return []

        features = data.get("features", [])
        logger.info(f"Retrieved {len(features)} records for {year}")
        return features

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {year} violations: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response for {year}: {e}")
        return []


def fetch_parking_violations_by_year(year: int) -> List[dict]:
    """
    Fetch parking violations for a specific year.

    Args:
        year: Year to fetch

    Returns:
        List of violation records
    """
    if year in PARKING_VIOLATIONS_SERVICES:
        config = PARKING_VIOLATIONS_SERVICES[year]
        return fetch_from_rest_service(config["service"], config["layer"], year)
    else:
        # Try default naming pattern for years not explicitly mapped
        service_name = f"Violations_Parking_{year}"
        logger.info(f"Attempting to fetch {year} using default service name: {service_name}")
        return fetch_from_rest_service(service_name, 10, year)


def save_violations_to_json(violations: List[dict], year: int, output_dir: Path):
    """
    Save parking violations to a JSON file.

    Args:
        violations: List of violation records
        year: Year the violations are from
        output_dir: Directory to save the file in
    """
    if not violations:
        logger.warning(f"No violations to save for {year}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"parking_violations_{year}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(violations, f, indent=2, default=str)
        logger.info(f"Saved {len(violations)} violations to {output_file}")
    except IOError as e:
        logger.error(f"Error saving violations to {output_file}: {e}")


def save_violations_to_csv(violations: List[dict], year: int, output_dir: Path):
    """
    Save parking violations to a CSV file.

    Args:
        violations: List of violation records
        year: Year the violations are from
        output_dir: Directory to save the file in
    """
    try:
        import pandas as pd
    except ImportError:
        logger.debug("pandas not installed, skipping CSV export")
        return

    if not violations:
        logger.warning(f"No violations to save for {year}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"parking_violations_{year}.csv"

    try:
        # Extract attributes from GeoJSON-like features
        records = []
        for feature in violations:
            record = feature.get("attributes", {})
            if "geometry" in feature:
                # Add geometry coordinates if present
                geom = feature["geometry"]
                if "x" in geom and "y" in geom:
                    record["longitude"] = geom["x"]
                    record["latitude"] = geom["y"]
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(violations)} violations to {output_file}")
    except Exception as e:
        logger.error(f"Error saving CSV for {year}: {e}")


def main():
    """Main function to fetch all DC parking violations."""

    # Output directory - write to data/
    output_dir = Path(__file__).parent.parent / "data"

    logger.info("=" * 60)
    logger.info("DC Parking Violations Data Fetcher")
    logger.info("=" * 60)

    all_violations = {}
    successful_years = []
    failed_years = []

    # Fetch data for each year
    for year in YEARS:
        logger.info(f"\nFetching data for {year}...")

        violations = fetch_parking_violations_by_year(year)

        if violations:
            all_violations[year] = violations
            successful_years.append(year)

            # Save individual year files
            save_violations_to_json(violations, year, output_dir)
            save_violations_to_csv(violations, year, output_dir)
        else:
            failed_years.append(year)

    # Save combined data
    if all_violations:
        combined_file = output_dir / "all_parking_violations.json"
        try:
            with open(combined_file, 'w') as f:
                json.dump(all_violations, f, indent=2, default=str)
            logger.info(f"\nSaved combined data to {combined_file}")
        except IOError as e:
            logger.error(f"Error saving combined violations: {e}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Data Fetch Summary")
        logger.info("=" * 60)
        for year in sorted(all_violations.keys()):
            violations = all_violations[year]
            logger.info(f"Year {year}: {len(violations)} violations")
        total = sum(len(v) for v in all_violations.values())
        logger.info(f"Total: {total} violations")

        if failed_years:
            logger.info(f"\nYears with no data: {', '.join(map(str, failed_years))}")
    else:
        logger.warning("No data was fetched. Check service availability.")

    logger.info("\n" + "=" * 60)
    logger.info("Fetch complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
