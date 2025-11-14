#!/usr/bin/env python3
"""
Extract parking ticket coordinates and datetime components using DuckDB.
Outputs a CSV file with: x, y, hour, minute, day, month, year
"""

import duckdb
from pathlib import Path

def extract_ticket_data(data_dir: str, output_file: str = "tickets_extracted.csv"):
    """
    Extract ticket coordinates and datetime components from parking violations data.

    Args:
        data_dir: Path to the directory containing parking violations CSV files
        output_file: Path to save the extracted data as CSV
    """

    # Connect to DuckDB
    conn = duckdb.connect()

    # Path pattern for all parking violations CSV files
    all_files_pattern = f"{data_dir}/parking_violations_*.csv"

    print("Loading parking violations data...")
    # Create a temporary table with all violations data
    conn.execute(f"""
        CREATE TEMP TABLE violations_raw AS
        SELECT * FROM read_csv_auto('{all_files_pattern}')
    """)

    print("Extracting ticket coordinates and datetime components...")

    # Extract the desired fields and save to a new table
    # ISSUE_TIME is in format like "09:35 AM" or "12:16 PM", ISSUE_DATE is Unix timestamp in milliseconds
    conn.execute("""
        CREATE TEMP TABLE tickets_extracted AS
        SELECT
            LONGITUDE as x,
            LATITUDE as y,
            CASE
                WHEN ISSUE_TIME IS NULL THEN NULL
                WHEN UPPER(ISSUE_TIME) LIKE '%AM' THEN
                    CASE
                        WHEN TRY_CAST(TRIM(SUBSTRING(ISSUE_TIME, 1, POSITION(':' IN ISSUE_TIME) - 1)) AS INT) = 12 THEN 0
                        ELSE COALESCE(TRY_CAST(TRIM(SUBSTRING(ISSUE_TIME, 1, POSITION(':' IN ISSUE_TIME) - 1)) AS INT), 0)
                    END
                ELSE
                    CASE
                        WHEN TRY_CAST(TRIM(SUBSTRING(ISSUE_TIME, 1, POSITION(':' IN ISSUE_TIME) - 1)) AS INT) = 12 THEN 12
                        ELSE COALESCE(TRY_CAST(TRIM(SUBSTRING(ISSUE_TIME, 1, POSITION(':' IN ISSUE_TIME) - 1)) AS INT), 0) + 12
                    END
            END as hour,
            CASE
                WHEN ISSUE_TIME IS NULL THEN NULL
                ELSE COALESCE(TRY_CAST(SUBSTRING(ISSUE_TIME, POSITION(':' IN ISSUE_TIME) + 1, 2) AS INT), 0)
            END as minute,
            DAY(to_timestamp(ISSUE_DATE / 1000)) as day,
            MONTH(to_timestamp(ISSUE_DATE / 1000)) as month,
            YEAR(to_timestamp(ISSUE_DATE / 1000)) as year
        FROM violations_raw
        WHERE LONGITUDE IS NOT NULL
            AND LATITUDE IS NOT NULL
            AND ISSUE_DATE IS NOT NULL
    """)

    # Get statistics
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_tickets,
            COUNT(DISTINCT year) as years_covered,
            MIN(year) as earliest_year,
            MAX(year) as latest_year
        FROM tickets_extracted
    """).fetchall()

    total_tickets, years_covered, earliest_year, latest_year = stats[0]

    print(f"\nExtraction complete:")
    print(f"  Total tickets: {total_tickets:,}")
    print(f"  Years covered: {years_covered}")
    print(f"  Date range: {int(earliest_year)} - {int(latest_year)}")

    # Display first 10 records
    print("\nFirst 10 records:")
    result = conn.execute("""
        SELECT x, y, hour, minute, day, month, year
        FROM tickets_extracted
        LIMIT 10
    """).fetchall()

    print(f"{'x':<12} {'y':<12} {'hour':<6} {'minute':<8} {'day':<5} {'month':<7} {'year':<6}")
    print("-" * 60)
    for row in result:
        x = f"{row[0]:.5f}" if row[0] is not None else "None"
        y = f"{row[1]:.5f}" if row[1] is not None else "None"
        hour = f"{row[2]}" if row[2] is not None else "None"
        minute = f"{row[3]}" if row[3] is not None else "None"
        day = f"{row[4]}" if row[4] is not None else "None"
        month = f"{row[5]}" if row[5] is not None else "None"
        year = f"{row[6]}" if row[6] is not None else "None"
        print(f"{x:<12} {y:<12} {hour:<6} {minute:<8} {day:<5} {month:<7} {year:<6}")

    # Export to CSV
    print(f"\nExporting to {output_file}...")
    conn.execute(f"""
        COPY tickets_extracted
        TO '{output_file}'
        (FORMAT CSV, HEADER TRUE)
    """)

    print(f"✓ Data successfully exported to {output_file}")

    conn.close()

    return output_file


if __name__ == "__main__":
    data_directory = "/Users/reedmarkham/github/meter-made/data/parking_violations"
    output_csv = "/Users/reedmarkham/github/meter-made/tickets_extracted.csv"

    print("="*60)
    print("DuckDB Parking Tickets Extraction")
    print("="*60)
    print(f"Data directory: {data_directory}\n")

    extract_ticket_data(data_directory, output_csv)
