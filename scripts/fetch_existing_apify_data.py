
import sys
import os

from dotenv import load_dotenv
from apify_client import ApifyClient

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.scraper.apify_macro import process_macro_events
from src.data.database import init_db, MacroEvent


# ============================================================
# IMPORTANT:
# This is the EXISTING successful Apify run.
# This script ONLY reads its existing dataset.
# It DOES NOT start a new Actor run.
# ============================================================

RUN_ID = "XGTXRdSYwJslAgaLt"


def main():

    # --------------------------------------------------------
    # 1. Load Apify token
    # --------------------------------------------------------
    load_dotenv()

    token = os.getenv("APIFY_API_TOKEN")

    if not token:
        print("Error: APIFY_API_TOKEN is missing in .env.")
        sys.exit(1)

    # --------------------------------------------------------
    # 2. Initialize local database
    # --------------------------------------------------------
    print("Initializing database...")

    db_path = "sqlite:///data/macro_trading.db"

    Session = init_db(db_path)
    session = Session()

    # --------------------------------------------------------
    # 3. Connect to Apify
    # --------------------------------------------------------
    print("Connecting to Apify...")

    client = ApifyClient(token)

    # --------------------------------------------------------
    # 4. Open EXISTING run
    # --------------------------------------------------------
    print(
        f"Opening existing Apify run: {RUN_ID}"
    )

    run_client = client.run(RUN_ID)

    run_info = run_client.get()

    if run_info is None:
        raise RuntimeError(
            f"Apify run {RUN_ID} was not found."
        )

    # Your installed client returns a Run object.
    status = getattr(run_info, "status", None)

    print(
        f"Existing run status: {status}"
    )

    if status != "SUCCEEDED":
        raise RuntimeError(
            f"Existing Apify run did not succeed. "
            f"Status: {status}"
        )

    print(
        "Existing Apify run is successful."
    )

    print(
        "NO new Actor run will be started."
    )

    # --------------------------------------------------------
    # 5. Get the EXISTING dataset ID
    # --------------------------------------------------------
    dataset_id = getattr(
        run_info,
        "default_dataset_id",
        None
    )

    if not dataset_id:
        raise RuntimeError(
            "Could not find default dataset ID "
            "for the existing Apify run."
        )

    print(
        f"Existing dataset ID: {dataset_id}"
    )

    # --------------------------------------------------------
    # 6. Open EXISTING dataset
    # --------------------------------------------------------
    print(
        "Opening existing Apify dataset..."
    )

    dataset_client = client.dataset(dataset_id)

    # --------------------------------------------------------
    # 7. Read EXISTING dataset
    # --------------------------------------------------------
    print(
        "Reading existing dataset items..."
    )

    raw_events = list(
        dataset_client.iterate_items()
    )

    print(
        f"Retrieved {len(raw_events)} items "
        f"from the existing Apify dataset."
    )

    if not raw_events:
        print(
            "No data found in the existing dataset."
        )
        return

    # --------------------------------------------------------
    # 8. Process macro events
    # --------------------------------------------------------
    print(
        "\nProcessing macro events..."
    )

    df_events = process_macro_events(
        raw_events
    )

    if df_events.empty:
        print(
            "No valid macro events were processed."
        )
        return

    print(
        f"Processed {len(df_events)} "
        f"valid macro events."
    )

    # --------------------------------------------------------
    # 9. Check existing database records
    # --------------------------------------------------------
    print(
        "\nChecking existing database records..."
    )

    existing_ids = set(
        row[0]
        for row in session.query(
            MacroEvent.event_id
        ).all()
    )

    print(
        f"Existing macro events in database: "
        f"{len(existing_ids)}"
    )

    # --------------------------------------------------------
    # 10. Remove duplicates
    # --------------------------------------------------------
    new_events = df_events[
        ~df_events["event_id"].isin(
            existing_ids
        )
    ].copy()

    print(
        f"New events to insert: "
        f"{len(new_events)}"
    )

    # --------------------------------------------------------
    # 11. Insert new events
    # --------------------------------------------------------
    if not new_events.empty:

        # Convert timezone-aware timestamps
        # to timezone-naive timestamps for SQLite.
        if "event_time_utc" in new_events.columns:

            new_events["event_time_utc"] = (
                new_events[
                    "event_time_utc"
                ].dt.tz_convert(None)
            )

        if "scraped_at" in new_events.columns:

            new_events["scraped_at"] = (
                new_events[
                    "scraped_at"
                ].dt.tz_convert(None)
            )

        new_events.to_sql(
            "macro_events",
            session.get_bind(),
            if_exists="append",
            index=False
        )

        print(
            f"Inserted {len(new_events)} "
            f"new events into macro_events table."
        )

    else:

        print(
            "No new events to insert. "
            "All events already exist."
        )

    # --------------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------------
    print("\n========================================")
    print("       MACRO DATA INGESTION SUMMARY")
    print("========================================")

    print(
        f"Apify Run ID: {RUN_ID}"
    )

    print(
        f"Dataset ID: {dataset_id}"
    )

    print(
        f"Raw events retrieved: "
        f"{len(raw_events)}"
    )

    print(
        f"Valid events processed: "
        f"{len(df_events)}"
    )

    print(
        f"New events inserted: "
        f"{len(new_events)}"
    )

    print(
        f"Events with Actual values: "
        f"{df_events['actual_value'].notna().sum()}"
    )

    print(
        f"Events with Forecast values: "
        f"{df_events['forecast_value'].notna().sum()}"
    )

    print(
        f"Events with Surprise values: "
        f"{df_events['macro_surprise'].notna().sum()}"
    )

    print("========================================")
    print(
        "Macro data ingestion completed."
    )
    print("========================================")


if __name__ == "__main__":
    main()
