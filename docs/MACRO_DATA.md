# Macroeconomic Event Data

## Apify Actor Selection
To fulfill the requirement of collecting a 180-day macroeconomic calendar, we selected the Apify actor **[Economic Calendar Data (Investing.com)](https://apify.com/pintostudio/economic-calendar-data-investing-com)**. 
- **Actor ID**: `pintostudio/economic-calendar-data-investing-com`
- **Reason for Selection**: This actor specializes in scraping Investing.com's high-quality economic calendar. It supports detailed filtering (importance, country) and reliably provides all required fields: actual, forecast, and previous values.

## Configuration & Required Input
The script expects an Apify API token to run.
1. Copy `.env.example` to `.env`.
2. Add your token: `APIFY_API_TOKEN=your_token_here`

The script automatically sets the `startDate` to 180 days ago and the `endDate` to the current UTC date. It filters for `US` events with `high` and `medium` importance.

## Fields & Processing
- **Timestamps**: `original_event_time` is parsed and explicitly converted to a UTC naive datetime (`event_time_utc`) before storage in SQLite to ensure proper alignment with the trading data.
- **Value Parsing**: String values containing units (e.g., `K`, `M`, `B`, `%`) are strictly converted to floating-point numbers.
- **Macro Surprise Calculation**: Calculated deterministically as `macro_surprise = actual_value - forecast_value`. If either is missing or non-numeric, it is left as `NULL`.

## Database Schema
Events are stored in the SQLite `macro_events` table alongside the trading logs, preventing duplicates via an `event_id` unique constraint.

## Reproducibility & Limitations
- **Reproducibility**: Run `python scripts/fetch_macro_data.py`. If no token is provided in `.env`, run with `python scripts/fetch_macro_data.py --mock` to test the pipeline safely.
- **Limitations**: Missing actuals/forecasts are inherent to certain events (e.g., speeches) and are safely ignored by the surprise calculator.

**Disclaimer**: This data is scraped from Investing.com via the aforementioned Apify actor. It is not official CrowdWisdomTrading data.
