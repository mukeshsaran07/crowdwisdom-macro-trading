import pytest
import pandas as pd
from src.scraper.apify_macro import parse_numeric, process_macro_events

def test_parse_numeric():
    assert parse_numeric("0.3%") == 0.3
    assert parse_numeric("336K") == 336_000.0
    assert parse_numeric("1.5M") == 1_500_000.0
    assert parse_numeric("-0.1") == -0.1
    assert parse_numeric("abc") is None
    assert parse_numeric(None) is None
    assert parse_numeric("") is None

def test_process_macro_events():
    mock_data = [
        {
            "id": "1",
            "eventName": "Core CPI",
            "timestamp": "2023-10-12T12:30:00Z",
            "actual": "0.4%",
            "forecast": "0.3%",
            "country": "US"
        },
        {
            "id": "2",
            "eventName": "NFP",
            "timestamp": "2023-10-06T12:30:00Z",
            "actual": "336K",
            "forecast": "170K"
        },
        {
            "id": "2", # Duplicate ID
            "eventName": "NFP",
            "timestamp": "2023-10-06T12:30:00Z",
            "actual": "336K",
            "forecast": "170K"
        },
        {
            "id": "3",
            "eventName": "Bad Date",
            "timestamp": "NotADate",
            "actual": "100"
        }
    ]
    
    df = process_macro_events(mock_data)
    
    # Duplicate 'id' 2 should be removed, and bad date 3 should be dropped
    assert len(df) == 2
    
    cpi = df[df['event_id'] == "1"].iloc[0]
    assert cpi['actual_value'] == 0.4
    assert cpi['forecast_value'] == 0.3
    assert abs(cpi['macro_surprise'] - 0.1) < 1e-5
    assert cpi['unit'] == '%'
    
    nfp = df[df['event_id'] == "2"].iloc[0]
    assert nfp['actual_value'] == 336000.0
    assert nfp['forecast_value'] == 170000.0
    assert nfp['macro_surprise'] == 166000.0
    assert nfp['unit'] == ''
