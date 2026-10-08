import re
import pandas as pd
from datetime import datetime, timezone
import hashlib

def parse_numeric(val: str):
    if val is None or pd.isna(val) or val == '' or str(val).strip() == '':
        return None
    val_str = str(val).upper().replace(',', '').strip()
    
    match = re.search(r'([-+]?[0-9]*\.?[0-9]+)\s*([A-Z%]*)$', val_str)
    if not match:
        match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', val_str)
        if not match:
            return None
        num_str = match.group(1)
        unit = ''
    else:
        num_str = match.group(1)
        unit = match.group(2)
        
    try:
        num = float(num_str)
    except ValueError:
        return None
        
    if unit == 'K':
        num *= 1_000
    elif unit == 'M':
        num *= 1_000_000
    elif unit == 'B':
        num *= 1_000_000_000
        
    return num

def generate_id(name: str, time_str: str) -> str:
    s = f"{name}_{time_str}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def process_macro_events(raw_events: list) -> pd.DataFrame:
    processed = []
    scraped_at = datetime.now(timezone.utc)
    seen_ids = set()
    
    for item in raw_events:
        event_name = item.get('eventName', item.get('title', item.get('name', 'Unknown')))
        orig_time = item.get('timestamp', item.get('time', item.get('date', '')))
        
        # Use provided ID or generate a deterministic one
        event_id = str(item.get('id', generate_id(event_name, str(orig_time))))
        
        if event_id in seen_ids:
            continue
        seen_ids.add(event_id)
        
        try:
            dt = pd.to_datetime(orig_time)
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC') # default assumption
            event_time_utc = dt.tz_convert('UTC')
        except:
            event_time_utc = None
            
        actual_val = parse_numeric(item.get('actual'))
        forecast_val = parse_numeric(item.get('forecast', item.get('consensus')))
        prev_val = parse_numeric(item.get('previous'))
        
        surprise = None
        if actual_val is not None and forecast_val is not None:
            surprise = actual_val - forecast_val
            
        # Extract unit if present directly in item or from string
        unit = item.get('unit', '')
        if not unit and isinstance(item.get('actual'), str) and '%' in item.get('actual'):
            unit = '%'
            
        processed.append({
            'event_id': event_id,
            'event_time_utc': event_time_utc,
            'original_event_time': str(orig_time),
            'original_timezone': str(item.get('timezone', 'UTC')),
            'country': item.get('country', 'Unknown'),
            'currency': item.get('currency', 'Unknown'),
            'event_name': event_name,
            'category': item.get('category', 'Macro'),
            'importance': str(item.get('importance', item.get('impact', 'Low'))),
            'actual_value': actual_val,
            'forecast_value': forecast_val,
            'previous_value': prev_val,
            'macro_surprise': surprise,
            'unit': unit,
            'source': 'pintostudio/economic-calendar-data-investing-com',
            'scraped_at': scraped_at
        })
        
    df = pd.DataFrame(processed)
    # Filter out records where event_time_utc could not be parsed
    if not df.empty:
        df = df.dropna(subset=['event_time_utc'])
    return df
