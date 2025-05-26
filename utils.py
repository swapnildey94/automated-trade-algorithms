import os
import csv
import json
from datetime import datetime

def log_latest_context(context_data, log_path='latest_trade_context_log.csv', mode='csv'):
    """
    Log the latest trade quantity context to a file.
    context_data: list of dicts with 'Parameter' and 'Value' keys
    log_path: file path to write to
    mode: 'csv' or 'json'
    """
    if not context_data:
        return
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Add a run timestamp to the log
    context_row = {d['Parameter']: d['Value'] for d in context_data}
    context_row['Log Timestamp'] = timestamp
    try:
        if mode == 'csv':
            file_exists = os.path.isfile(log_path)
            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(context_row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(context_row)
        elif mode == 'json':
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(context_row) + '\n')
    except Exception as e:
        # Fail silently, do not interrupt UI/CLI
        pass
