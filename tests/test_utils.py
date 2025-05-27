import unittest
import os
import pandas as pd
import json
from datetime import datetime
from unittest.mock import patch, mock_open

# Assuming utils.py is in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import log_latest_context

class TestLogLatestContext(unittest.TestCase):
    def setUp(self):
        # Create a dummy directory for logs
        self.test_log_dir = "test_logs"
        os.makedirs(self.test_log_dir, exist_ok=True)
        self.addCleanup(self.cleanup_test_logs)

    def cleanup_test_logs(self):
        # Remove the dummy directory and its contents
        if os.path.exists(self.test_log_dir):
            for f in os.listdir(self.test_log_dir):
                os.remove(os.path.join(self.test_log_dir, f))
            os.rmdir(self.test_log_dir)

    @patch('utils.datetime')
    def test_log_to_csv_new_file(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        context_data = [{'Parameter': 'key1', 'Value': 'value1'}, {'Parameter': 'key2', 'Value': 123}]
        log_file_path = os.path.join(self.test_log_dir, "test_log.csv")
        # Ensure file does not exist before test
        if os.path.exists(log_file_path): os.remove(log_file_path)

        log_latest_context(context_data, log_path=log_file_path, mode="csv")
        
        self.assertTrue(os.path.exists(log_file_path))
        df = pd.read_csv(log_file_path)
        expected_columns = ["Log Timestamp", "key1", "key2"]
        self.assertCountEqual(list(df.columns), expected_columns) 
        self.assertEqual(df.iloc[0]["key1"], "value1")
        self.assertEqual(df.iloc[0]["key2"], 123)
        self.assertEqual(df.iloc[0]["Log Timestamp"], "2023-01-01 12:00:00")

    @patch('utils.datetime')
    def test_log_to_csv_append_file(self, mock_datetime):
        mock_datetime.now.side_effect = [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 1, 12, 5, 0)]
        log_file_path = os.path.join(self.test_log_dir, "test_log_append.csv")
        if os.path.exists(log_file_path): os.remove(log_file_path)

        context_data1 = [{'Parameter': 'key1', 'Value': 'value1'}, {'Parameter': 'key2', 'Value': 123}]
        log_latest_context(context_data1, log_path=log_file_path, mode="csv")
        
        context_data2 = [{'Parameter': 'keyA', 'Value': 'valueA_next'}, {'Parameter': 'keyB', 'Value': 789}] # Different keys for second log
        log_latest_context(context_data2, log_path=log_file_path, mode="csv")
        
        self.assertTrue(os.path.exists(log_file_path))
        df = pd.read_csv(log_file_path)
        self.assertEqual(len(df), 2)
        
        # The current utils.py implementation:
        # 1. Writes header based on keys of the first data record: ["Log Timestamp", "key1", "key2"]
        # 2. For subsequent writes, it initializes DictWriter with the keys of the current record.
        #    It then writes these values in the order of *its own keys*.
        #    So, context_data2 (keys: LT, keyA, keyB) will write its timestamp, then context_data2['keyA'], then context_data2['keyB'].
        #    These will align with the columns established by the header: 'Log Timestamp', 'key1', 'key2'.
        # Therefore, 'keyA's value will land in column 'key1', and 'keyB's value in column 'key2' for the second row.
        # The actual columns in the CSV will only be those from the first write.
        expected_columns = ["Log Timestamp", "key1", "key2"]
        self.assertCountEqual(list(df.columns), expected_columns)

        # Check first row
        self.assertEqual(df.iloc[0]["Log Timestamp"], "2023-01-01 12:00:00")
        self.assertEqual(df.iloc[0]["key1"], "value1")
        self.assertEqual(df.iloc[0]["key2"], 123)
        
        # Check second row
        self.assertEqual(df.iloc[1]["Log Timestamp"], "2023-01-01 12:05:00")
        self.assertEqual(df.iloc[1]["key1"], "valueA_next") # Value of 'keyA' lands in column 'key1'
        self.assertEqual(df.iloc[1]["key2"], 789)         # Value of 'keyB' lands in column 'key2'
        self.assertEqual(df.iloc[1]["Log Timestamp"], "2023-01-01 12:05:00")

    @patch('utils.datetime')
    def test_log_to_json_new_file(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2023, 1, 2, 10, 0, 0)
        context_data = [{'Parameter': 'event', 'Value': 'test_event'}, {'Parameter': 'user_id', 'Value': 'user123'}]
        log_file_path = os.path.join(self.test_log_dir, "test_log.json")
        if os.path.exists(log_file_path): os.remove(log_file_path)

        log_latest_context(context_data, log_path=log_file_path, mode="json")

        self.assertTrue(os.path.exists(log_file_path))
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        log_entry = json.loads(lines[0])
        
        self.assertEqual(log_entry["event"], "test_event")
        self.assertEqual(log_entry["user_id"], "user123")
        self.assertEqual(log_entry["Log Timestamp"], "2023-01-02 10:00:00")

    @patch('utils.datetime')
    def test_log_to_json_append_file(self, mock_datetime):
        mock_datetime.now.side_effect = [datetime(2023, 1, 2, 10, 0, 0), datetime(2023, 1, 2, 10, 5, 0)]
        log_file_path = os.path.join(self.test_log_dir, "test_log_append.json")
        if os.path.exists(log_file_path): os.remove(log_file_path)
        
        context_data1 = [{'Parameter': 'event', 'Value': 'event1'}, {'Parameter': 'data', 'Value': 'data1'}]
        log_latest_context(context_data1, log_path=log_file_path, mode="json")
        
        context_data2 = [{'Parameter': 'event', 'Value': 'event2'}, {'Parameter': 'other_data', 'Value': 'data2'}]
        log_latest_context(context_data2, log_path=log_file_path, mode="json")

        self.assertTrue(os.path.exists(log_file_path))
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        
        log_entry1 = json.loads(lines[0])
        self.assertEqual(log_entry1["event"], "event1")
        self.assertEqual(log_entry1["data"], "data1") # Ensure original keys are checked
        self.assertEqual(log_entry1["Log Timestamp"], "2023-01-02 10:00:00")
        
        log_entry2 = json.loads(lines[1])
        self.assertEqual(log_entry2["event"], "event2")
        self.assertEqual(log_entry2["other_data"], "data2") # Ensure new keys are checked
        self.assertEqual(log_entry2["Log Timestamp"], "2023-01-02 10:05:00")

    def test_log_empty_context_csv(self):
        log_file_path = os.path.join(self.test_log_dir, "test_empty.csv")
        # Ensure file does not exist initially
        if os.path.exists(log_file_path): os.remove(log_file_path)

        log_latest_context([], log_path=log_file_path, mode="csv")
        self.assertFalse(os.path.exists(log_file_path), "File should not be created for empty context if it doesn't exist.")

        # Test case where file already exists
        # Create a dummy file first
        with open(log_file_path, 'w') as f:
            f.write("some,initial,content\n") # Give it some content
        self.assertTrue(os.path.exists(log_file_path))
        initial_size = os.path.getsize(log_file_path)
        
        log_latest_context([], log_path=log_file_path, mode="csv")
        final_size = os.path.getsize(log_file_path)
        self.assertEqual(initial_size, final_size, "File should not be modified for empty context if it exists.")
        # Verify content is unchanged
        with open(log_file_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, "some,initial,content\n")


    def test_log_empty_context_json(self):
        log_file_path = os.path.join(self.test_log_dir, "test_empty.json")
        if os.path.exists(log_file_path): os.remove(log_file_path)

        log_latest_context([], log_path=log_file_path, mode="json")
        self.assertFalse(os.path.exists(log_file_path), "File should not be created for empty context if it doesn't exist.")

        # Test case where file already exists
        with open(log_file_path, 'w') as f:
            f.write("{\"initial_key\": \"initial_value\"}\n") # Give it some content
        self.assertTrue(os.path.exists(log_file_path))
        initial_size = os.path.getsize(log_file_path)

        log_latest_context([], log_path=log_file_path, mode="json")
        final_size = os.path.getsize(log_file_path)
        self.assertEqual(initial_size, final_size, "File should not be modified for empty context if it exists.")
        # Verify content is unchanged
        with open(log_file_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, "{\"initial_key\": \"initial_value\"}\n")


    def test_directory_creation(self):
        new_log_dir = os.path.join(self.test_log_dir, "new_subdir_for_test") # More specific name
        # Ensure the directory does not exist before the call
        if os.path.exists(new_log_dir):
            # Clean up files inside before rmdir if any exist from previous runs
            for f_name in os.listdir(new_log_dir):
                os.remove(os.path.join(new_log_dir, f_name))
            os.rmdir(new_log_dir)

        context_data = [{'Parameter': 'data', 'Value': 'test'}]
        log_file_path = os.path.join(new_log_dir, "test_in_subdir.csv")
        # log_latest_context should create new_log_dir
        log_latest_context(context_data, log_path=log_file_path, mode="csv")
        
        self.assertTrue(os.path.exists(new_log_dir), "Directory should be created by log_latest_context.")
        self.assertTrue(os.path.exists(log_file_path), "Log file should be created in the new directory.")
        
        # Clean up: remove file first, then directory
        os.remove(log_file_path)
        os.rmdir(new_log_dir)

if __name__ == '__main__':
    unittest.main()
