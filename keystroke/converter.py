import pandas as pd
import json

# Load CSV file into a DataFrame
csv_file_path = 'Sofia_keypress_data.csv'
df = pd.read_csv(csv_file_path)

# Create list of keys
keys = df['Key'].tolist()

# Create list of delta times
delta_times = df['Delta Time (ms)'].tolist()

# Convert DataFrame to JSON format
json_data = {
    "data": {
        "Key": keys,
        "Delta": delta_times
    }
}

# Convert dictionary to JSON string
json_string = json.dumps(json_data, indent=4)


with open('Sofia_keypress_data.json', 'w') as json_file:
    json_file.write(json_string)
