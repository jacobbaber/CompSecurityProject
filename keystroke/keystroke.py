import os
import time
import csv
from pynput import keyboard

# Function to handle keypress events
def on_key_press(key):
    try:
        current_time_ms = time.time() * 1000  # Convert current time to milliseconds
        key_char = key.char  # G.et the character of the pressed key
        delta_time_ms = current_time_ms - on_key_press.prev_time_ms

        # If more than 2 seconds have passed, reset the time
        if delta_time_ms > 2000:
            delta_time_ms = 0
            
        on_key_press.prev_time_ms = current_time_ms
        print(f"Key: {key_char}, Time Since Previous Key (ms): {delta_time_ms:.2f}ms")

        # Determine the filename based on the user's name
        file_name = f"{user_name}_keypress_data.csv"

        # Check if the file exists
        file_exists = os.path.isfile(file_name)

        # Write data to the CSV file
        with open(file_name, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # If the file didn't exist before, write the header
            if not file_exists:
                csv_writer.writerow(["Key", "Delta Time (ms)"])

            csv_writer.writerow([key_char, delta_time_ms])
    except AttributeError:
        # Handle special keys that don't have a 'char' attribute
        pass

# Get the name of the user
user_name = input("Enter your name: ")

# Initialize the previous keypress time to the current time
on_key_press.prev_time_ms = time.time() * 1000

# Create a listener for keyboard events
with keyboard.Listener(on_press=on_key_press) as listener:
    listener.join()

