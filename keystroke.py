import time
import csv
from pynput import keyboard

# Function to handle keypress events
def on_key_press(key):
    try:
        current_time_ms = time.time() * 1000  # Convert current time to milliseconds
        key_char = key.char  # Get the character of the pressed key
        delta_time_ms = current_time_ms - on_key_press.prev_time_ms
        on_key_press.prev_time_ms = current_time_ms
        print(f"Key: {key_char}, Time Since Previous Key (ms): {delta_time_ms:.2f}ms")

        # Write data to the CSV file
        with open('keypress_data.csv', mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([key_char, delta_time_ms])
    except AttributeError:
        # Handle special keys that don't have a 'char' attribute
        pass

# Initialize the previous keypress time to the current time
on_key_press.prev_time_ms = time.time() * 1000

# Create a listener for keyboard events
with keyboard.Listener(on_press=on_key_press) as listener:
    listener.join()