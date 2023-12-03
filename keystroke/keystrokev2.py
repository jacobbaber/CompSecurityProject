import os
import time
import csv
from pynput import keyboard

# Function to handle keypress events
global current_time_ms
global delta_time_ms
global release_time_ms
global press_time_ms
global duration_ms
global key_char

# Initialize variables
current_time_ms = 0
delta_time_ms = 0
release_time_ms = 0
press_time_ms = 0
duration_ms = 0
key_char = ""

def on_key_press(key):
    global current_time_ms
    global delta_time_ms
    global key_char

    try:
        current_time_ms = time.time() * 1000  # Convert current time to milliseconds
        key_char = key.char  # Get the character of the pressed key
        delta_time_ms = current_time_ms - on_key_press.prev_time_ms

        # If more than 2 seconds have passed, reset the time
        if delta_time_ms > 2000:
            delta_time_ms = 0

        on_key_press.prev_time_ms = current_time_ms

    except AttributeError:
        # Handle special keys that don't have a 'char' attribute
        pass


def on_key_release(key):
    global release_time_ms
    global duration_ms
    global key_char

    release_time_ms = time.time() * 1000  # Convert current time to milliseconds

    # Calculate the duration of the keypress
    duration_ms = release_time_ms - current_time_ms

    # Print the keypress information
    print(f"Key: {key_char}, Time Since Previous Key (ms): {delta_time_ms:.2f}ms, Duration (ms): {duration_ms:.2f}ms")

    # Write the keypress information to a CSV file
    with open('keypress_data.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key_char, delta_time_ms, duration_ms])

# Get the name of the user
user_name = input("Enter your name: ")

# Initialize the previous keypress time to the current time
on_key_press.prev_time_ms = time.time() * 1000

# Create a listener for keyboard events
with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
    listener.join()



