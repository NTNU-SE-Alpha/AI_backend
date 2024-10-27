import requests
import os
from pydub import AudioSegment
import uuid

# Function to split audio and send segments to the API
def split_and_send_audio(file_path, api_url, save_dir="saved_segments"):
    try:
        audio = AudioSegment.from_file(file_path)  # Automatically detects the file format
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    segment_length = 10 * 1000  # Each segment is 10 seconds (in milliseconds)
    total_duration = len(audio)
    task_id = str(uuid.uuid4())  # Generate a unique task_id
    order = 0

    # Create directory to save audio segments if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for start in range(0, total_duration, segment_length):
        end = min(start + segment_length, total_duration)
        segment = audio[start:end]
        
        # Save each audio segment to a file in the save_dir
        temp_file_path = os.path.join(save_dir, f"segment_{order}.mp3")
        segment.export(temp_file_path, format="mp3")
        
        # Determine if this is the last segment
        is_end = 'true' if end == total_duration else 'false'

        # Upload audio segment to API
        data = {
            'task_id': task_id,
            'order': order,
            'end': is_end,
            'file_path': temp_file_path  # Provide the path to the audio segment
        }
        
        try:
            # Send a POST request with JSON data
            response = requests.post(f"{api_url}/api/tasks/{task_id}/audio", json=data)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

            # Handle API response
            try:
                result = response.json()
                # Check if 'text' is in the response (as per your Flask API)
                if 'text' in result:
                    # Print the rearranged text received from the API
                    print(f"Order {order}: Received rearranged text:\n{result['text']}\n")
                else:
                    # Print any other information returned by the API
                    print(f"Order {order}: API response: {result}")
            except ValueError:
                # Handle cases where response is not JSON
                print(f"Order {order}: Non-JSON response received: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send order {order}: {e}")
        
        order += 1

    print(f"All segments have been processed and saved in {save_dir}.")

# Main function
if __name__ == "__main__":
    api_url = "http://localhost:5000"  # Base URL of your API (without the specific endpoint)
    audio_file = r"C:\Users\41147\Downloads\my (mp3cut.net).mp3"  # Your MP3 source file

    if not os.path.exists(audio_file):
        print(f"Audio file does not exist: {audio_file}")
    else:
        split_and_send_audio(audio_file, api_url, save_dir="saved_segments")  # Save segments in 'saved_segments' directory
