import json
import time
import base64
import pandas as pd
import os
from google.cloud import pubsub_v1

# Set the correct service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sofe4630-8e3862153346.json"

# Set up Pub/Sub
PROJECT_ID = "sofe4630"
TOPIC_NAME = "microservices_bus"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load CSV file
csv_file = "Labels.csv"
df = pd.read_csv(csv_file)

def encode_image(image_path):
    """Reads an image and encodes it to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Warning: Image {image_path} not found!")
        return None

for _, row in df.iterrows():
    # Encode images
    occluded_image_b64 = encode_image(row["Occluded_Image_View"])
    occluding_image_b64 = encode_image(row["Occluding_Image_View"])

    if occluded_image_b64 is None or occluding_image_b64 is None:
        print("Skipping row due to missing image.")
        continue  # Skip sending this message if an image is missing

    # Construct the message (Keep values as lists, not NumPy arrays)
    message_data = {
        "Timestamp": str(row["Timestamp"]),
        "Car2_Location": [row["Car2_Location_X"], row["Car2_Location_Y"]],
        "Car1_dimensions": [row["Car1_Length"], row["Car1_Width"]],
        "Car2_dimensions": [row["Car2_Length"], row["Car2_Width"]],
        "Occluded_Image_View": occluded_image_b64,  # Base64 image
        "Occluding_Image_View": occluding_image_b64,  # Base64 image
        "stage": "pedestrian_detection"
    }

    # Prepare a clean version of the message for display (exclude image data)
    display_message = message_data.copy()
    display_message.pop("Occluded_Image_View", None)
    display_message.pop("Occluding_Image_View", None)

    # Publish the message
    future = publisher.publish(topic_path, json.dumps(message_data).encode("utf-8"))
    print(f"\n[INFO] Published message with ID: {future.result()}")
    print("[INFO] Message contents (excluding image data):")
    print(json.dumps(display_message, indent=2))

    time.sleep(2)  # Small delay to avoid overwhelming the system
