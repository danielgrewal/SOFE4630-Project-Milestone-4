from PIL import Image
import depth_pro
import numpy as np
import torch
import json
import base64
import os
from io import BytesIO
from google.cloud import pubsub_v1

# Set the correct service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sofe4630-8e3862153346.json"

# Set up Pub/Sub
PROJECT_ID = "sofe4630"
TOPIC_NAME = "microservices_bus"
SUBSCRIPTION_NAME = "vehicle-depth-sub"

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Focal length for vehicle depth camera
F_PX = 2500
DEPTH_THRESHOLD = 20  # Keep only vehicles within 20 meters

# Load model and transform once
model, transform = depth_pro.create_model_and_transforms()
model.eval()

def decode_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(img_data))
        return image
    except Exception as e:
        print(f"[ERROR] Failed to decode image: {e}")
        return None

def process_message(message):
    print(f"[INFO] Received a new message.")

    try:
        input_data = json.loads(message.data)

        if input_data.get("stage") != "vehicle_depth":
            print(f"[INFO] Skipping message - wrong stage: {input_data.get('stage')}")
            message.ack()
            return

        vehicles = input_data.get("vehicles", [])
        occluding_image_b64 = input_data.get("Occluding_Image_View")

        if not vehicles or not occluding_image_b64:
            print("[INFO] No vehicles or image data found. Skipping.")
            message.ack()
            return

        occluding_image = decode_image(occluding_image_b64)
        if occluding_image is None:
            print("[ERROR] Failed to load image.")
            message.ack()
            return

        print("[INFO] Successfully decoded image. Running depth estimation...")

        # Preprocess and infer
        occluding_image = transform(occluding_image)
        prediction = model.infer(occluding_image, f_px=torch.Tensor([F_PX]))
        depth_map = prediction["depth"].squeeze().cpu().numpy()

        filtered_vehicles = []
        vehicle_depths = []

        for box in vehicles:
            x1, y1, x2, y2 = map(int, box)
            y1, y2 = max(0, y1), min(depth_map.shape[0], y2)
            x1, x2 = max(0, x1), min(depth_map.shape[1], x2)

            depth_value = np.median(depth_map[y1:y2, x1:x2])
            print(f"[DEBUG] Vehicle box: {box} | Estimated depth: {depth_value:.2f} meters")

            if depth_value < DEPTH_THRESHOLD:
                filtered_vehicles.append(box)
                vehicle_depths.append(depth_value)

        print(f"[INFO] Filtered {len(filtered_vehicles)} vehicles within {DEPTH_THRESHOLD}m.")

        # Prepare final message
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Pedestrians": input_data["Pedestrians"],
            "Pedestrians_longitudinal": input_data["Pedestrians_longitudinal"],
            "Pedestrians_lateral": input_data["Pedestrians_lateral"],
            "vehicles": filtered_vehicles,
            "vehicles_depth": [float(d) for d in vehicle_depths],
            "stage": "vehicle_distance"
        }

        # Log without image
        print(f"[INFO] Final output before publishing (without images):")
        print(json.dumps(output_data, indent=2))

        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")

    message.ack()

# Subscribe to the topic
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
print(f"<<<STAGE 5>>> [INFO] Listening for messages on {subscription_path}...")

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
