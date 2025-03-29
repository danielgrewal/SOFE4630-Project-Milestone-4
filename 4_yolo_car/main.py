from ultralytics import YOLO
import cv2
import numpy as np
import json
import base64
import os
from io import BytesIO
from PIL import Image
from google.cloud import pubsub_v1

# Set Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sofe4630-8e3862153346.json"

# GCP Pub/Sub Setup
PROJECT_ID = "sofe4630"
TOPIC_NAME = "microservices_bus"
SUBSCRIPTION_NAME = "vehicle-detection-sub"

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()

subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load YOLO model
model = YOLO("./yolo11n.pt")

def decode_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("[ERROR] Failed to decode image.")
        return image
    except Exception as e:
        print(f"[ERROR] Exception while decoding image: {e}")
        return None

def encode_image(image):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to encode image: {e}")
        return None

def process_message(message):
    print(f"[INFO] Received a new message.")
    
    try:
        input_data = json.loads(message.data)

        if input_data.get("stage") != "vehicle_detection":
            print(f"[INFO] Skipping message - wrong stage: {input_data.get('stage')}")
            message.ack()
            return

        print(f"[INFO] Processing Timestamp: {input_data['Timestamp']}")
        print(f"[INFO] Car2_Location: {input_data['Car2_Location']}")
        print(f"[INFO] Car1_dimensions: {input_data['Car1_dimensions']}")
        print(f"[INFO] Car2_dimensions: {input_data['Car2_dimensions']}")
        print("[INFO] Image data received (not printing Base64).")

        Occluding_Image_View = decode_image(input_data["Occluding_Image_View"])
        if Occluding_Image_View is None:
            print("[ERROR] Failed to decode image.")
            message.ack()
            return

        print("[INFO] Successfully decoded image. Running YOLO model...")

        results = model.predict(source=Occluding_Image_View, save=True, save_txt=True)

        car_boxes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if result.names[int(cls)] == 'car':
                    rounded_box = [int(round(coord)) for coord in box[:4]]
                    car_boxes.append(rounded_box)

        print(f"[INFO] Extracted {len(car_boxes)} vehicle bounding boxes.")

        # Re-encode image for output
        Occluding_Image_View_b64 = encode_image(Occluding_Image_View)
        if Occluding_Image_View_b64 is None:
            print("[ERROR] Failed to encode image.")
            message.ack()
            return

        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Occluding_Image_View": Occluding_Image_View_b64,
            "Pedestrians": input_data["Pedestrians"],
            "Pedestrians_longitudinal": input_data["Pedestrians_longitudinal"],
            "Pedestrians_lateral": input_data["Pedestrians_lateral"],
            "vehicles": car_boxes,
            "stage": "vehicle_depth"
        }

        # Print output without image data
        output_log = output_data.copy()
        output_log.pop("Occluding_Image_View", None)
        print("[INFO] Final output before publishing (without images):")
        print(json.dumps(output_log, indent=2))

        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")

    message.ack()

# Subscribe to input Pub/Sub topic
streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
print(f"<<<STAGE 4>>> [INFO] Listening for messages on {subscription_path}...")

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
