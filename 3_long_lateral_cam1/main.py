import torch
from torch import nn
import numpy as np
import json
import base64
import os
from google.cloud import pubsub_v1

# Set the correct service account JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sofe4630-8e3862153346.json"

# Google Pub/Sub setup
PROJECT_ID = "sofe4630"
TOPIC_NAME = "microservices_bus"
SUBSCRIPTION_NAME = "pedestrian-distance-sub"

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Load MLP Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_max = np.array([1920, 1080, 1920, 1080, 20])
        self.y_max = torch.Tensor([10, 10])
        self.mlp = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Tanh()
        )

    def forward(self, x):
        inputs = torch.Tensor(x / self.x_max).float()
        with torch.no_grad():
            logits = self.mlp(inputs) * self.y_max
        return logits

mlp = NeuralNetwork()
mlp.load_state_dict(torch.load("mlp_camA.pkl", weights_only=True))
mlp.eval()

# Function to process messages
def process_message(message):
    print(f"[INFO] Received a new message.")
    
    try:
        input_data = json.loads(message.data.decode("utf-8"))

        # Ensure it's the correct stage
        if input_data.get("stage") != "pedestrian_distance":
            print(f"[INFO] Skipping message - wrong stage: {input_data.get('stage')}")
            message.ack()
            return

        print(f"[INFO] Processing Timestamp: {input_data['Timestamp']}")
        print(f"[INFO] Car2_Location: {input_data['Car2_Location']}")
        print(f"[INFO] Car1_dimensions: {input_data['Car1_dimensions']}")
        print(f"[INFO] Car2_dimensions: {input_data['Car2_dimensions']}")
        print(f"[INFO] Image data received (not printing Base64).")

        # Extract input data
        pedestrians = np.array(input_data["Pedestrians"])
        pedestrian_depths = np.array(input_data["Pedestrians_depth"])

        # Run MLP model for each pedestrian
        outputs = []
        for i in range(pedestrians.shape[0]):
            box = pedestrians[i, :]
            depth = pedestrian_depths[i]
            x = np.array([*box, depth])
            outputs.append(np.array(mlp(torch.Tensor(x))))

        outputs = np.array(outputs)
        pedestrians_longitudinal = outputs[:, 0].tolist()
        pedestrians_lateral = outputs[:, 1].tolist()

        # Create output message
        output_data = {
            "Timestamp": input_data["Timestamp"],
            "Car2_Location": input_data["Car2_Location"],
            "Car1_dimensions": input_data["Car1_dimensions"],
            "Car2_dimensions": input_data["Car2_dimensions"],
            "Occluding_Image_View": input_data["Occluding_Image_View"], # Kept in message, not logged
            "Pedestrians": input_data["Pedestrians"],
            "Pedestrians_longitudinal": pedestrians_longitudinal,
            "Pedestrians_lateral": pedestrians_lateral,
            "stage": "vehicle_detection"  # Next stage
        }

        print("[INFO] Final output before publishing (without images):")
        output_copy = output_data.copy()
        output_copy.pop("Occluding_Image_View", None)  # Remove Base64 before printing
        print(json.dumps(output_copy, indent=2))

        # Publish message to shared bus
        publisher.publish(topic_path, json.dumps(output_data).encode("utf-8"))
        print("[INFO] Published to shared bus.")

        # Acknowledge message
        message.ack()

    except Exception as e:
        print(f"[ERROR] Exception processing message: {e}")
        message.ack()

# Subscribe and listen for messages
print(f"<<<STAGE 3>>> [INFO] Listening for messages on {subscription_path}...")
streaming_pull_future = subscriber.subscribe(subscription_path, process_message)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()
