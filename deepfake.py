from flask import Flask, request, jsonify
import os
import subprocess
import json
import logging
from bson.objectid import ObjectId
from datetime import datetime
from pymongo import MongoClient

app = Flask(__name__)

# Configure logging to ensure messages are displayed in logs
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure paths
TEMP_FOLDER = r"custom_data\temp"
REAL_FOLDER = r"custom_data\train\real"
FAKE_FOLDER = r"custom_data\train\fake"

# Ensure directories exist
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(REAL_FOLDER, exist_ok=True)
os.makedirs(FAKE_FOLDER, exist_ok=True)

# MongoDB setup
client = MongoClient("mongodb://localhost/deepfake")
db = client["deepfake"]
collection = db["files"]

@app.route("/")
def home():
    # logging.info("Home endpoint accessed.")
    return "Hello, World!"

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        # logging.warning("No file part in request.")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        # logging.warning("No file selected.")
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(TEMP_FOLDER, file.filename)
    file.save(filepath)

    # logging.info(f"File uploaded: {file.filename}")

    # Process the file
    process_result = process_files(TEMP_FOLDER)
    return jsonify(process_result)
    # return {"status": 200, "data": jsonify(process_result)}

@app.route("/train", methods=["GET"])
def train_model():
    try:
        SCRIPT_PATH = "train.py"

        command1 = [
            "python", SCRIPT_PATH, 
            "--d", "custom_data", 
            "--m", "vae", 
            "-e", 1,
            "-t", "y"
        ]

        command2 = [
            "python", SCRIPT_PATH, 
            "--d", "custom_data", 
            "--m", "ed", 
            "--e", 5,
            "-t", "y"
        ]

        # Run prediction script
        subprocess.run(command1, capture_output=True, text=True, check=True)
        subprocess.run(command2, capture_output=True, text=True, check=True)

        return {"status": 200, "msg": "Model trained successfully"}

    except subprocess.CalledProcessError as e:
        return {"status": 500, "error": str(e)}
    except Exception as e:
        return {"status": 500, "error": str(e)}

def process_files(folder):
    try:
        SCRIPT_PATH = "prediction.py"
        ENCODER_MODEL = "genconvit_ed_Mar_22_2025_01_07_24"
        VAE_MODEL = "genconvit_vae_Mar_22_2025_01_16_49"
        FRAMES = "10"

        command = [
            "python", SCRIPT_PATH, 
            "--p", folder, 
            "--e", ENCODER_MODEL, 
            "--v", VAE_MODEL, 
            "--f", FRAMES
        ]

        # Run prediction script
        subprocess.run(command, capture_output=True, text=True, check=True)

        JSON_FILE = get_most_recent_file("result")  # Ensure this path is correct

        if not JSON_FILE or not os.path.exists(JSON_FILE):
            return {"error": "Output JSON file missing after execution"}

        with open(JSON_FILE, "r") as json_file:
            data = json.load(json_file)

        results = []
        video_info = data.get("video", {})
        video_name = video_info.get("name", [])
        predictions = video_info.get("pred", [])
        classes = video_info.get("klass", [])
        pred_labels = video_info.get("pred_label", [])
        correct_labels = video_info.get("correct_label", [])

        for i, name in enumerate(video_name):
            prediction = predictions[i]
            # klass = classes[i]
            pred_label = pred_labels[i]
            # correct_label = correct_labels[i]

            # Move file based on prediction
            dest_folder = REAL_FOLDER if pred_label == "REAL" else FAKE_FOLDER
            src_path = os.path.join(TEMP_FOLDER, name)
            dest_path = os.path.join(dest_folder, name)

            if os.path.exists(dest_path):
                os.remove(src_path)  # If file already exists in destination, remove from temp
            else:
                os.rename(src_path, dest_path)  # Move only if not already present

            # Store result in MongoDB
            result = {
                "filename": name,
                "pred": prediction,
                # "klass": klass,
                "status": pred_label,
                # "correct_label": correct_label,
                "created_at": datetime.now(),  # Store created timestamp
                "updated_at": datetime.now(),   # Store updated timestamp
                "metadata": video_info
            }
            inserted_id = collection.insert_one(result).inserted_id  # Get MongoDB _id

            # Convert ObjectId to string before returning JSON
            result["_id"] = str(inserted_id)

            results.append(result)

        return results

    except subprocess.CalledProcessError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

def get_most_recent_file(folder_path):
    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Define the filename format and how the datetime is embedded
    file_format = "%B_%d_%Y_%H_%M_%S"

    latest_file = None
    latest_time = None

    # Iterate through files and check the timestamp from the filename
    for file in files:
        if "prediction_other_genconvit_" not in file or not file.endswith(".json"):
            continue  # Skip files that don't match the expected prefix

        # The datetime portion is always at the end of the filename, after the second underscore
        parts = file.split('_')
        if len(parts) < 4:
            # Skip if the filename doesn't match the expected pattern
            continue

        # Extract the datetime portion
        parts = file.split('_')
        timestamp_str = "_".join(parts[-6:]).replace(".json", "")  # Last 6 parts are the timestamp

        try:
            file_time = datetime.strptime(timestamp_str, file_format)
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = file
        except ValueError:
            print(f"Skipping file (invalid datetime format): {file}")
            continue

    return os.path.join(folder_path, latest_file) if latest_file else ""  # Return full path


if __name__ == "__main__":
    app.run(debug=True)
