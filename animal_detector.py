import cv2
import torch
import pygame
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import os
import time

# Initialize Pygame mixer for sound
pygame.mixer.init()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)  # Use the small YOLOv5 model

# Animal class IDs in the COCO dataset
ANIMAL_CLASS_IDS = {16: "dog", 17: "cat", 19: "horse", 20: "sheep", 21: "cow"}

# Function to detect animals in a video and draw bounding boxes
def detect_animals_in_video(video_file, sound_file):
    # Load the sound file if specified
    if sound_file:
        try:
            sound = pygame.mixer.Sound(sound_file)
        except pygame.error as e:
            print(f"Error loading sound: {e}")
            sound = None
    else:
        sound = None

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    # Streamlit display setup
    st.title("Animal Detection in Video")
    st.sidebar.header("Settings")

    # Create a Stop button outside the loop so it only appears once
    stop_button = st.button('Stop Detection', key="stop_button")
    # Create a placeholder for video frame display
    video_placeholder = st.empty()

    # Play the video and process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Skip frames
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(frame_rgb)

            # Draw detections
            animals_detected = False
            for *xyxy, conf, cls in results.xyxy[0]:
                class_id = int(cls.item())
                if class_id in ANIMAL_CLASS_IDS and conf.item() > 0.5:
                    animals_detected = True
                    # Draw bounding box
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

                    # Get animal type from class ID
                    animal_type = ANIMAL_CLASS_IDS[class_id]
                    # Format confidence as a percentage
                    confidence_percentage = int(conf.item() * 100)
                    label = f"{animal_type} {confidence_percentage}%"
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Play alert sound if an animal is detected
            if animals_detected and sound:
                sound.play()

            # Resize the frame to fit the window size
            frame_resized = cv2.resize(frame, (original_width, original_height))

            # Convert to PIL image for Streamlit display
            pil_image = Image.fromarray(frame_resized)

            # Update the placeholder with the current frame
            video_placeholder.image(pil_image, caption="Detected Frame", use_column_width=True)

        frame_count += 1

        # Stop detection when the button is clicked
        if stop_button:
            st.write("Detection stopped.")
            break

        # Control frame rate (simulating video playback)
        time.sleep(0.03)  # Sleep to simulate ~30 FPS playback

    cap.release()

# Streamlit File uploader widgets
def load_video():
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    return video_file

def load_sound():
    sound_file = st.file_uploader("Upload a Sound", type=["wav", "mp3", "ogg"])
    return sound_file

# Main Streamlit app function
def main():
    # Load video and sound files
    video_file = load_video()
    sound_file = load_sound()

    if video_file is not None and sound_file is not None:
        st.write("Video and sound file loaded successfully!")
        
        # Convert video file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_temp:
            video_temp.write(video_file.read())
            video_file_path = video_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as sound_temp:
            sound_temp.write(sound_file.read())
            sound_file_path = sound_temp.name

        # Run the detection function
        detect_animals_in_video(video_file_path, sound_file_path)
        
        # Clean up temporary files after processing
        os.remove(video_file_path)
        os.remove(sound_file_path)
    elif video_file is not None:
        st.write("Please upload a sound file.")
    elif sound_file is not None:
        st.write("Please upload a video file.")
    else:
        st.write("Please upload both video and sound files to start detection.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
