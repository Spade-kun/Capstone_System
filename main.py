import os
from dotenv import load_dotenv
from google.cloud import vision
import io
import cv2
import re

# Set the path to the service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'aidriven-452606-ab0d220198ff.json'

# Load environment variables from .env file
load_dotenv()

def is_valid_plate(text):
    """Check if the text matches Philippine plate number patterns."""
    # Philippine plate patterns
    patterns = [
        r'^[A-Z]{3}\d{3,4}$',      # ZMC145, ABC1234
        r'^[A-Z]{2}[A-Z0-9]\d{3,4}$',  # For plates like ZMC145
        r'^[A-Z]{2}\d{3,4}$',      # For some special plates
    ]
    
    # Clean the text: remove spaces and convert to uppercase
    cleaned_text = ''.join(text.split()).upper()
    
    # Check if the cleaned text matches any of the patterns
    for pattern in patterns:
        if re.match(pattern, cleaned_text):
            return True, cleaned_text
    return False, None

def detect_text(image_content):
    """Detects text in the image content and filters for plate numbers."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Process only if texts were detected
    if texts:
        # The first text contains all text in the image
        full_text = texts[0].description
        # Split the text into lines and process each line
        for line in full_text.split('\n'):
            is_plate, formatted_plate = is_valid_plate(line)
            if is_plate:
                print(f'Found plate number: {formatted_plate}')
                vertices = (['({},{})'.format(vertex.x, vertex.y)
                           for vertex in texts[0].bounding_poly.vertices])
                print('bounds: {}'.format(','.join(vertices)))
                return formatted_plate

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return None

def capture_and_detect():
    """Captures frames from the camera and detects text."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        image_content = buffer.tobytes()

        # Detect text in the frame
        plate_number = detect_text(image_content)
        
        # If a plate number is detected, display it on the frame
        if plate_number:
            cv2.putText(frame, plate_number, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_detect()