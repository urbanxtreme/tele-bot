import os
from google.cloud import vision

# ✅ Update the correct file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Sreehari S\OneDrive\Desktop\github\tele-bot-453207-1a80190e0458.json"

client = vision.ImageAnnotatorClient()
print("✅ Google Cloud Vision API is working correctly!")
