# import os
# import logging
# import torch
# import pytesseract

# from telegram import Update
# from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# from PIL import Image
# from google.cloud import vision
# from transformers import CLIPProcessor, CLIPModel

# # Load environment variables securely (Set these in your OS environment variables)
# BOT_TOKEN = os.getenv("7611226934:AAGteQBKrerj_xHLV-LWERpibiDTHVVpHy4")  # Set this before running
# GOOGLE_CREDENTIALS_PATH = os.getenv("C:\\Users\\Sreehari S\\OneDrive\\Desktop\\github\\tele-bot-453207-1a80190e0458.json")  # Set this before running

# if not BOT_TOKEN or not GOOGLE_CREDENTIALS_PATH:
#     raise ValueError("Please set TELEGRAM_BOT_TOKEN and GOOGLE_APPLICATION_CREDENTIALS in environment variables.")

# # Set up Google Cloud Vision authentication
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# # Set Tesseract executable path (Windows users only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Load AI Model for Image Understanding
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Enable Logging
# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# # Initialize Google Vision Client
# vision_client = vision.ImageAnnotatorClient()

# async def start(update: Update, context: CallbackContext) -> None:
#     """ Sends a welcome message when the bot starts. """
#     await update.message.reply_text("Hello! Upload an image, and I will generate a LinkedIn post for you.")

# async def handle_image(update: Update, context: CallbackContext) -> None:
#     """ Handles image uploads and processes text extraction & captioning. """
#     try:
#         # Download the image from Telegram
#         photo_file = await update.message.photo[-1].get_file()
#         photo_path = "image.jpg"
#         await photo_file.download_to_drive(photo_path)

#         # Extract text from image
#         extracted_text = extract_text(photo_path)

#         # Generate LinkedIn post
#         post_text = generate_linkedin_post(photo_path, extracted_text)

#         # Send response to user
#         await update.message.reply_text(post_text)

#     except Exception as e:
#         logging.error(f"Error processing image: {e}")
#         await update.message.reply_text("Sorry, I couldn't process the image. Please try again.")

# def extract_text(image_path):
#     """ Extracts text from an image using Google Vision OCR with a Tesseract fallback. """
#     try:
#         # Using Google Vision OCR
#         with open(image_path, "rb") as image_file:
#             content = image_file.read()
#         image = vision.Image(content=content)
#         response = vision_client.text_detection(image=image)
#         text = response.text_annotations[0].description if response.text_annotations else ""

#         if text.strip():
#             return text
#     except Exception as e:
#         logging.error(f"Google Vision failed: {e}")

#     # Fallback to Tesseract OCR
#     try:
#         return pytesseract.image_to_string(Image.open(image_path))
#     except Exception as e:
#         logging.error(f"Tesseract OCR failed: {e}")
#         return "No text found."

# def generate_linkedin_post(image_path, extracted_text):
#     """ Generates a LinkedIn-style caption using the AI model CLIP. """
#     try:
#         image = Image.open(image_path)
#         inputs = processor(images=image, return_tensors="pt").to(device)

#         with torch.no_grad():
#             outputs = model.get_image_features(**inputs)

#         # LinkedIn-style post generation
#         caption = "📷 Here's a moment to share!\n"
#         if extracted_text.strip():
#             caption += f"📝 Extracted text: '{extracted_text}'\n"
#         caption += "#Networking #Event #Experience"

#         return caption
#     except Exception as e:
#         logging.error(f"CLIP model processing failed: {e}")
#         return "📷 Here's a moment to share!\n#Networking #Event #Experience"

# def main():
#     """ Initializes and starts the Telegram bot. """
#     app = Application.builder().token(BOT_TOKEN).build()

#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(MessageHandler(filters.PHOTO, handle_image))

#     logging.info("Bot is running...")
#     app.run_polling()

# if __name__ == "__main__":
#     main()

import os
import logging
import torch
import pytesseract

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

from PIL import Image
from google.cloud import vision
from transformers import CLIPProcessor, CLIPModel

# Load environment variables securely
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Ensure you set this in your system environment variables
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Ensure you set this in your system environment variables

if not BOT_TOKEN or not GOOGLE_CREDENTIALS_PATH:
    raise ValueError("Please set TELEGRAM_BOT_TOKEN and GOOGLE_APPLICATION_CREDENTIALS in environment variables.")

# Set up Google Cloud Vision authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# Set Tesseract executable path (Windows users only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load AI Model for Image Understanding
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Enable Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Initialize Google Vision Client
vision_client = vision.ImageAnnotatorClient()


async def start(update: Update, context: CallbackContext) -> None:
    """ Sends a welcome message when the bot starts. """
    await update.message.reply_text("Hello! Upload an image, and I will generate a LinkedIn post for you.")


async def handle_image(update: Update, context: CallbackContext) -> None:
    """ Handles image uploads and processes text extraction & captioning. """
    try:
        # Download the image from Telegram
        photo_file = await update.message.photo[-1].get_file()
        photo_path = "image.jpg"
        await photo_file.download_to_drive(photo_path)

        # Extract text from image
        extracted_text = extract_text(photo_path)

        # Generate LinkedIn post
        post_text = generate_linkedin_post(photo_path, extracted_text)

        # Send response to user
        await update.message.reply_text(post_text)

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        await update.message.reply_text("Sorry, I couldn't process the image. Please try again.")


def extract_text(image_path):
    """ Extracts text from an image using Google Vision OCR with a Tesseract fallback. """
    try:
        # Using Google Vision OCR
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        text = response.text_annotations[0].description if response.text_annotations else ""

        if text.strip():
            return text
    except Exception as e:
        logging.error(f"Google Vision failed: {e}")

    # Fallback to Tesseract OCR
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        logging.error(f"Tesseract OCR failed: {e}")
        return "No text found."


def generate_linkedin_post(image_path, extracted_text):
    """ Generates a LinkedIn-style caption using the AI model CLIP. """
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # LinkedIn-style post generation
        caption = "📷 Here's a moment to share!\n"
        if extracted_text.strip():
            caption += f"📝 Extracted text: '{extracted_text}'\n"
        caption += "#Networking #Event #Experience"

        return caption
    except Exception as e:
        logging.error(f"CLIP model processing failed: {e}")
        return "📷 Here's a moment to share!\n#Networking #Event #Experience"


def main():
    """ Initializes and starts the Telegram bot. """
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    logging.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
