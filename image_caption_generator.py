from transformers import pipeline
from PIL import Image

IMAGES = [
    'images/image-1.jpg',
    'images/image-10.jpg',
    'images/image-11.jpg',
    'images/image-12.jpg',
    'images/image-101.jpg',
    'images/image-104.jpg',
    'images/image-110.jpg',
    'images/image-115.jpg',
    'images/image-118.jpg',
    'images/image-119.jpg',
    'images/image-120.jpg',
    'images/image-199.jpg',
    'images/image-286.jpg',
    'images/image-371.jpg',
    'images/image-420.jpg',
    'images/image-475.jpg',
    'images/image-544.jpg',
]

def load_image(image_path):
    """
    Loads an image from the given path and returns it.
    """
    return Image.open(image_path)


def generate_caption(image):
    """
    Generates a caption for the given image using Hugging Face's image captioning pipeline.
    """
    # Initialize the image captioning pipeline
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Generate caption
    caption = caption_pipeline(image)

    return caption[0]['generated_text']


if __name__ == "__main__":
    for image in IMAGES:
        caption = generate_caption(load_image(image))
        print("Image ", image, "Generated Caption:", caption)
