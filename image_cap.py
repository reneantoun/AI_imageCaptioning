import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image (replace 'YOUR_IMAGE_NAME.jpeg' with your actual image file name)
img_path = "fun.png"  # Example: "dog.jpg"
image = Image.open(img_path).convert('RGB')

# Prepare the inputs
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Generate a caption
outputs = model.generate(**inputs, max_length=50)

# Decode and print the generated caption
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Generated Caption:", caption)
