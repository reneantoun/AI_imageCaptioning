import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert the input image (numpy array) to a PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    inputs = processor(images=raw_image, text="a photo of", return_tensors="pt")
    
    # Generate a caption
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(type="numpy"),  # Input: Image
    outputs="text",  # Output: Text caption
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
