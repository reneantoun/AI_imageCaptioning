import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"

# Download and parse the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
img_elements = soup.find_all('img')

for img_element in img_elements:
    img_url = img_element.get('src')
    if 'svg' in img_url or '1x1' in img_url:  # Skip small or SVG images
        continue
    if img_url.startswith('//'):  # Correct partial URLs
        img_url = 'https:' + img_url
    elif not img_url.startswith('http'):
        continue

    try:
        # Download the image
        response = requests.get(img_url)
        raw_image = Image.open(BytesIO(response.content)).convert('RGB')

        # Process and generate caption
        inputs = processor(images=raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"Caption for {img_url}: {caption}")
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")

with open("captions.txt", "w") as caption_file:
    for img_element in img_elements:
        # Generate captions as shown above
        caption_file.write(f"{img_url}: {caption}\n")
