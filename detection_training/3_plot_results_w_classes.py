import json
from PIL import Image, ImageDraw, ImageFont
import os

from CameraTraps.detection.utils import count_detections, get_bboxes
from CameraTraps.metadate.utils import get_counts_dir

# Directories
images_dir = "/home/jcuomo/images"
detections_file = "/home/jcuomo/CameraTraps/output/detection_training/detections_all.json"
classification_path = "/home/jcuomo/CameraTraps/output/classification/classifications.json"
output_dir = "/home/jcuomo/CameraTraps/output/detection_training/with_classification"
folders = {"good": "good", "under": "under", "over": "over"}

# Create directories if they don't exist
for folder in folders.values():
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Load detections, counts, and bounding boxes
n_detections = count_detections(detections_file)
n_counts = get_counts_dir(images_dir)
bboxes = get_bboxes(detections_file)

# Load a font (change the path if necessary)
try:
    font = ImageFont.truetype("arial.ttf", 50)
except IOError:
    font = ImageFont.load_default()

# Initialize HTML content
html_content = """
<html>
<head><title>Image Results</title></head>
<body>
<h1>Image Results</h1>
"""

N=0
good=0
over=0
under=0
for image_path in n_detections.keys():
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Get detections and counts
    detection_number = n_detections[image_path]
    count_number = n_counts[image_path]

    with open(classification_path, "r") as classification_file:
        classification = json.load(classification_file)

    width, height = image.size
    # Draw bounding boxes
    for bbox_raw, confidence in bboxes[image_path]:
        x1, y1, w_box, h_box = bbox_raw
        bbox = [x1 * width, y1 * height, (x1 + w_box) * width, (y1 + h_box) * height]
        color = "green" if confidence > 0.6 else ("blue" if confidence > 0.2 else "red")

        draw.rectangle(bbox, outline=color, width=5)
        draw.text((bbox[0], bbox[1] - 70), f'{confidence:.2f}', fill=color, font=font)
        if image_path in classification:
            for pred_boxes in classification[image_path]:
                if bbox_raw == pred_boxes["bbox"]:
                    classification_confidence = pred_boxes["pred_class_prob"]
                    pred_class = pred_boxes["pred_class"]
                    draw.text((bbox[2] - 70, bbox[1] - 70), f'{classification_confidence:.2f}', fill="orange", font=font)
                    draw.text((bbox[2] - 70, bbox[1] - 130), f'{pred_class}', fill="orange", font=font)
                    break
            

    # Add detection and count numbers
    draw.text((10, 10), f'Detections: {detection_number}', fill="blue", font=font)
    draw.text((10, 60), f'Count: {count_number}', fill="blue", font=font)

    # Determine folder based on comparison of detection and count
    N+=1
    if detection_number == count_number:
        category = "good"
        good+=1
    elif detection_number < count_number:
        category = "over"
        over+=1
    else:
        category = "under"
        under+=1
    # Save image to the appropriate folder
    result_path = os.path.join(output_dir, folders[category], os.path.basename(image_path))
    image.save(result_path)

    # Add image to HTML
    html_content += f'<h2>{category.capitalize()} - {os.path.basename(image_path)}</h2>\n'
    html_content += f'<img src="{result_path}" alt="{category}" style="max-width:100%; height:auto;">\n'

result_str = f"good:{good/N*100:.1f}, over:{over/N*100:.1f}, under:{under/N*100:.1f}"
print(result_str)
# Finish HTML content
html_content += result_str
html_content += """
</body>
</html>
"""

# Save the HTML file
html_file_path = os.path.join(output_dir, "index.html")
with open(html_file_path, "w", encoding="utf-8") as html_file:
    html_file.write(html_content)

print(f"HTML file created: {html_file_path}")
