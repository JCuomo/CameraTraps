import os
import json
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

images_dir = "/home/jcuomo/images/CH08__P40406151__V2"
output_base_dir = "/home/jcuomo/GroundingDINO/output"
detections_file = os.path.join(output_base_dir, "detections.json")

model = load_model("/home/jcuomo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/jcuomo/GroundingDINO/weights/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "animals"
BOX_TRESHOLD = 0.6
TEXT_TRESHOLD = 0.75

# List to store all detections
all_detections = []

for image_name in os.listdir(images_dir):
    if not image_name.endswith(".JPG"):
        continue

    image_path = os.path.join(images_dir, image_name)
    image_source, image = load_image(image_path)
    
    print("Processing image:", image_name)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    # Append detection details for the current image
    detection = {
        "image_path": image_path,
        "boxes": boxes.tolist(),  # Convert to list if boxes is a NumPy array
        "logits": logits.tolist(),  # Convert to list if logits is a NumPy array
    }
    all_detections.append(detection)
    
    # Annotate and save the image with detections
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(os.path.join(output_base_dir, image_name), annotated_frame)

# Write all detections to a JSON file
with open(detections_file, "w") as f:
    json.dump(all_detections, f, indent=4)

print("Detections saved to", detections_file)
