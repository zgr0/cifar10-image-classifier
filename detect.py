import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import CIFAR10Classifier
from torchvision import transforms

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_model():
    model = CIFAR10Classifier()
    model.load_state_dict(torch.load('cifar10_model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Center crop to maintain aspect ratio
    h, w = original.shape[:2]
    size = min(h, w)
    offset_h, offset_w = (h - size) // 2, (w - size) // 2
    cropped = original[offset_h:offset_h+size, offset_w:offset_w+size]
    
    resized = cv2.resize(cropped, (32, 32))
    tensor = transform(resized).unsqueeze(0)
    return original, tensor, (offset_w, offset_h, size)

def classify_image(model, image_path):
    try:
        original, tensor, crop_coords = preprocess_image(image_path)
        
        with torch.no_grad():
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, class_id = torch.max(probs, 1)
        
        # Draw bounding box
        offset_w, offset_h, size = crop_coords
        cv2.rectangle(original, 
                      (offset_w, offset_h), 
                      (offset_w + size, offset_h + size), 
                      (0, 255, 0), 2)
        
        label = f"{class_names[class_id]}: {conf.item():.2f}"
        cv2.putText(original, label, (offset_w + 5, offset_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(original)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    model = load_model()
    classify_image(model, "image.png")