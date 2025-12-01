import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Define class names as specified
CLASSES = ['profile_ok', 'ventral', 'dorsal', 'other']

# Global model instance
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Expected path for the model weights
MODEL_PATH = os.getenv("ORIENTATION_MODEL_PATH", "orientation_model.pth")

def load_model():
    """
    Loads the ResNet18 model adapted for 4 classes.
    """
    global MODEL
    if MODEL is not None:
        return MODEL

    try:
        # Load pre-trained ResNet18
        # User said "Use ResNet18 (pretrained)".
        # Usually we load the architecture and then load our custom weights.
        model = torchvision.models.resnet18(weights=None)

        # Modify the final layer to match 4 classes
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))

        # Load weights if they exist
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Loaded orientation model from {MODEL_PATH}")
        else:
            print(f"Warning: Orientation model weights not found at {MODEL_PATH}. Using random weights (predictions will be meaningless).")

        model.to(DEVICE)
        model.eval()
        MODEL = model
    except Exception as e:
        print(f"Error loading orientation model: {e}")
        MODEL = None

    return MODEL

def classify_orientation(image: np.ndarray) -> tuple:
    """
    Classifies the orientation of a tadpole image.
    Input: RGB/BGR numpy array (cropped tadpole image)
    Output: (label, probability)
    """
    model = load_model()
    if model is None:
        return "unknown", 0.0

    # Preprocessing
    try:
        # Convert BGR to RGB (OpenCV default is BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Standard ResNet preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE) # Create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top class
        prob, cat_id = torch.max(probabilities, 0)

        label = CLASSES[cat_id.item()]
        probability = prob.item()

        return label, probability

    except Exception as e:
        print(f"Error during orientation classification: {e}")
        return "error", 0.0
