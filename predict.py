import sys
import torch
import torchvision.transforms as T
from PIL import Image
from models import get_resnet50
from data_loader import get_data_loaders

def predict_leaf(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, classes = get_data_loaders()
    
    model = get_resnet50(num_classes=len(classes)).to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)

    predicted_class = classes[pred_idx.item()]
    confidence_pct = conf.item() * 100

    print("==================================================")
    print("      PLANT DISEASE INFERENCE RESULT")
    print("==================================================")
    print(f"Input Image : {image_path}")
    print(f"Prediction  : {predicted_class}")
    print(f"Confidence  : {confidence_pct:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_leaf(sys.argv[1])
    else:
        print("[INFO] Usage: python predict.py <path_to_leaf_image.jpg>")