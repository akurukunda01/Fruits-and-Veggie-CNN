import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from flask import Flask, jsonify, request
from flask_cors import CORS
import io
from PIL import Image

app = Flask(__name__)
CORS(app)


class_names = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 'Apple 5', 'Apple 6', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Braeburn 1', 'Apple Core 1', 'Apple Crimson Snow 1', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith 1', 'Apple hit 1', 'Apple Pink Lady 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious 1', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apple Rotten 1', 'Apple worm 1', 'Apricot 1', 'Avocado 1', 'Avocado Black 1', 'Avocado Green 1', 'Avocado ripe 1', 'Banana 1', 'Banana 3', 'Banana 4', 'Banana Lady Finger 1', 'Banana Red 1', 'Beans 1', 'Beetroot 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 'Blackberrie not rippen 1', 'Blueberry 1', 'Cabbage red 1', 'Cabbage white 1', 'Cactus fruit 1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula 1', 'Carrot 1', 'Cauliflower 1', 'Cherimoya 1', 'Cherry 1', 'Cherry 2', 'Cherry 3', 'Cherry 4', 'Cherry 5', 'Cherry Rainier 1', 'Cherry Rainier 2', 'Cherry Rainier 3', 'Cherry Sour 1', 'Cherry Wax Black 1', 'Cherry Wax not ripen 1', 'Cherry Wax not ripen 2', 'Cherry Wax Red 1', 'Cherry Wax Red 2', 'Cherry Wax Red 3', 'Cherry Wax Yellow 1', 'Chestnut 1', 'Clementine 1', 'Cocos 1', 'Corn 1', 'Corn Husk 1', 'Cucumber 1', 'Cucumber 10', 'Cucumber 11', 'Cucumber 3', 'Cucumber 4', 'Cucumber 5', 'Cucumber 7', 'Cucumber 9', 'Cucumber Ripe 1', 'Cucumber Ripe 2', 'Dates 1', 'Eggplant 1', 'Eggplant long 1', 'Fig 1', 'Ginger Root 1', 'Gooseberry 1', 'Granadilla 1', 'Grape Blue 1', 'Grape Pink 1', 'Grape White 1', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink 1', 'Grapefruit White 1', 'Guava 1', 'Hazelnut 1', 'Huckleberry 1', 'Kaki 1', 'Kiwi 1', 'Kohlrabi 1', 'Kumquats 1', 'Lemon 1', 'Lemon Meyer 1', 'Limes 1', 'Lychee 1', 'Mandarine 1', 'Mango 1', 'Mango Red 1', 'Mangostan 1', 'Maracuja 1', 'Melon Piel de Sapo 1', 'Mulberry 1', 'Nectarine 1', 'Nectarine Flat 1', 'Nectarine Flat 2', 'Nut 1', 'Nut 2', 'Nut 3', 'Nut 4', 'Nut 5', 'Nut Forest 1', 'Nut Pecan 1', 'Onion 2', 'Onion Red 1', 'Onion Red 2', 'Onion Red Peeled 1', 'Onion White 1', 'Onion White Peeled 1', 'Orange 1', 'Papaya 1', 'Passion Fruit 1', 'Peach 1', 'Peach 2', 'Peach 3', 'Peach 4', 'Peach 5', 'Peach 6', 'Peach Flat 1', 'Pear 1', 'Pear 10', 'Pear 11', 'Pear 12', 'Pear 13', 'Pear 2', 'Pear 3', 'Pear 5', 'Pear 6', 'Pear 7', 'Pear 8', 'Pear 9', 'Pear Abate 1', 'Pear Forelle 1', 'Pear Kaiser 1', 'Pear Monster 1', 'Pear Red 1', 'Pear Stone 1', 'Pear Williams 1', 'Pepino 1', 'Pepper Green 1', 'Pepper Orange 1', 'Pepper Red 1', 'Pepper Yellow 1', 'Physalis 1', 'Physalis with Husk 1', 'Pineapple 1', 'Pineapple Mini 1', 'Pistachio 1', 'Pitahaya Red 1', 'Plum 1', 'Plum 2', 'Plum 3', 'Pomegranate 1', 'Pomelo Sweetie 1', 'Potato Red 1', 'Potato Red Washed 1', 'Potato Sweet 1', 'Potato White 1', 'Quince 1', 'Quince 2', 'Quince 3', 'Quince 4', 'Rambutan 1', 'Raspberry 1', 'Redcurrant 1', 'Salak 1', 'Strawberry 1', 'Strawberry Wedge 1', 'Tamarillo 1', 'Tangelo 1', 'Tomato 1', 'Tomato 10', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato 5', 'Tomato 7', 'Tomato 8', 'Tomato 9', 'Tomato Cherry Maroon 1', 'Tomato Cherry Orange 1', 'Tomato Cherry Red 1', 'Tomato Cherry Red 2', 'Tomato Cherry Yellow 1', 'Tomato Heart 1', 'Tomato Maroon 1', 'Tomato Maroon 2', 'Tomato not Ripen 1', 'Tomato Yellow 1', 'Walnut 1', 'Watermelon 1', 'Zucchini 1', 'Zucchini dark 1']

model_ft = models.resnet50(weights=None)


try:
    num_classes = len(class_names)
except:
    print("Warning: class_names not properly defined. Assuming 223 classes.")
    num_classes = 223

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# Determine the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved best model weights
weights_path = weights_path = r'C:\Users\kuruk\Desktop\VS_Projects\Fruit_ML\best_pytorch_model_weights.pth'
try:
    model_ft.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✓ Model weights loaded successfully from {weights_path}")
except FileNotFoundError:
    print(f"ERROR: Weight file not found at {weights_path}")
    print("Please update the weights_path variable with the correct path.")
except Exception as e:
    print(f"ERROR loading model weights: {str(e)}")

# Set model to evaluation mode
model_ft.eval()
model_ft = model_ft.to(device)

print(f"✓ Model loaded on device: {device}")
print(f"✓ Number of classes: {num_classes}")

# Image preprocessing - ResNet50 standard preprocessing
target_size = (224, 224)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if files were uploaded
        if not request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        results = []
        
        # Process each uploaded file
        for key in request.files:
            file = request.files[key]
            
            # Read image from uploaded file
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Preprocess image
            img_tensor = preprocess(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = model_ft(img_tensor)
            
            # Get predicted class
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.item()
            
            # Get class name (handle case where we don't have all class names)
            if predicted_idx < len(class_names) and class_names[predicted_idx]:
                predicted_class = class_names[predicted_idx]
            else:
                predicted_class = f"Class_{predicted_idx}"
            
            # Get confidence scores (softmax probabilities)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx].item()
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities[0], min(5, num_classes))
            top5_predictions = []
            for prob, idx in zip(top5_prob, top5_idx):
                idx_val = idx.item()
                if idx_val < len(class_names) and class_names[idx_val]:
                    class_name = class_names[idx_val]
                else:
                    class_name = f"Class_{idx_val}"
                
                top5_predictions.append({
                    "class": class_name,
                    "confidence": float(prob.item())
                })
            
            results.append({
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": float(confidence),
                "top5": top5_predictions
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "ResNet50",
        "device": str(device),
        "num_classes": num_classes,
        "classes_defined": len(class_names)
    })

@app.route("/classes", methods=["GET"])
def get_classes():
    """Endpoint to view all available classes"""
    return jsonify({
        "num_classes": num_classes,
        "classes": class_names
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

