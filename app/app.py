import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Thư mục lưu ảnh
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')  # Chuẩn hóa đường dẫn
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Thiết bị
device = torch.device("cpu")

# Định nghĩa CNN
class LungDiseaseCNN(nn.Module):
    def __init__(self):
        super(LungDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Tải mô hình
model = LungDiseaseCNN().to(device)
model.load_state_dict(torch.load('model/cnn_lung_disease_model.pth', map_location=device))
model.eval()

# Biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        try:
            img = Image.open(file).convert('L')
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            return f"Error opening image: {str(e)}", 400
        
        # Tạo tên file duy nhất với timestamp
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Lưu ảnh và kiểm tra
        try:
            img.save(img_path)
            if os.path.exists(img_path):
                print(f"Saved image to: {img_path}")
            else:
                print(f"Failed to save image to: {img_path}")
                return "Failed to save image", 500
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return f"Error saving image: {str(e)}", 500
        
        # Chuẩn bị ảnh cho CNN
        try:
            img_tensor = transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error transforming image: {str(e)}")
            return f"Error transforming image: {str(e)}", 500
        
        # Dự đoán
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, dim=1).item()
        
        result = 'Normal' if prediction == 0 else 'Pneumonia'
        prob_normal = probabilities[0].item() * 100
        prob_pneumonia = probabilities[1].item() * 100
        
        print(f"Rendering result with image_file: {filename}")
        return render_template('result.html', 
                             result=result, 
                             prob_normal=prob_normal, 
                             prob_pneumonia=prob_pneumonia, 
                             image_file=filename)
    return render_template('upload.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    print(f"Serving file: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return "File not found", 404
    print(f"File found, serving: {file_path}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)