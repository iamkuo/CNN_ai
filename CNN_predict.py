import torch
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義 CNN 模型
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.cnn2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 10)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

# 載入模型
def load_model():
    model = CNN_Model().to(device)
    model.load_state_dict(torch.load("D:\program\cnn_ai\model.pth", map_location=device, weights_only=True))
    model.eval()
    return model

# 預測函數
def predict(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# GUI 繪圖應用程式
class DigitRecognizer:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("手寫數字辨識")
        self.canvas = Canvas(self.root, width=280, height=280, bg="black")  # 初始背景為黑色
        self.canvas.pack()
        self.button_clear = Button(self.root, text="清除", command=self.clear_canvas)
        self.button_clear.pack()
        self.button_predict = Button(self.root, text="辨識", command=self.predict_digit)
        self.button_predict.pack()
        self.label_result = Label(self.root, text="請畫出一個數字")
        self.label_result.pack()
        
        self.image = Image.new("L", (28, 28), 0)  # 黑色背景 (灰度值 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)
        self.photo = None
        self.last_pos = None
        self.paint_count = 0
        self.paint_threshold = 3
    
    def paint(self, event):
        x, y = event.x, event.y
        current_pos = (x, y)
        
        if self.last_pos:
            x1, y1 = self.last_pos
            x2, y2 = current_pos
            x1_img, y1_img = x1 // 10, y1 // 10
            x2_img, y2_img = x2 // 10, y2 // 10
            self.draw.line([(x1_img, y1_img), (x2_img, y2_img)], fill=255, width=1)  # 白色線條
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance > 0:
                steps = int(distance / 2)
                for i in range(steps + 1):
                    t = i / max(steps, 1)
                    xi = x1_img + t * (x2_img - x1_img)
                    yi = y1_img + t * (y2_img - y1_img)
                    self.draw.ellipse([xi - 0.5, yi - 0.5, xi + 0.5, yi + 0.5], fill=255)
        else:
            self.draw.ellipse([x // 10, y // 10, (x + 10) // 10, (y + 10) // 10], fill=255)
        
        self.last_pos = current_pos
        
        self.paint_count += 1
        if self.paint_count >= self.paint_threshold:
            self.update_canvas_with_blur()
            self.paint_count = 0
    
    def reset_last_pos(self, event):
        self.last_pos = None
    
    def update_canvas_with_blur(self):
        img = self.image.copy()
        img_array = np.array(img, dtype=np.uint8)
        img_array = cv2.GaussianBlur(img_array, (3, 3), sigmaX=1.0, sigmaY=1.0)
        img_pil = Image.fromarray(img_array).resize((280, 280), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 0)  # 重置為黑色背景
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.configure(bg="black")  # 確保畫布為黑色
        self.photo = None
        self.last_pos = None
        self.paint_count = 0
        self.label_result.config(text="請畫出一個數字")
    
    def predict_digit(self):
        img = self.image.copy()
        img_array = np.array(img, dtype=np.uint8)
        img_array = cv2.GaussianBlur(img_array, (3, 3), sigmaX=1.0, sigmaY=1.0)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = self.center_image(img_array)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        predicted_label = predict(self.model, img_tensor)
        self.label_result.config(text=f"辨識結果: {predicted_label}")
    
    def center_image(self, img_array):
        y, x = np.where(img_array > 0.1)
        if len(x) == 0 or len(y) == 0:
            return img_array
        center_x, center_y = np.mean(x), np.mean(y)
        shift_x = 14 - center_x
        shift_y = 14 - center_y
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img_shifted = cv2.warpAffine(img_array, M, (28, 28), borderValue=0.0)
        return img_shifted

if __name__ == "__main__":
    model = load_model()
    root = Tk()
    app = DigitRecognizer(root, model)
    root.mainloop()