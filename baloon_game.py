import torch
import numpy as np
from keras import datasets
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import cv2
import random
import time

# 載入 MNIST 測試資料集
(_, _), (X_test, Y_test) = datasets.mnist.load_data()
X_test = X_test.astype('float32') / 255

# 轉換為 PyTorch 張量
X_test_tensor = torch.from_numpy(X_test).unsqueeze(1)
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

# 遊戲類
class BalloonDigitGame:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("氣球數字遊戲")
        self.canvas = Canvas(self.root, width=560, height=560, bg="black")  # 畫布放大到 560x560
        self.canvas.pack()
        self.button_clear = Button(self.root, text="清除", command=self.clear_canvas)
        self.button_clear.pack(side=LEFT)
        self.button_predict = Button(self.root, text="辨識", command=self.predict_digit)
        self.button_predict.pack(side=LEFT)
        self.button_restart = Button(self.root, text="重新開始", command=self.restart_game)
        self.button_restart.pack(side=LEFT)
        self.label_score = Label(self.root, text="分數: 0")
        self.label_score.pack(side=LEFT)
        self.label_result = Label(self.root, text="請畫出數字打破氣球")
        self.label_result.pack(side=LEFT)
        
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)
        self.photo = None
        self.last_pos = None
        self.paint_count = 0
        self.paint_threshold = 3
        self.score = 0
        self.balloons = []  # 儲存氣球的 (id, text_id, x, y, number)
        self.game_running = True
        self.base_speed = 20  # 基礎速度 5 像素/秒
        self.last_update = time.time()
        self.spawn_balloon()
        self.update_balloons()

    def spawn_balloon(self):
        if len(self.balloons) < 5 and self.game_running:  # 最多 5 個氣球
            existing_xs = [balloon[2] for balloon in self.balloons]
            for _ in range(10):
                x = random.randint(50, 510)  # 適應 560 寬度
                if all(abs(x - ex) >= 80 for ex in existing_xs):  # 間距 80 像素
                    break
            else:
                return
            number = random.randint(0, 9)
            balloon_id = self.canvas.create_oval(x-40, 0, x+40, 80, fill="red")  # 氣球 80x80
            text_id = self.canvas.create_text(x, 40, text=str(number), fill="white", font=("Arial", 24))
            self.balloons.append([balloon_id, text_id, x, 0, number])
            self.root.after(1000, self.spawn_balloon)  # 每 1 秒生成新氣球

    def update_balloons(self):
        if not self.game_running:
            return
        current_time = time.time()
        elapsed = current_time - self.last_update
        speed = self.base_speed + self.score // 50
        pixels_to_move = elapsed * speed
        self.last_update = current_time
        
        for balloon in self.balloons[:]:
            balloon_id, text_id, x, y, number = balloon
            y += pixels_to_move
            balloon[3] = y
            self.canvas.coords(balloon_id, x-40, y, x+40, y+80)
            self.canvas.coords(text_id, x, y+40)
            if y >= 560:  # 適應新畫布高度
                self.game_over()
                return
        self.root.after(50, self.update_balloons)

    def game_over(self):
        self.game_running = False
        self.canvas.delete("all")
        self.canvas.create_text(280, 280, text=f"遊戲結束！最終分數: {self.score}", fill="white", font=("Arial", 24))
        self.label_result.config(text="遊戲結束")

    def paint(self, event):
        if not self.game_running:
            return
        x, y = event.x, event.y
        current_pos = (x, y)
        
        if self.last_pos:
            x1, y1 = self.last_pos
            x2, y2 = current_pos
            x1_img, y1_img = x1 // 20, y1 // 20  # 適應 560x560，縮放到 28x28
            x2_img, y2_img = x2 // 20, y2 // 20
            self.draw.line([(x1_img, y1_img), (x2_img, y2_img)], fill=255, width=1)
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance > 0:
                steps = int(distance / 4)  # 適應更大畫布，增加插值密度
                for i in range(steps + 1):
                    t = i / max(steps, 1)
                    xi = x1_img + t * (x2_img - x1_img)
                    yi = y1_img + t * (y2_img - y1_img)
                    self.draw.ellipse([xi - 0.5, yi - 0.5, xi + 0.5, yi + 0.5], fill=255)
        else:
            self.draw.ellipse([x // 20, y // 20, (x + 20) // 20, (y + 20) // 20], fill=255)
        
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
        img_pil = Image.fromarray(img_array).resize((560, 560), Image.Resampling.LANCZOS)  # 放大到 560x560
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("drawn_image")
        self.canvas.create_image(0, 0, anchor=NW, image=self.photo, tags="drawn_image")
    
    def clear_canvas(self):
        if not self.game_running:
            return
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("drawn_image")
        self.canvas.configure(bg="black")
        self.photo = None
        self.last_pos = None
        self.paint_count = 0
        self.label_result.config(text="請畫出數字打破氣球")
    
    def predict_digit(self):
        if not self.game_running:
            return
        img = self.image.copy()
        img_array = np.array(img, dtype=np.uint8)
        img_array = cv2.GaussianBlur(img_array, (3, 3), sigmaX=1.0, sigmaY=1.0)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = self.center_image(img_array)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        predicted_label = predict(self.model, img_tensor)
        
        # 檢查是否匹配氣球數字
        for balloon in self.balloons[:]:
            if balloon[4] == predicted_label:
                self.canvas.delete(balloon[0])
                self.canvas.delete(balloon[1])
                self.balloons.remove(balloon)
                self.score += 10
                self.label_score.config(text=f"分數: {self.score}")
                self.label_result.config(text=f"擊破！數字: {predicted_label}")
                # 立即檢查並生成新氣球
                if len(self.balloons) < 5:
                    self.spawn_balloon()
                break
        else:
            self.label_result.config(text=f"辨識結果: {predicted_label}，無匹配氣球")
        
        self.clear_canvas()

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
    
    def restart_game(self):
        self.game_running = True
        self.score = 0
        self.label_score.config(text="分數: 0")
        self.label_result.config(text="請畫出數字打破氣球")
        self.balloons = []
        self.canvas.delete("all")
        self.canvas.configure(bg="black")
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.photo = None
        self.last_pos = None
        self.paint_count = 0
        self.last_update = time.time()
        self.spawn_balloon()
        self.update_balloons()

if __name__ == "__main__":
    model = load_model()
    root = Tk()
    app = BalloonDigitGame(root, model)
    root.mainloop()