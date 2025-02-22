import torch
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets

# 載入 MNIST 測試資料集
(_, _), (X_test, Y_test) = datasets.mnist.load_data()
X_test = X_test.astype('float32') / 255  # 正規化數據

# 轉換為 PyTorch 張量
X_test_tensor = torch.from_numpy(X_test).unsqueeze(1)  # 增加通道維度
Y_test_tensor = torch.from_numpy(Y_test).to(torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義 CNN 模型 (需要與訓練時的架構相同)
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
    model.load_state_dict(torch.load("model.pth", map_location=device,weights_only=True))
    model.eval()  # 設定為推理模式
    return model

# 進行預測
def predict(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)  # 增加 batch 維度
    with torch.no_grad():
        output = model(image_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# 測試預測功能
def test_prediction():
    model = load_model()
    idx = np.random.randint(0, len(X_test))  # 隨機選取測試圖片
    image = X_test[idx]
    label = Y_test[idx]
    
    # 預測
    predicted_label = predict(model, X_test_tensor[idx])
    
    # 顯示圖片與預測結果
    plt.imshow(image, cmap="gray")
    plt.title(f"True Label: {label}, Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_prediction()
