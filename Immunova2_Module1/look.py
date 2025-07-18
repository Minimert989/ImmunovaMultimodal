import torch
import torch.nn as nn
from torchvision.models import resnet18

# 커스텀 래퍼 클래스 정의 (당신이 훈련할 때 사용한 구조)
class TILClassifier(nn.Module):
    def __init__(self):
        super(TILClassifier, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # 1개 클래스 분류용

    def forward(self, x):
        return self.backbone(x)

# 모델 인스턴스화 및 state_dict 로드
model_path = "til_model.pth"
model = TILClassifier()
model.load_state_dict(torch.load(model_path, map_location="cpu"))

# 레이어 이름 출력
print("\n📌 모델의 레이어 이름:")
for name, module in model.named_modules():
    print(f"- {name} : {module.__class__.__name__}")