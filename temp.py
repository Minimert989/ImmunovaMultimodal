import torch

# 1. 파일 경로
pt_path = "Immunova2_Module1/acc/TCGA-OR-A5J5.pt"

# 2. 로드 시도
try:
    data = torch.load(pt_path)
    print("📦 로드 성공!")
    print("▶ 포함된 키들:", list(data.keys()))
    
    if "images" not in data:
        print("❌ 'images' 키가 없습니다.")
    else:
        patches = data["images"]
        print("✅ 'images' 키 존재")
        print("📐 이미지 배열 shape:", patches.shape)
        print("📈 dtype:", patches.dtype)
        print("🔍 첫 번째 patch 요약:", patches[0].shape if len(patches) > 0 else "비어 있음")
except Exception as e:
    print("❌ 파일 로드 실패:", e)