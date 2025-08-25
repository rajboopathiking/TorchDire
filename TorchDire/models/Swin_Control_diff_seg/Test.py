from SwinControlDiffSeg import SwinControlDiffSeg
import torch
model = SwinControlDiffSeg(
    num_classes=4
)

model.eval()
with torch.no_grad():
    output = model(torch.randn(1,3,224,224))


assert output.shape[2:] == torch.Size([224, 224])
print("Test Success")

