### SWIN CONTROL DIFFUSION SEG

  Its a Segmentation Model Used For Multiclass/Multilabel Segmentation . Swin Transformer Used as a Encoder and ControlNet-bottleneck given then Unet Like Decoder Provide Segmentation Output. 

  Its Deterministic Model.


  ```bash
     pip install pytorch-dire
  ``` 

  Code Example :

  ```python

  from TorchDire.models.Swin_Control_diff_seg import SwinControlDiffSeg

  model = SwinControlDiffSeg(
   num_classes=4
  )

  ```

  Check Model Output Example:

  ```python

  from PIL import Image
  import torch
  from torchvision import transforms as T

  transform = T.Compose([
   T.Resize((256,256))
   T.Normalize(),
   T.ToTensor()
  ])

  image = Image.open("<image_path.png>").convert("RGB")

  image = transform(image)

  torch.eval()
  with torch.no_grad():
   outputs = model(image)

  ```


**Note If Need Pretrained Weights Use Pretrained=True as a argument in SwinControlDiffSeg**