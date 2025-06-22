# ⚙️ DDP (Distributed Data Parallel Processing)

This module simplifies the setup and training of models using **Distributed Data Parallel (DDP)** in PyTorch. DDP provides efficient, scalable multi-GPU training—far superior to `torch.nn.DataParallel`, which often suffers from performance bottlenecks and bugs in complex models.

---

## 🔍 Why Use DDP Instead of DataParallel?

While `DataParallel` splits batches across GPUs, it **still uses one process and one main GPU**, which creates a communication bottleneck. DDP, on the other hand:

- Launches **one process per GPU**
- Allows **true parallelism** with better performance
- Is **more stable** for complex models and large datasets
- Works with mixed precision and model checkpointing easily

---

## 📁 Project Structure

To use this module effectively, you should structure your project like this:

.
├── train.py # Training logic (train_epoch, eval_epoch, etc.)
├── model_fn.py # Function that returns the model (get_model)
├── dataset_fn.py # Custom dataset class or loading logic
├── DDP.py # Contains the Trainer class for DDP
├── any_dataframe.csv # Optional: dataset info if needed
└── main.py # Entrypoint to run DDP training

yaml
Copy
Edit

> **Note:** Each component must be in a separate `.py` file due to how PyTorch DDP spawns subprocesses.
This is a PyTorch requirement, not a limitation of this module.

---

## 🚀 How to Use

### 1. Define the Required Components

Create these Python files:

- `model_fn.py` → contains `get_model()` that returns your model
- `dataset_fn.py` → contains a class or function that returns your `Dataset`
- `train.py` → contains `train(model, train_loader, val_loader, device, num_epochs)`

---

### 2. Create `main.py`

```python
from DDP import Trainer
from model_fn import get_model
from dataset_fn import YourDataset
from train import train
import pandas as pd

df = pd.read_csv("any_dataframe.csv")  # Optional if your dataset needs it

trainer = Trainer(
    model_fn=get_model,
    dataset_fn=YourDataset,
    df=df,  # or None
    train_fn=train,
    batch_size=4,
    num_epochs=20
)

trainer.start()
🧠 Tips
❗ Avoid defining functions inline or inside if __name__ == "__main__"

DDP spawns subprocesses that require importable top-level functions and classes.

✅ If you get errors like "Expected to have finished reduction...", set find_unused_parameters=True in DDP(...).

🧪 Enable debugging:

bash
Copy
Edit
TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py
