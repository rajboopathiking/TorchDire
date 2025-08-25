import os
import gc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split

# Trainer class

# find_unused_parameters=False,  # <-- default
#     check_reduction=False,
#     gradient_as_bucket_view=False,
#     static_graph=False

class Trainer:
    def __init__(self, model_fn, dataset_fn, df, train_fn,
                  collate_fn=None, batch_size=4, num_epochs=10):
        self.model_fn = model_fn
        self.dataset_fn = dataset_fn
        self.train_fn = train_fn
        self.world_size = torch.cuda.device_count()
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.df = df

    def setup_ddp(self, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="nccl", rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)

    def cleanup_ddp(self):
        dist.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()

    def run(self, rank):
        self.setup_ddp(rank)
        device = torch.device(f"cuda:{rank}")

        model = self.model_fn().to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        if self.df is not None:
            dataset = self.dataset_fn(self.df)
        else:
            dataset = self.dataset_fn()

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=4, collate_fn=self.collate_fn, pin_memory=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, sampler=val_sampler,
                                num_workers=4, collate_fn=self.collate_fn, pin_memory=True)

        try:
            self.train_fn(model=model, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=self.num_epochs)
        except Exception as e:
            print(f"[Rank {rank}] ❌ Error during training: {e}")
        finally:
            self.cleanup_ddp()

    def start(self):
        mp.spawn(self.run, args=(), nprocs=self.world_size, join=True)