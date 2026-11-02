import os
from datetime import timedelta

import torch
import torch.distributed as dist


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

print(
    f"[rank {rank}] "
    f"WORLD_SIZE={world_size} "
    f"LOCAL_RANK={local_rank} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"device_count={torch.cuda.device_count()}",
    flush=True,
)

torch.cuda.set_device(local_rank)

print(
    f"[rank {rank}] set cuda:{local_rank} "
    f"({torch.cuda.get_device_name(local_rank)})",
    flush=True,
)

print(
    f"[rank {rank}] BEFORE init_process_group "
    f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
    f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
    flush=True,
)

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=timedelta(minutes=2),
)

print(
    f"[rank {rank}] AFTER init_process_group",
    flush=True,
)

dist.barrier()

print(
    f"[rank {rank}] AFTER barrier",
    flush=True,
)

x = torch.tensor(
    [float(rank + 1)],
    device=f"cuda:{local_rank}",
)

dist.all_reduce(x)

print(
    f"[rank {rank}] all_reduce result={x.item()}",
    flush=True,
)

dist.destroy_process_group()

print(
    f"[rank {rank}] DDP TEST PASSED",
    flush=True,
)