"""Check physical NCCL collectives before a multi-node serving experiment."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=90))
try:
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    value = torch.tensor([rank + 1.], device=device)
    dist.all_reduce(value)
    assert value.item() == world * (world + 1) / 2
    measurements = []
    for chunk in (1024, 65536, 1048576):
        # Integer payloads make transport correctness checks exact.
        original = torch.arange(world * chunk, device=device, dtype=torch.int64) + rank * world * chunk
        received = torch.empty_like(original)
        restored = torch.empty_like(original)
        expected = torch.cat([torch.arange(chunk, device=device, dtype=torch.int64)
            + sender * world * chunk + rank * chunk for sender in range(world)])
        dist.all_to_all_single(received, original)
        assert torch.equal(received, expected), "all-to-all rank permutation is incorrect"
        dist.all_to_all_single(restored, received)
        assert torch.equal(restored, original), "all-to-all round trip is incorrect"
        elapsed = []
        for _ in range(3):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_to_all_single(received, original)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
        measurements.append({"chunk_items": chunk, "item_bytes": 8,
            "alltoall_wall_ms": elapsed, "transport_correct": True})
    report = {"rank": rank, "world": world, "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "nccl": torch.cuda.nccl.version(), "socket_interface": os.environ.get("NCCL_SOCKET_IFNAME"),
        "measurements": measurements, "status": "PASS"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"transport.rank{rank}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
finally:
    dist.destroy_process_group()
