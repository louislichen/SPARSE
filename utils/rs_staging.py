"""Ownership rules for asynchronous expert staging, independent of CUDA.

Speculation never writes the resident cache. Closing a target prevents any new
copy submissions; already-issued copies are handed to its demand executor.
The lock protects copy *submission*, never predictor execution or GPU completion.
"""
from dataclasses import dataclass
import threading


@dataclass
class Transfer:
    expert: int
    payload: object
    event: object
    slot: object
    nbytes: int
    used: bool = False
    retained: bool = False


class TargetJob:
    def __init__(self, ticket, slots):
        self.ticket = ticket
        self.slots = slots
        self.transfers = {}
        self.closed = False
        self.lock = threading.Lock()
        self.future = None
        self.plan_count = None

    def publish(self, expert, submit):
        """Atomically submit and publish one transfer, or decline after close.

        submit must use preallocated buffers and enqueue only; it must not
        synchronize a CUDA event, wait on a future, allocate, or plan candidates.
        A target cannot race a hidden late copy and load the same expert twice.
        """
        with self.lock:
            if self.closed:
                return False
            transfer = submit()
            assert transfer.expert == expert and expert not in self.transfers
            self.transfers[expert] = transfer
            return True

    def close(self):
        with self.lock:
            self.closed = True
            return dict(self.transfers)


class DemandCacheView:
    """Expose issued weights to existing demand code without speculative LRU.

    The caller supplies the copy events to the native demand wait path. has()
    means a payload exists (possibly in flight), exactly like the old cache.
    Resident capacity, eviction and hot-state operations stay on the real cache.
    Only actual accesses can authorize later demand-cache admission. The
    controller commits them after the target forward has submitted all users.
    """
    def __init__(self, resident):
        self.resident = resident
        self.staged = {}
        self.used_in_order = []

    def __getattr__(self, name):
        return getattr(self.resident, name)

    def has(self, expert):
        return self.resident.has(expert) or int(expert) in self.staged

    def get(self, expert):
        payload = self.resident.get(expert)
        if payload is not None:
            return payload
        transfer = self.staged.get(int(expert))
        if transfer is None:
            return None
        if not transfer.used:
            transfer.used = True
            self.used_in_order.append(transfer)
        return transfer.payload

    def peek(self, expert):
        # Peeking must not update either demand recency or speculative usage.
        payload = self.resident.peek(expert)
        if payload is not None:
            return payload
        transfer = self.staged.get(int(expert))
        return None if transfer is None else transfer.payload

    def resident_ids(self):
        return self.resident.resident_ids()


def select_candidates(ids, weights, *, allowed, resident, minimum, budget, diagnostics=None):
    """Original aggregate probability/reuse score, with no victim selection."""
    sums, counts, maxima = {}, {}, {}
    nonlocal_entries = resident_entries = 0
    for eid, weight in zip(ids, weights):
        eid, weight = int(eid), float(weight)
        if eid not in allowed:
            nonlocal_entries += 1
            continue
        if eid in resident:
            resident_entries += 1
            continue
        sums[eid] = sums.get(eid, 0.0) + weight
        counts[eid] = counts.get(eid, 0) + 1
        maxima[eid] = max(maxima.get(eid, 0.0), weight)
    scores = {eid: value * (1 + .15 * (counts[eid] - 1)) + .05 * maxima[eid]
              for eid, value in sums.items()}
    if diagnostics is not None:
        diagnostics.update(nonlocal_entries=nonlocal_entries, resident_entries=resident_entries,
                           scored=len(scores), below_threshold=sum(v < minimum for v in scores.values()))
    return [eid for eid in sorted(scores, key=scores.get, reverse=True)
            if scores[eid] >= minimum][:budget]
