"""Bounded next-layer RS pipeline for layer-local CPU-offloaded experts.

Install explicitly on a fresh service version, then prepare_graphs before the
measured requests. begin/finish must remain inside the service wall timer.
No source-return join, speculative resident eviction, or duplicate demand copy.
An issued expert may wait only at its actual use, via the native copy event.

The four-slot pool is reserved for both ON/OFF controls. Hardware interference
and the brief copy-publication lock are measured costs, not assumed to be zero.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

try:
    from .rs_staging import DemandCacheView, TargetJob, Transfer, select_candidates
except ImportError:  # Private standalone deployment alongside this file.
    from rs_staging import DemandCacheView, TargetJob, Transfer, select_candidates


class GraphPrediction:
    def __init__(self, torch, model, batch, width, k, dtype, normalize, device):
        self.torch, self.model = torch, model
        self.k, self.dtype, self.normalize = k, dtype, normalize
        self.input = torch.zeros((batch, width), device=device, dtype=torch.float32)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.compute(self.input)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.ids, self.values = self.compute(self.input)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.validations = 0

    def compute(self, x):
        t = self.torch
        values, ids = t.topk(self.model(x).to(t.float32), k=self.k, dim=-1)
        if self.normalize:
            values = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
        return ids.to(t.long), values.to(self.dtype)

    def __call__(self, x):
        self.input.copy_(x)
        self.graph.replay()
        return self.ids, self.values

    def verify(self, x):
        expected_ids, expected_values = self.compute(x.float())
        ids, values = self(x.float())
        assert self.torch.equal(ids, expected_ids)
        assert self.torch.equal(values, expected_values)
        self.validations += 1


class BoundedRS:
    def __init__(self, core, *, slots=4, empty_steps=16):
        import torch
        self.torch, self.core = torch, core
        self.enabled = False
        self.slot_count, self.empty_steps = int(slots), int(empty_steps)
        assert self.slot_count >= 2 and self.empty_steps >= 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rs-staging')
        self.graphs, self.buffers, self.jobs, self.retired = {}, {}, {}, []
        self.slots, self.views, self.templates = [], {}, {}
        self.next_probe, self.allowed = {}, {}
        self.pending_completion_metrics = []
        self.pending_accuracy = {}
        self.epoch, self.stream, self.copy_stream = 0, None, None
        self.owner = threading.get_ident()
        self.originals, self.wrappers = {}, {}
        self.stats = {}

    def _weights(self, block, eid):
        if block.experts_kind == 'fused':
            return block.local_gate_up_cpu[eid], block.local_down_cpu[eid]
        return block.local_gate_cpu[eid], block.local_up_cpu[eid], block.local_down_cpu2[eid]

    def prepare_graphs(self, *, batch, dtype, k):
        t, ctx = self.torch, self.core.SparseContext
        assert not self.jobs and not self.retired
        for layer, block in sorted(ctx.layer_blocks.items()):
            assert not block._global_cache_enabled()
            assert block.expert_offload == 'cpu' and block.cache_policy == 'lru'
            assert block._light_policy is None
            assert not block._transition_prefetch_enabled and not block._moe_infinity_policy_enabled
            if layer + 1 >= block.num_layers:
                continue
            model = ctx.rs_bank[layer + 1]
            assert model.rs_feature_mode == 'hidden'
            graph = GraphPrediction(t, model, batch, block.hidden_size, k, dtype,
                                    block.norm_topk_prob, block.device)
            for seed in range(3):
                x = t.arange(batch * block.hidden_size, device=block.device, dtype=t.float32).reshape(batch, -1)
                graph.verify(((x % (97 + seed)) - 48) / 30)
            self.graphs[layer, k] = graph
            self.buffers[layer, k] = (
                t.empty((batch, k), dtype=t.long, pin_memory=True),
                t.empty((batch, k), dtype=dtype, pin_memory=True),
                t.cuda.Event(blocking=False))
        if not self.slots:
            blocks = list(ctx.layer_blocks.values())
            block = blocks[0]
            template = self._weights(block, next(iter(block._local_eid_set)))
            signature = [(x.shape, x.dtype) for x in template]
            for item in blocks:
                self.templates[item.layer_idx] = {eid: self._weights(item, eid) for eid in item._local_eid_set}
                for weights in self.templates[item.layer_idx].values():
                    assert [(x.shape, x.dtype) for x in weights] == signature
                    assert all(x.is_pinned() for x in weights), 'Asynchronous H2D requires pinned weights'
            size = sum(x.numel() * x.element_size() for x in template)
            # Retain room for a demand/EPIPE expert and its configured floor.
            floors = [max(float(getattr(b, '_epipe_min_free_mb', 0)),
                          float(b._demand_h2d_min_free_mb())) for b in blocks]
            free, _ = t.cuda.mem_get_info(block.device)
            assert free >= size * (self.slot_count + 1) + (max(floors) + 32) * 2**20
            for _ in range(self.slot_count):
                payload = tuple(t.empty_like(x, device=block.device) for x in template)
                self.slots.append(dict(payload=payload, event=t.cuda.Event(blocking=False),
                                       retire=None, busy=False, nbytes=size))
            self.stream = t.cuda.Stream(device=block.device, priority=0)
            self.copy_stream = t.cuda.Stream(device=block.device, priority=0)
        t.cuda.synchronize()

    def begin(self, case):
        assert not self.jobs and not self.retired and not self.views
        self.epoch += 1
        self.case, self.next_probe = case, {}
        assert not self.pending_completion_metrics
        assert not self.pending_accuracy
        self.stats = dict(predictions=0, plans=0, source_joins=0, issued=0, used=0,
                          wasted=0, closed_before_publish=0, skipped_slots=0,
                          skipped_cooldown=0, worker_ms=0., readback_wait_ms=0.,
                          main_submission_ms=0., target_close_ms=0., request_drain_ms=0.,
                          unified_epipe_calls=0,
                          demand_deadline_activations=0,
                          empty_plan_closes=0, unplanned_closes=0,
                          planned_no_publish_closes=0,
                          retained=0, recycled_payloads=0, allocated_replacements=0,
                          staging_bytes=sum(s['nbytes'] for s in self.slots))
        if not self.enabled:
            return
        ctx = self.core.SparseContext
        for layer, block in ctx.layer_blocks.items():
            assert block.expert_source == 'router' and block.exec_routing_mode == 'router'
            assert ctx.oracle_mode == 'none' and ctx.rs_bank_layout == 'next_layer'
            assert not block.scp_apply_decode, 'Prepared ownership assumes fixed RR decode'
            view = DemandCacheView(block._cache)
            self.views[layer] = view
            block._cache = view
            # The supported deployment decodes with RR owners. Resolve outside
            # the per-layer/per-candidate hot path, with explicit decode stage.
            self.allowed[layer] = frozenset(eid for eid in block._local_eid_set
                if block.resolve_prefetch_target_rank(layer_idx=layer, source_rank=block.comm_rank,
                    expert_id=eid, stage='decode', count_metrics=False,
                    reason='rs_staging_prepare').target_rank == block.comm_rank)

    def _retire(self, slot, event):
        assert slot['busy']
        slot['busy'], slot['retire'] = False, event

    def _collect(self):
        keep = []
        for job, transferred in self.retired:
            if not job.future.done():
                keep.append((job, transferred))
                continue
            row = job.future.result()  # Already completed: never a layer join.
            self.stats['worker_ms'] += row['worker_ms']
            self.stats['readback_wait_ms'] += row['readback_wait_ms']
            used_slots = {id(x.slot) for x in transferred.values()}
            for slot in job.slots:
                if id(slot) not in used_slots:
                    self._retire(slot, None)
        self.retired = keep

    def _reserve(self, budget):
        self._collect()
        available = [slot for slot in self.slots if not slot['busy']
                     and (slot['retire'] is None or slot['retire'].query())]
        selected = available[:budget]
        for slot in selected:
            slot['busy'], slot['retire'] = True, None
        return selected

    def _work(self, job, target, ids, weights, readback, resident, minimum):
        t = self.torch
        started = time.perf_counter()
        waited = 0.
        # Snapshot-only planning. No live cache or metrics mutation here.
        with t.cuda.device(target.device), t.inference_mode():
            stamp = time.perf_counter()
            readback.synchronize()
            waited = (time.perf_counter() - stamp) * 1000
            if not job.closed:
                chosen = select_candidates(ids.reshape(-1).tolist(), weights.float().reshape(-1).tolist(),
                    allowed=self.allowed[target.layer_idx], resident=resident,
                    minimum=minimum, budget=len(job.slots))
                job.plan_count = len(chosen)
                for eid, slot in zip(chosen, job.slots):
                    source = self.templates[target.layer_idx][eid]
                    def submit(eid=eid, slot=slot, source=source):
                        with t.cuda.stream(self.copy_stream):
                            for dest, cpu in zip(slot['payload'], source):
                                dest.copy_(cpu, non_blocking=True)
                            slot['event'].record(self.copy_stream)
                        return Transfer(eid, slot['payload'], slot['event'], slot, slot['nbytes'])
                    if not job.publish(eid, submit):
                        break
        return dict(worker_ms=(time.perf_counter() - started)*1000, readback_wait_ms=waited)

    def predict(self, block, x, seq_len, source_idx=None, source_w=None):
        if not self.enabled or seq_len != 1 or block.planner_mode != 'rs_prefetch':
            return self.originals['_compute_and_schedule_next_layer_rs'](
                block, x, seq_len, source_idx=source_idx, source_w=source_w)
        ctx, t = self.core.SparseContext, self.torch
        target_layer = int(block.layer_idx) + 1
        if target_layer >= block.num_layers:
            return
        assert threading.get_ident() == self.owner
        assert ctx.rs_bank_source_layers[target_layer] == block.layer_idx
        target = ctx.layer_blocks[target_layer]
        step = int(ctx.runtime_call_id)
        if step < self.next_probe.get(target_layer, 0):
            self.stats['skipped_cooldown'] += 1
            return
        # Do not reuse graph/host buffers while an old worker can still read.
        self._collect()
        if target_layer in self.jobs or any(j.ticket[2] == target_layer for j, _ in self.retired):
            self.stats['skipped_slots'] += 1
            return
        budget = min(int(target._effective_rs_prefetch_top_m()), self.slot_count // 2)
        if budget <= 0:
            return
        slots = self._reserve(budget)
        if not slots:
            self.stats['skipped_slots'] += 1
            return
        stamp = time.perf_counter()
        k = min(block.num_experts, target._rs_prefetch_predict_topk())
        graph = self.graphs[block.layer_idx, k]
        host_ids, host_weights, event = self.buffers[block.layer_idx, k]
        assert tuple(x.shape) == tuple(graph.input.shape)
        self.stream.wait_stream(t.cuda.current_stream(block.device))
        x.record_stream(self.stream)
        with t.cuda.stream(self.stream), t.inference_mode():
            ids, weights = graph(x)
            host_ids.copy_(ids, non_blocking=True)
            host_weights.copy_(weights, non_blocking=True)
            event.record(self.stream)
        job = TargetJob((self.epoch, step, target_layer), slots)
        self.jobs[target_layer] = job
        resident = frozenset(self.views[target_layer].resident_ids())
        job.future = self.pool.submit(self._work, job, target, host_ids, host_weights, event,
                                      resident, float(target._prefetch_min_candidate_score))
        if block.metrics is not None:
            block.metrics.rs_next_layer_pred_calls += 1
        self.stats['predictions'] += 1
        self.stats['main_submission_ms'] += (time.perf_counter()-stamp)*1000

    def activate(self, block):
        layer = int(block.layer_idx)
        view = self.views[layer]
        assert not view.staged and not view.used_in_order
        job = self.jobs.pop(layer, None)
        if job is None:
            return
        stamp = time.perf_counter()
        transfers = job.close()
        self.stats['target_close_ms'] += (time.perf_counter()-stamp)*1000
        ticket = (self.epoch, int(self.core.SparseContext.runtime_call_id), layer)
        assert job.ticket == ticket, 'Stale prediction crossed a request/step boundary'
        self.retired.append((job, transfers))
        if job.plan_count == 0:
            self.next_probe[layer] = ticket[1] + self.empty_steps + 1
        if not transfers:
            self.stats['closed_before_publish'] += 1
            if job.plan_count is None:
                self.stats['unplanned_closes'] += 1
            elif job.plan_count == 0:
                self.stats['empty_plan_closes'] += 1
            else:
                self.stats['planned_no_publish_closes'] += 1
        view.staged = transfers
        block._prefetch_current_group_start = layer
        block._prefetch_current_group_end = layer + 1
        self.stats['plans'] += bool(transfers)
        for eid, transfer in transfers.items():
            block._prefetched_ids_this_group.add(eid)
            block._prefetch_events[eid] = transfer.event
            block._record_h2d_bytes(transfer.nbytes, is_prefetch=True)
            block._record_rs_counter('prefetch_issued_count', 1)
            block._record_rs_counter('prefetch_admitted_count', 1)
            self.stats['issued'] += 1

    def activate_at_demand(self, block):
        """Keep publication open through router and EP dispatch, without a join."""
        self.activate(block)
        self.stats['demand_deadline_activations'] += 1
        actual = self.pending_accuracy.pop(int(block.layer_idx), None)
        if actual is not None:
            idx, seq_len = actual
            self.originals['_record_prefetch_eval_actual'](block, idx, seq_len)

    def _retain_used(self, block, transfer):
        """Commit a real demand after the layer; exchange ownership, not bytes.

        Keeping a successful prefetch transient forever skips the demand LRU
        update and can cause future misses. The old resident victim supplies
        the next scratch payload. Its storage is protected by the consumer
        event recorded after this entire forward, just like the used payload.
        A free cache entry needs one replacement allocation, as ordinary demand
        would, but no additional H2D or D2D transfer.
        """
        assert transfer.used and not transfer.retained
        resident = self.views[int(block.layer_idx)].resident
        before = {eid: resident.peek(eid) for eid in resident.resident_ids()}
        block._on_demand_access(transfer.expert)
        block._cache_put_guarded(transfer.expert, transfer.payload, pinned=False,
                                 prefer_hot=block._should_prefer_hot(transfer.expert, False))
        if not resident.has(transfer.expert):
            return
        evicted = [eid for eid in before if not resident.has(eid)]
        assert len(evicted) <= 1, 'Only a layer-local plain LRU is supported'
        if evicted:
            replacement = before[evicted[0]]
            self.stats['recycled_payloads'] += 1
        else:
            replacement = tuple(self.torch.empty_like(x) for x in transfer.payload)
            self.stats['allocated_replacements'] += 1
        assert all(x.shape == y.shape and x.dtype == y.dtype
                   for x, y in zip(replacement, transfer.payload))
        transfer.slot['payload'] = replacement
        transfer.retained = True
        self.stats['retained'] += 1

    def release(self, block):
        view = self.views[int(block.layer_idx)]
        t = self.torch
        used = [x for x in view.staged.values() if x.used]
        done = None
        if used:
            # Record after all consumers on the demand stream. A pool slot is
            # never reused merely because its earlier H2D event completed.
            done = t.cuda.Event(blocking=False)
            done.record(t.cuda.current_stream(block.device))
        # All current-layer consumers are submitted now, so true-demand LRU
        # admission cannot evict a resident expert before this forward uses it.
        for transfer in view.used_in_order:
            self._retain_used(block, transfer)
        for eid, transfer in view.staged.items():
            self._retire(transfer.slot, done if transfer.used else transfer.event)
            self.stats['used' if transfer.used else 'wasted'] += 1
            if not transfer.used:
                block._record_prefetch_wasted(1)
                self.pending_completion_metrics.append(block)
            block._prefetch_events.pop(eid, None)
            block._prefetched_ids_this_group.discard(eid)
            block._prefetch_used_ids_this_group.discard(eid)
        view.staged = {}
        view.used_in_order = []

    def finish(self):
        stamp = time.perf_counter()
        for layer, job in list(self.jobs.items()):
            # Normally every target is visited. Exceptions must still retain
            # in-flight storage and prevent a late background cache mutation.
            transfers = job.close()
            self.retired.append((job, transfers))
            for transfer in transfers.values():
                self._retire(transfer.slot, transfer.event)
        self.jobs.clear()
        self.pending_accuracy.clear()
        for job, _ in self.retired:
            job.future.result(timeout=30)
        self._collect()
        if self.stream is not None:
            self.stream.synchronize()
            self.copy_stream.synchronize()
        for layer, view in self.views.items():
            block = self.core.SparseContext.layer_blocks[layer]
            if view.staged:
                self.release(block)
            assert block._cache is view
            block._cache = view.resident
        self.views = {}
        for block in self.pending_completion_metrics:
            block._record_rs_counter('prefetch_completed_count', 1)
        self.pending_completion_metrics = []
        self.stats['request_drain_ms'] = (time.perf_counter()-stamp)*1000
        return dict(case=self.case, **self.stats)

    def install(self):
        cls, owner = self.core.DistributedSparseMoeBlock, self
        for name in ('_compute_and_schedule_next_layer_rs', 'forward',
                     '_record_prefetch_eval_actual', '_execute_local_recv_groups',
                     '_experts_world1_exact',
                     '_try_execute_local_recv_groups_epipe_rs_residual_scan'):
            self.originals[name] = getattr(cls, name)
        def predict(block, *args, **kwargs):
            return owner.predict(block, *args, **kwargs)
        def forward(block, hidden_states, *args, **kwargs):
            active = owner.enabled and hidden_states.shape[1] == 1
            try:
                return owner.originals['forward'](block, hidden_states, *args, **kwargs)
            finally:
                if active:
                    # A path that never reaches an expert executor must still
                    # close its job before releasing the layer's state.
                    if int(block.layer_idx) in owner.jobs:
                        owner.activate_at_demand(block)
                    owner.release(block)
        def record(block, idx, seq_len):
            if owner.enabled and seq_len == 1:
                layer = int(block.layer_idx)
                assert layer not in owner.pending_accuracy
                owner.pending_accuracy[layer] = (idx.detach(), seq_len)
                return
            return owner.originals['_record_prefetch_eval_actual'](block, idx, seq_len)
        def local_execute(block, *args, **kwargs):
            if owner.enabled and block._current_stage() == 'decode':
                owner.activate_at_demand(block)
            return owner.originals['_execute_local_recv_groups'](block, *args, **kwargs)
        def world1_execute(block, *args, **kwargs):
            if owner.enabled and block._current_stage() == 'decode':
                owner.activate_at_demand(block)
            return owner.originals['_experts_world1_exact'](block, *args, **kwargs)
        self.upfront = cls._try_execute_local_recv_groups_epipe
        def execute(block, *args, **kwargs):
            if owner.enabled and block._current_stage() == 'decode':
                owner.stats['unified_epipe_calls'] += 1
                return owner.upfront(block, *args, **kwargs)
            return owner.originals['_try_execute_local_recv_groups_epipe_rs_residual_scan'](block, *args, **kwargs)
        self.wrappers = dict(_compute_and_schedule_next_layer_rs=predict, forward=forward,
                            _record_prefetch_eval_actual=record,
                            _execute_local_recv_groups=local_execute,
                            _experts_world1_exact=world1_execute,
                            _try_execute_local_recv_groups_epipe_rs_residual_scan=execute)
        for name, fn in self.wrappers.items():
            setattr(cls, name, fn)

    def restore(self):
        assert not self.jobs and not self.retired and not self.views
        self.pool.shutdown(wait=True)
        cls = self.core.DistributedSparseMoeBlock
        for name, fn in self.originals.items():
            assert getattr(cls, name) is self.wrappers[name]
            setattr(cls, name, fn)
