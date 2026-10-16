"""Deterministic communication and LRU replay for SCP's offline guards.

Replay estimates are planning proxies, never substitutes for timed serving.
An invocation groups all source ranks; source-specific request IDs must not
split a collective into separate invocations. Expert requests are coalesced at
the destination before cache access, as in the runtime's expert executor.
"""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np


def load_profile_with_trace(path):
    with np.load(Path(path), allow_pickle=False) as data:
        profile = {key:np.asarray(data[key]).copy() for key in data.files if key != 'metadata_json'}
        profile['metadata'] = json.loads(str(data['metadata_json'].item())) if 'metadata_json' in data else {}
    required = ('trace_stage','trace_call_id','trace_step','trace_layer','trace_source_rank','trace_expert','trace_count')
    if any(key not in profile for key in required):
        raise ValueError('ordered routing trace is incomplete')
    n = len(profile['trace_count'])
    for key in required:
        if profile[key].ndim != 1 or len(profile[key]) != n:
            raise ValueError('trace arrays must be aligned vectors')
    if np.any(profile['trace_count'] < 0):
        raise ValueError('negative trace item count')
    prefill, decode = profile['A_prefill'], profile['A_decode']
    if prefill.ndim != 3 or prefill.shape != decode.shape:
        raise ValueError('profile demand shape mismatch')
    layers,world,experts = prefill.shape
    for key,limit in [('trace_layer',layers),('trace_source_rank',world),('trace_expert',experts)]:
        if np.any(profile[key]<0) or np.any(profile[key]>=limit):
            raise ValueError('out-of-range '+key)
    if not np.isin(profile['trace_stage'],[0,1]).all():
        raise ValueError('unknown trace stage')
    for stage,expected in [(0,prefill),(1,decode)]:
        observed=np.zeros_like(expected,dtype=np.int64)
        mask=profile['trace_stage']==stage
        np.add.at(observed,(profile['trace_layer'][mask],profile['trace_source_rank'][mask],profile['trace_expert'][mask]),profile['trace_count'][mask])
        if not np.array_equal(observed,expected):
            raise ValueError('ordered trace and aggregate demand disagree')
    return profile


def _stage_id(stage):
    if stage in ('prefill',0):return 0
    if stage in ('decode',1):return 1
    if stage in ('mixed','all',None,''):return None
    raise ValueError('unknown stage '+str(stage))


def grouped_trace_indices(profile,stage='prefill'):
    n=len(profile['trace_count']); selected=_stage_id(stage)
    arrays=[np.asarray(profile.get(key,np.zeros(n,dtype=np.int64)),dtype=np.int64)
            for key in ('trace_shard_id','trace_stage','trace_step','trace_call_id','trace_layer')]
    groups={}
    for i in range(n):
        if selected is not None and int(profile['trace_stage'][i])!=selected:continue
        key=tuple(int(a[i]) for a in arrays)
        groups.setdefault(key,[]).append(i)
    return [np.asarray(groups[key],dtype=np.int64) for key in sorted(groups)]


def _mapping(placement,topology,layer,source,expert,stage,world):
    nodes={int(k):int(v) for k,v in topology.get('rank_to_node',{}).items()}
    node=nodes.get(source,source)
    table=placement.get('dest_rank_table_'+stage,placement.get('dest_rank_table'))
    if table is not None:
        row=table.get(str(layer),{}).get(str(node))
        if row is None:raise ValueError(f'dispatch table missing layer {layer} source node {node}')
        return int(row[expert] if isinstance(row,list) else row[str(expert)])
    replicas=placement.get('replicas',{}).get(str(layer),{}).get(str(expert),[expert%world])
    if not replicas or any(int(r)<0 or int(r)>=world for r in replicas):
        raise ValueError('invalid expert replicas')
    matrix=placement.get('replica_choice_cost_matrix')
    def cost(rank):
        rank=int(rank)
        if matrix is not None:return float(matrix[source][rank])
        values=topology.get('cost',{})
        return float(values.get('same_rank',0)) if source==rank else float(values.get('same_node',1) if node==nodes.get(rank,rank) else values.get('cross_node',8))
    return int(min(replicas,key=lambda r:(cost(r),int(r)!=expert%world,int(r))))


def _invocation(profile,indices,placement,topology,dispatch_bytes,combine_bytes):
    world=int(profile['A_prefill'].shape[1]);nodes={int(k):int(v) for k,v in topology.get('rank_to_node',{}).items()}
    pair=np.zeros((world,world),dtype=np.float64);demand=[{} for _ in range(world)]
    stage='decode' if int(profile['trace_stage'][indices[0]]) else 'prefill'
    for i in indices:
        layer,source,expert,count=(int(profile[k][i]) for k in ('trace_layer','trace_source_rank','trace_expert','trace_count'))
        dest=_mapping(placement,topology,layer,source,expert,stage,world)
        if not 0<=dest<world:raise ValueError('dispatch table rank out of range')
        pair[source,dest]+=count;demand[dest][expert]=demand[dest].get(expert,0)+count
    bw=topology.get('bandwidth_model',{});comm=np.zeros(world);cross=np.zeros(world);events=0;items=0
    for s in range(world):
        for r in range(world):
            count=pair[s,r]
            if count<=0 or s==r:continue
            remote=nodes.get(s,s)!=nodes.get(r,r)
            bytes_=count*(dispatch_bytes+combine_bytes)
            bandwidth=float(bw.get('cross_node_bw_MBps',100) if remote else bw.get('same_node_bw_MBps',12000))
            if bandwidth<=0:raise ValueError('bandwidth must be positive')
            comm[s]+=bytes_/bandwidth/1000
            if remote:
                comm[s]+=max(0,float(bw.get('cross_node_latency_ms',0)))
                cross[s]+=bytes_;events+=1;items+=int(count)
    return comm,cross,events,items,demand,pair


def evaluate_grouped_comm_tail(profile,placement,topology,*,stage='prefill',dispatch_item_bytes=4106.,combine_item_bytes=4096.):
    world=profile['A_prefill'].shape[1];source=np.zeros(world);cross=np.zeros(world);tails=[];events=items=0
    groups=grouped_trace_indices(profile,stage)
    nodes={int(k):int(v) for k,v in topology.get('rank_to_node',{}).items()}
    direction_bytes={};direction_events={}
    for indices in groups:
        comm,volume,n_events,n_items,_,pair=_invocation(profile,indices,placement,topology,dispatch_item_bytes,combine_item_bytes)
        source+=comm;cross+=volume;events+=n_events;items+=n_items;tails.append(float(comm.max(initial=0)))
        for s,r in zip(*np.nonzero(pair)):
            if nodes.get(int(s),int(s))==nodes.get(int(r),int(r)):continue
            key=f'node{nodes.get(int(s),int(s))}->node{nodes.get(int(r),int(r))}'
            direction_bytes[key]=direction_bytes.get(key,0.)+float(pair[s,r])*(dispatch_item_bytes+combine_item_bytes)
            direction_events[key]=direction_events.get(key,0)+1
    return dict(J_comm_ms=float(sum(tails)),total_cross_node_bytes=float(cross.sum()),
        rankmax_inter_node_bytes=float(cross.max(initial=0)),active_cross_node_events=events,cross_node_items=items,
        source_comm_ms=source.tolist(),source_cross_node_bytes=cross.tolist(),groups=len(groups),
        per_source_total_comm_ms=source.tolist(),per_source_total_cross_node_bytes=cross.tolist(),
        rankmax_comm_ms_legacy=float(source.max(initial=0)),total_cross_node_items=items,
        per_direction_bytes=direction_bytes,per_direction_events=direction_events,
        communication_tail_p95_ms=float(np.percentile(tails,95)) if tails else 0.,
        semantics='sum of invocation maximum source dispatch/combine proxies; aggregate event latency once per active cross-node pair')


def _rank_values(topology,key,world,default):
    raw=topology.get(key,default)
    if isinstance(raw,dict):values=[float(raw.get(str(r),raw.get(r,default))) for r in range(world)]
    elif isinstance(raw,(list,tuple)):values=list(map(float,raw)) if raw else [float(default)]*world
    else:values=[float(raw)]*world
    if len(values)!=world or any(not np.isfinite(x) or x<=0 for x in values):raise ValueError('invalid '+key)
    return np.asarray(values)


def evaluate_profile_placement_with_cache_replay(profile,placement,topology,*,cache_capacity=8,
        cache_capacity_by_rank=None,cache_policy='lru',mode='no_rs',stage='prefill',
        dispatch_item_bytes=4106.,combine_item_bytes=4096.,compute_ms_per_item=None,
        warm_cache=False,warmup_placement=None,warmup_stage=None):
    if cache_policy!='lru' or mode!='no_rs':raise ValueError('replay supports reactive LRU only; RS requires actual prediction/admission traces')
    layers,world,_=profile['A_prefill'].shape
    capacities=[int(cache_capacity)]*world if cache_capacity_by_rank is None or len(cache_capacity_by_rank)==0 else list(map(int,cache_capacity_by_rank))
    if len(capacities)!=world or min(capacities)<0:raise ValueError('invalid cache capacities')
    h2d_scale=_rank_values(topology,'rank_h2d_time_scale',world,1.)
    compute_scale=_rank_values(topology,'rank_compute_time_scale',world,1.)
    bandwidth=_rank_values(topology,'h2d_bw_MBps_by_rank',world,12000.)
    expert_bytes=topology.get('expert_weight_bytes')
    calibrated=expert_bytes is not None and 'h2d_bw_MBps_by_rank' in topology
    caches=[[OrderedDict() for _ in range(world)] for _ in range(layers)]
    total_hit=total_items=miss_experts=0;h2d=np.zeros(world);h2d_bytes=np.zeros(world)
    comm_total=np.zeros(world);compute=np.zeros(world);objective=0.;cross_bytes=events=cross_items=0
    groups=grouped_trace_indices(profile,'mixed' if warm_cache else stage)
    selected=_stage_id(stage);warm_stage=_stage_id(warmup_stage or 'prefill');previous=None
    for indices in groups:
        first=int(indices[0]);shard=int(profile.get('trace_shard_id',np.zeros(len(profile['trace_count'])))[first])
        if shard!=previous:
            caches=[[OrderedDict() for _ in range(world)] for _ in range(layers)];previous=shard
        current=int(profile['trace_stage'][first]);measured=selected is None or current==selected
        if not measured and not (warm_cache and current==warm_stage):continue
        plan=placement if measured or warmup_placement is None else warmup_placement
        comm,cross,n_events,n_items,demand,_=_invocation(profile,indices,plan,topology,dispatch_item_bytes,combine_item_bytes)
        layer=int(profile['trace_layer'][first]);loads=np.zeros(world);work=np.zeros(world)
        for rank in range(world):
            cache=caches[layer][rank]
            for expert,count in sorted(demand[rank].items()):
                hit=expert in cache
                if hit:cache.move_to_end(expert)
                else:
                    if capacities[rank]>0:
                        cache[expert]=None
                        if len(cache)>capacities[rank]:cache.popitem(last=False)
                    if measured:
                        size=float(expert_bytes.get(str(layer),{}).get(str(expert),0)) if isinstance(expert_bytes,dict) else float(expert_bytes or 0)
                        if calibrated and size<=0:raise ValueError('missing expert payload size')
                        # Without measured payload/bandwidth, only relative miss-load
                        # units are available. Keep this explicit in the report.
                        loads[rank]+=(size/bandwidth[rank]/1000 if calibrated else 1.)*h2d_scale[rank]
                        h2d_bytes[rank]+=size;miss_experts+=1
                if measured:total_items+=count;total_hit+=int(hit)*count;work[rank]+=count
        if measured:
            work*=compute_scale*float(compute_ms_per_item if compute_ms_per_item is not None else .001)
            h2d+=loads;compute+=work;comm_total+=comm;cross_bytes+=float(cross.sum());events+=n_events;cross_items+=n_items
            objective+=float((comm+loads+work).max(initial=0))
    return dict(objective_ms=objective,rankmax_h2d_ms=float(h2d.max(initial=0)),h2d_demand_mb=float(h2d_bytes.sum()/1024**2),
        h2d_demand_bytes_by_rank=h2d_bytes.tolist(),h2d_time_by_rank=h2d.tolist(),
        resident_hit_rate_by_item=total_hit/max(1,total_items),resident_hit_items=total_hit,total_items=total_items,
        demand_expert_loads=miss_experts,rankmax_comm_ms=float(comm_total.max(initial=0)),
        rankmax_compute_items=float((compute/float(compute_ms_per_item or .001)).max(initial=0)),
        cross_node_items=cross_items,active_cross_node_pair_events=events,cross_node_bytes=cross_bytes,
        h2d_timing_calibrated=calibrated,
        h2d_time_semantics='payload bytes / measured per-rank bandwidth' if calibrated else 'unfitted unit cost per expert miss; numerical ms fields are proxies',
        limitations='aggregate expert-count traces omit per-token order; replay uses runtime expert-ID segment order and is not a serving latency measurement')
