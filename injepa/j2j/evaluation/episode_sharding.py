"""Transport-only partitions of the canonical ledger and strict union reduction."""
from __future__ import annotations
import copy
import hashlib
import json
from .metrics import aggregate_episode_metrics, aggregate_episode_latency


def episode_shard_indices(episode_count, *, shard_index=0, shard_count=1):
    if any(type(v) is not int for v in (episode_count, shard_index, shard_count)):
        raise TypeError('partition controls must be integers')
    if not 1 <= shard_count <= episode_count or not 0 <= shard_index < shard_count:
        raise ValueError('invalid episode partition')
    return list(range(shard_index, episode_count, shard_count))


def episode_seed(seed, ledger_row_index):
    return (int(seed) + int(ledger_row_index)) % (2**32)


def merge_episode_shards(shards, *, ledger_episode_keys, episode_ledger_sha256):
    shards = list(shards)
    if not shards:
        raise ValueError('no shards')
    first = shards[0]
    required = ('schema', 'policy', 'checkpoint', 'resolved_config_sha256', 'decision', 'analysis_manifest', 'episode_ledger_sha256')
    if any(not first.get(k) for k in required):
        raise ValueError('missing scientific identity')
    if first['schema'] != 'J2J_CONTEXT4_CLOSED_LOOP_EVALUATION_V1' or first['policy'] != 'context4':
        raise ValueError('not a Context4 scientific receipt')
    runtime = first.get('adapter_provenance', {}).get('runtime_identity')
    if not runtime:
        raise ValueError('missing adapter runtime identity')
    count = first.get('shard_count')
    if type(count) is not int or count < 2 or len(shards) != count:
        raise ValueError('incomplete shard set')
    identities = ('schema', 'policy', 'checkpoint', 'resolved_config_sha256', 'decision',
                  'analysis_manifest', 'episode_ledger_sha256')
    if first.get('episode_ledger_sha256') != episode_ledger_sha256:
        raise ValueError('ledger identity drift')
    ranks, rows = set(), {}
    for shard in shards:
        if shard.get('adapter_provenance', {}).get('runtime_identity') != runtime:
            raise ValueError('adapter runtime identity drift')
        rank = shard.get('shard_index')
        if type(rank) is not int or rank in ranks or shard.get('shard_count') != count:
            raise ValueError('duplicate or inconsistent shard rank')
        expected = episode_shard_indices(len(ledger_episode_keys), shard_index=rank, shard_count=count)
        ranks.add(rank)
        if shard.get('status') != 'PARTIAL' or any(shard.get(k) != first.get(k) for k in identities):
            raise ValueError('shard scientific identity or status drift')
        part = shard.get('episodes', [])
        if [r.get('ledger_row_index') for r in part] != expected or shard.get('episode_count') != len(part):
            raise ValueError('shard rows do not cover assigned indices')
        if shard.get('episode_keys') != [ledger_episode_keys[i] for i in expected]:
            raise ValueError('shard episode keys drift')
        for row in part:
            i = row['ledger_row_index']
            if i in rows or row.get('episode_key') != ledger_episode_keys[i] or row.get('episode_ledger_sha256') != episode_ledger_sha256:
                raise ValueError('duplicate or incorrectly bound episode row')
            rows[i] = copy.deepcopy(row)
    if sorted(rows) != list(range(len(ledger_episode_keys))):
        raise ValueError('incomplete canonical ledger coverage')
    result = copy.deepcopy(first)
    result.pop('shard_index', None)
    result['status'] = 'PASS'
    result['episodes'] = [rows[i] for i in range(len(rows))]
    result['episode_count'] = len(rows)
    result['episode_keys'] = copy.deepcopy(ledger_episode_keys)
    result['episode_key_order_sha256'] = hashlib.sha256(json.dumps(ledger_episode_keys, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
    result['metrics'] = aggregate_episode_metrics(result['episodes'])
    result['latency'] = aggregate_episode_latency(result['episodes'])
    return result
