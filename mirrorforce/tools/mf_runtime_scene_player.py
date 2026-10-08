#!/usr/bin/env python3
"""Finite stdio public-policy endpoint for independent scene validation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--config-sha256', required=True)
    args = p.parse_args()
    from mirrorforce.agent.train.checkpoint_store import file_sha256
    if file_sha256(args.config) != args.config_sha256:
        raise ValueError('policy configuration changed')
    cfg = json.loads(args.config.read_bytes())
    if set(cfg) != {'assets', 'candidate', 'recipe'}:
        raise ValueError('no scene setup is accepted by the policy process')
    assets = cfg['assets']
    for item in assets.values():
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('policy asset changed')
    spec = importlib.util.spec_from_file_location('duel_native', assets['native']['path'])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if line.split()[-1].endswith('/libmfcore.so')}
    if mapped != {str(Path(assets['core']['path']).resolve())}:
        raise ValueError('wrong native core mapped')
    native.init_module(assets['cards_db']['path'], assets['code_list']['path'], {})
    from mirrorforce.agent.env.announce_law import register
    register(native, Path(assets['announce']['path']), 192)
    from mirrorforce.agent.train.policy_io import Policy, load_policy
    from mirrorforce.agent.train.sky_onpolicy import PublicPlayer, validate_request
    agent, variables, receipt = load_policy(assets['checkpoint']['path'], 'iterate',
        semantic_file=assets['semantic']['path'], code_list_file=assets['code_list']['path'],
        card_tables_file=assets['card_tables']['path'], native=native)
    if cfg['candidate'] is not None:
        import flax.serialization
        import jax
        item = cfg['candidate']
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('candidate changed')
        branch = flax.serialization.msgpack_restore(Path(item['path']).read_bytes())
        if branch['parent_checkpoint_sha256'] != assets['checkpoint']['sha256']:
            raise ValueError('candidate belongs to another parent')
        old, new = variables['params'], branch['params']
        if jax.tree.structure(old) != jax.tree.structure(new) or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in
                zip(jax.tree.leaves(old), jax.tree.leaves(new))):
            raise ValueError('candidate architecture changed')
        variables['params'] = new
    policy = Policy(agent, variables, receipt)
    player = None
    print('SCENE-PLAYER-READY '+json.dumps({
        'parent_checkpoint_sha256':assets['checkpoint']['sha256'],
        'actor_checkpoint_sha256':assets['checkpoint']['sha256'] if cfg['candidate'] is None else cfg['candidate']['sha256']}),flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request == {'reset':True}:
            player = PublicPlayer(native, policy, cfg['recipe'])
            print(json.dumps({'reset':True}), flush=True)
        else:
            if player is None:
                raise ValueError('reset the whole-game public memory first')
            print(json.dumps(player.answer(validate_request(request))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
