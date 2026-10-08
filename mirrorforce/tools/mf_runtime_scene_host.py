#!/usr/bin/env python3
"""Run predeclared heldout scenes against a separate public-only endpoint."""
import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    from mirrorforce.agent.train.checkpoint_store import file_sha256
    if file_sha256(args.plan)!=args.plan_sha256:
        raise ValueError('scene plan changed')
    plan=json.loads(args.plan.read_bytes())
    for item in plan['host_assets'].values():
        if file_sha256(item['path'])!=item['sha256']:
            raise ValueError('host asset changed')
    if args.out.exists():raise ValueError('preserve old validation attempts')
    args.out.mkdir(parents=True)
    from mirrorforce.effectinfo import get_effectinfo_core
    from mirrorforce.worldmodel.engine import DeckList
    from mirrorforce.agent.train.sky_scene_host import run_scene
    recipe=plan['recipe'];deck=DeckList('independent-scene-validation',tuple(recipe['main']),tuple(recipe['extra']))
    results=[]
    with (args.out/'player.stderr').open('xb') as err, (args.out/'protocol.jsonl').open('x') as log:
        proc=subprocess.Popen(plan['player_argv'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        buffer=b'';deadline=time.monotonic()+600
        def read_line():
            nonlocal buffer
            while b'\n' not in buffer:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('original scene/RPC budget exhausted')
                if not selector.select(min(remaining,10)):continue
                chunk=os.read(proc.stdout.fileno(),1<<20)
                if not chunk:raise RuntimeError('public player exited before response')
                buffer+=chunk
            line,buffer=buffer.split(b'\n',1)
            log.write(json.dumps({'receive':line.decode()})+'\n');log.flush()
            return line.decode()
        def request(row):
            log.write(json.dumps({'send':row})+'\n');log.flush()
            proc.stdin.write((json.dumps(row)+'\n').encode());proc.stdin.flush()
            return json.loads(read_line())
        try:
            readback=None
            while True:
                line=read_line()
                if line.startswith('CONFIG-READBACK '):readback=json.loads(line[len('CONFIG-READBACK '):])
                if line.startswith('SCENE-PLAYER-READY '):
                    ready=json.loads(line[len('SCENE-PLAYER-READY '):]);break
            if (readback is None or readback.get('job_sha256')!=plan['player_job_sha256'] or
                    ready!={k:plan[k] for k in ('parent_checkpoint_sha256','actor_checkpoint_sha256')}):
                raise ValueError('autonomous player did not load the registered actor and job')
            print('AUTONOMOUS-PLAYER-READY',flush=True)
            for i,scene in enumerate(plan['scenes']):
                if scene.get('split')!='heldout':raise ValueError('this pilot admits only predeclared heldout scenes')
                deadline=time.monotonic()+900
                if request({'reset':True})!={'reset':True}:raise ValueError('memory reset failed')
                result=run_scene(get_effectinfo_core(),deck,scene,request)
                result.update(actor_checkpoint_sha256=ready['actor_checkpoint_sha256'],
                              parent_checkpoint_sha256=ready['parent_checkpoint_sha256'])
                with (args.out/f'scene-{i:02d}.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2)
                results.append(result)
                print('AUTONOMOUS-SCENE '+json.dumps({k:result[k] for k in
                    ('scene','success','winner','lp','own_prompts','main2_set','finisher_selected')}),flush=True)
            proc.stdin.close()
            code=proc.wait(timeout=30)
            if code!=0:raise RuntimeError('public player exit was not successful')
        finally:
            selector.close()
            if proc.poll() is None:
                proc.stdin.close();proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
        with (args.out/'result.json').open('x') as f:
            json.dump({'complete':True,'player_actual_exit_code':code,'plan_sha256':args.plan_sha256,
                'teacher_forcing':False,'scene_successes':sum(r['success'] for r in results),
                'scenes':len(results),'playing_strength_evidence':False},f,sort_keys=True,indent=2)
    return 0


if __name__=='__main__':raise SystemExit(main())
