"""Replay real callbacks without DDS; record per-candidate decisions for labeling."""
import argparse
import importlib.util
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from ar_perception.lidar_perception import LidarPerception
from ar_perception.wall_filter import WallFilter, WALL_DEFAULTS
from evaluate_accumulation import offline_node, defaults, Sink


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('original_lidar',args.baseline)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    params=defaults(root/'ar_perception/lidar_perception.py',root/'config/lidar.yaml')
    params.update(WALL_DEFAULTS)
    nodes=[offline_node(mod.LidarPerception,dict(params)),
           offline_node(LidarPerception,dict(params,use_wall_filter=False)),
           offline_node(LidarPerception,dict(params,use_wall_filter=True))]
    for n in nodes:
        n.wall_filter=WallFilter();n.pub_wall_filter_debug=Sink()
    db=sqlite3.connect(f'file:{args.bag}?mode=ro',uri=True)
    topics={i:(name,get_message(typ)) for i,name,typ in db.execute('select id,name,type from topics') if name in ['/scan','/odom']}
    start=db.execute('select min(timestamp) from messages').fetchone()[0]
    rows=[];timings=[[],[],[]];same=True;other_same=True;scans=0;differences=[]
    for tid,t,raw in db.execute('select topic_id,timestamp,data from messages where topic_id in (%s) order by timestamp,id'%','.join(map(str,topics))):
        topic,typ=topics[tid];msg=deserialize_message(raw,typ)
        if topic=='/odom':
            for n in nodes:n.odom_callback(msg)
            continue
        scans+=1
        for i,n in enumerate(nodes):
            before=time.perf_counter();n.on_scan(msg);timings[i].append((time.perf_counter()-before)*1000)
        for key in ['obstacles','event','walls','parking_gap','debug','clusters']:
            a,b,c=[getattr(n,'pub_'+key).msg for n in nodes]
            if a!=b:
                same=False;differences.append([scans,key])
            if key!='obstacles':other_same &= b==c
        diag=json.loads(nodes[2].pub_wall_filter_debug.msg.data)
        # Current-only ablation uses exactly the same detected geometry.
        current=WallFilter();current.walls=[(w,s) for w,s in nodes[2].wall_filter.walls if s=='scan']
        pts=nodes[1].remove_self_points(nodes[1].to_base_link(msg))
        candidates=[]
        for cluster in nodes[1].euclidean_clustering(nodes[1].apply_roi(pts)):
            circle=nodes[1].fit_circle(cluster)
            limo=None if nodes[1].is_cone_candidate(cluster,circle) else nodes[1].estimate_limo_candidate(cluster,circle)
            if limo is not None:candidates.append((cluster,limo))
        assert len(candidates)==len(diag['candidates'])
        for j,(d,(cluster,limo)) in enumerate(zip(diag['candidates'],candidates)):
            d.update(t=(t-start)/1e9,scan=scans,index=j,scan_only_rejected=current.reject(cluster)[0],
                     surface=np.mean(cluster,axis=0).tolist(),points=np.asarray(cluster).tolist())
            rows.append(d)
    windows=[(4,6),(18,20)] if 'lap15' in args.bag else [(132,136),(145,149)]
    summary={'bag':args.bag,'scans':scans,'disabled_identical':bool(same),'disabled_differences':differences[:20],
             'other_outputs_identical':bool(other_same),'candidate_count':len(rows),
             'removed':sum(r['rejected'] for r in rows),
             'callback_ms':[dict(median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(max(v))) for v in timings],
             'windows':[]}
    for a,b in windows:
        rr=[r for r in rows if a<=r['t']<b]
        summary['windows'].append(dict(start=a,end=b,baseline=len(rr),scan_removed=sum(r['scan_only_rejected'] for r in rr),
            tracked_removed=sum(r['rejected'] for r in rr),kept=sum(not r['rejected'] for r in rr)))
    Path(args.output).write_text(json.dumps(dict(summary=summary,candidates=rows),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
