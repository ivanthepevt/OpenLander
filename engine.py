"""OpenLander inference, fusion, temporal decisions, and immutable run storage."""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
VERSION = "1.0"
VIEW_DIRS = {"Final Decision":"annotated", "Raw Camera":"raw/frames", "Depth":"depth", "AI Perception":"perception", "Landing Score":"heatmap", "Optical Flow":"flow", "Debug Dashboard":"dashboard"}
SEMANTIC_PROMPTS = ["flat open ground","safe ground","rough terrain","rock","boulder","crater","building","vehicle","road","solar panel","habitat","landing pad"]
DEBUG_METADATA = {"crop_center_x_norm","crop_center_y_norm","crop_scale"}


def compute_device() -> tuple[str, str]:
    try:
        import torch
        if torch.backends.mps.is_available(): return "mps", "Apple MPS"
        if torch.cuda.is_available(): return "cuda", "CUDA"
    except Exception:
        pass
    return "cpu", "CPU"


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def validate_scenario(folder: str | Path):
    folder = Path(folder).expanduser().resolve()
    scenario_file, metadata_file, frames_dir = folder/"scenario.json", folder/"metadata.csv", folder/"frames"
    if not scenario_file.is_file() or not metadata_file.is_file() or not frames_dir.is_dir():
        raise ValueError("Raw scenario must contain scenario.json, metadata.csv, and frames/.")
    scenario = json.loads(scenario_file.read_text())
    metadata = pd.read_csv(metadata_file)
    required = {"frame_id","filename","timestamp_s"}
    missing = required - set(metadata.columns)
    if missing: raise ValueError(f"metadata.csv is missing: {', '.join(sorted(missing))}")
    if metadata.empty: raise ValueError("metadata.csv contains no frames.")
    metadata = metadata.sort_values("timestamp_s",kind="stable").reset_index(drop=True)
    paths = [frames_dir/str(x) for x in metadata.filename]
    absent = [p.name for p in paths if not p.is_file()]
    if absent: raise ValueError(f"Missing frame files: {', '.join(absent[:5])}")
    shapes=[]
    for p in paths:
        im=cv2.imread(str(p));
        if im is None: raise ValueError(f"Unreadable image: {p.name}")
        shapes.append(im.shape[:2])
    if len(set(shapes)) != 1: raise ValueError("Every frame must have identical dimensions.")
    if scenario.get("mission_mode","SAFETY") not in {"SAFETY","RECOVERY"}: raise ValueError("mission_mode must be SAFETY or RECOVERY.")
    return folder, scenario, metadata, paths, shapes[0]


def validate_run(folder: str | Path):
    folder=Path(folder).expanduser().resolve()
    if not (folder/"run.json").is_file() or not (folder/"result.json").is_file():
        raise ValueError("Processed run must contain run.json and result.json.")
    run=json.loads((folder/"run.json").read_text()); result=json.loads((folder/"result.json").read_text())
    if not result.get("frames"): raise ValueError("Processed run has no frame results.")
    return folder,run,result


def list_folders(kind: str) -> list[str]:
    base=ROOT/("scenarios" if kind=="scenario" else "runs")
    marker="scenario.json" if kind=="scenario" else "run.json"
    return [str(p) for p in sorted(base.iterdir(), key=lambda x:x.stat().st_mtime, reverse=True) if p.is_dir() and (p/marker).exists()] if base.exists() else []


def load_models(device: str, progress: Callable[[str],None] | None=None):
    def note(x):
        if progress: progress(x)
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, CLIPSegProcessor, CLIPSegForImageSegmentation
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Raw analysis needs the packages in requirements.txt. Processed-run replay does not.") from e
    note("Loading Depth Anything V2 Small…")
    depth_name="depth-anything/Depth-Anything-V2-Small-hf"
    depth_processor=AutoImageProcessor.from_pretrained(depth_name, cache_dir=str(ROOT/"models_cache"))
    depth_model=AutoModelForDepthEstimation.from_pretrained(depth_name, cache_dir=str(ROOT/"models_cache")).to(device).eval()
    note("Loading CLIPSeg zero-shot perception…")
    seg_name="CIDAS/clipseg-rd64-refined"
    seg_processor=CLIPSegProcessor.from_pretrained(seg_name, cache_dir=str(ROOT/"models_cache"))
    seg_model=CLIPSegForImageSegmentation.from_pretrained(seg_name, cache_dir=str(ROOT/"models_cache")).to(device).eval()
    note("Loading YOLO nano detector…")
    detector=YOLO("yolo11n.pt")
    return {"torch":torch,"depth_processor":depth_processor,"depth_model":depth_model,"seg_processor":seg_processor,"seg_model":seg_model,"detector":detector,"names":{"depth":depth_name,"segmentation":seg_name,"detection":"Ultralytics YOLO11n"}}


def normalize(a):
    a=np.nan_to_num(a.astype(np.float32)); lo,hi=np.percentile(a,(2,98))
    return np.clip((a-lo)/(hi-lo+1e-6),0,1).astype(np.float32)


def infer_ai(pil: Image.Image, models, device: str):
    torch=models["torch"]; w,h=pil.size
    with torch.inference_mode():
        inp=models["depth_processor"](images=pil,return_tensors="pt").to(device)
        pred=models["depth_model"](**inp).predicted_depth
        pred=torch.nn.functional.interpolate(pred.unsqueeze(1),size=(h,w),mode="bicubic",align_corners=False).squeeze().detach().float().cpu().numpy()
        depth=normalize(pred)
        sinp=models["seg_processor"](text=SEMANTIC_PROMPTS,images=[pil]*len(SEMANTIC_PROMPTS),padding=True,return_tensors="pt").to(device)
        logits=models["seg_model"](**sinp).logits
        probs=torch.sigmoid(torch.nn.functional.interpolate(logits.unsqueeze(1),size=(h,w),mode="bilinear",align_corners=False)).squeeze(1).detach().float().cpu().numpy()
    detections=[]
    try:
        result=models["detector"].predict(pil,device=device,verbose=False,conf=.2,imgsz=min(960,max(w,h)))[0]
        names=result.names
        for box,conf,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
            detections.append({"box":[round(float(v),1) for v in box],"confidence":round(float(conf),3),"label":names[int(cls)]})
    except Exception as e:
        # MPS kernels can occasionally be unsupported inside third-party post-processing.
        if device != "cpu":
            result=models["detector"].predict(pil,device="cpu",verbose=False,conf=.2,imgsz=min(960,max(w,h)))[0]
            for box,conf,cls in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                detections.append({"box":[round(float(v),1) for v in box],"confidence":round(float(conf),3),"label":result.names[int(cls)]})
        else: raise RuntimeError(f"YOLO inference failed: {e}") from e
    del inp,pred,sinp,logits
    if device=="mps": torch.mps.empty_cache()
    return depth,dict(zip(SEMANTIC_PROMPTS,probs)),detections


def estimate_motion(prev_gray, gray, dt):
    empty={"vehicle_dx_px_s":0.,"vehicle_dy_px_s":0.,"visual_drift_speed":0.,"camera_rotation_deg":0.,"frame_scale_change":1.,"tracks":[]}
    if prev_gray is None: return empty
    pts=cv2.goodFeaturesToTrack(prev_gray,maxCorners=220,qualityLevel=.015,minDistance=12)
    if pts is None: return empty
    nxt,status,_=cv2.calcOpticalFlowPyrLK(prev_gray,gray,pts,None,winSize=(25,25),maxLevel=3)
    good0,good1=pts[status.ravel()==1],nxt[status.ravel()==1]
    if len(good0)<5: return empty
    affine,inliers=cv2.estimateAffinePartial2D(good0,good1,method=cv2.RANSAC,ransacReprojThreshold=3)
    if affine is None: return empty
    dx,dy=float(affine[0,2])/max(dt,.001),float(affine[1,2])/max(dt,.001)
    scale=math.hypot(affine[0,0],affine[0,1]); rot=math.degrees(math.atan2(affine[1,0],affine[0,0]))
    indices=np.where(inliers.ravel()>0)[0][:55] if inliers is not None else range(min(55,len(good0)))
    tracks=[[*map(lambda v:round(float(v),1),good0[j,0]),*map(lambda v:round(float(v),1),good1[j,0])] for j in indices]
    return {"vehicle_dx_px_s":round(-dx,2),"vehicle_dy_px_s":round(-dy,2),"visual_drift_speed":round(math.hypot(dx,dy),2),"camera_rotation_deg":round(rot,3),"frame_scale_change":round(scale,4),"tracks":tracks}


def footprint_px(row, scenario, width):
    try:
        altitude=float(row.get("altitude_m")); hfov=float(row.get("camera_hfov_deg")); radius=float(scenario.get("vehicle_radius_m",2.5))*1.8
        ground_width=2*altitude*math.tan(math.radians(hfov)/2)
        return int(np.clip(radius/ground_width*width, width*.018, width*.12))
    except Exception: return int(width*.045)


def fuse_maps(depth, sem, detections, motion, row, scenario, previous_xy):
    h,w=depth.shape; radius=footprint_px(row,scenario,w)
    gx=cv2.Sobel(depth,cv2.CV_32F,1,0,ksize=5); gy=cv2.Sobel(depth,cv2.CV_32F,0,1,ksize=5)
    terrain=1-normalize(np.hypot(gx,gy)); terrain=cv2.GaussianBlur(terrain,(0,0),max(3,radius/3))
    safe=np.maximum.reduce([sem["flat open ground"],sem["safe ground"],sem["landing pad"]])
    semantic_hazard=np.maximum.reduce([sem[x] for x in ["rough terrain","rock","boulder","crater"]])
    infrastructure=np.maximum.reduce([sem[x] for x in ["building","vehicle","road","solar panel","habitat"]])
    hazard=np.clip(.60*semantic_hazard+.28*(1-terrain),0,1)
    for det in detections:
        x1,y1,x2,y2=map(int,det["box"]); cv2.rectangle(hazard,(x1,y1),(x2,y2),1.,-1)
    k=max(3,radius*2+1); k += 1-k%2
    expanded=cv2.dilate((hazard*255).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))).astype(np.float32)/255
    clearance=1-cv2.GaussianBlur(expanded,(0,0),max(2,radius/2))
    # Reachable ellipse shifts opposite visual drift and follows telemetry if present.
    vx=float(row.get("velocity_x_mps",0) if pd.notna(row.get("velocity_x_mps",0)) else 0); vy=float(row.get("velocity_y_mps",0) if pd.notna(row.get("velocity_y_mps",0)) else 0)
    shift_x=np.clip(-motion["vehicle_dx_px_s"]*.65+vx*w*.012,-w*.2,w*.2); shift_y=np.clip(-motion["vehicle_dy_px_s"]*.65+vy*h*.012,-h*.2,h*.2)
    cx,cy=w/2+shift_x,h/2+shift_y; yy,xx=np.mgrid[:h,:w]
    alt=row.get("altitude_m",np.nan); shrink=.70 if pd.isna(alt) else np.clip(.25+float(alt)/220,.3,.78)
    dist=((xx-cx)/(w*shrink))**2+((yy-cy)/(h*shrink*.8))**2
    reachable=np.clip(1.35-dist,0,1).astype(np.float32)
    # Recovery prefers a safe annulus around infrastructure; SAFETY ignores it.
    inf8=(infrastructure*255).astype(np.uint8); distance=cv2.distanceTransform((inf8<100).astype(np.uint8),cv2.DIST_L2,5)
    annulus=np.exp(-((distance-w*.16)/(w*.13+1))**2).astype(np.float32)
    facility=annulus*(1-infrastructure) if scenario.get("mission_mode")=="RECOVERY" else np.ones_like(depth)*.5
    stability=np.ones_like(depth)*.55
    if previous_xy:
        stability=.35+.65*np.exp(-(((xx-previous_xy[0])/(w*.22))**2+((yy-previous_xy[1])/(h*.22))**2)).astype(np.float32)
    score=.23*terrain+.21*safe+.22*clearance+.20*reachable+.08*facility+.06*stability
    score*=np.clip(1-.72*expanded,0,1); score*=.22+.78*reachable
    margin=max(radius,8); score[:margin]=score[-margin:]=0; score[:,:margin]=score[:,-margin:]=0
    return {"score":np.clip(score,0,1),"terrain":terrain,"clearance":clearance,"safe":safe,"reachable":reachable,"facility":facility,"hazard":expanded,"stability":stability},radius,(cx,cy,w*shrink,h*shrink*.8)


def candidates(maps, count=3):
    work=cv2.GaussianBlur(maps["score"],(0,0),10); h,w=work.shape; out=[]
    sep=int(min(h,w)*.18)
    for i in range(count):
        _,value,_,loc=cv2.minMaxLoc(work); x,y=loc
        vals={k:float(v[y,x]) for k,v in maps.items() if k not in {"hazard","score"}}
        out.append({"id":f"LZ-{i+1:02d}","xy":[x,y],"total_score":round(float(value),3),"terrain_score":round(vals["terrain"],3),"clearance_score":round(vals["clearance"],3),"semantic_safety":round(vals["safe"],3),"reachability":round(vals["reachable"],3),"facility_preference":round(vals["facility"],3),"temporal_stability":round(vals["stability"],3)})
        cv2.circle(work,(x,y),sep,0,-1)
    return out


def decide(cands, previous, stable_count, index, total):
    best=cands[0]; retarget=False
    if previous:
        near=min(cands,key=lambda c:math.dist(c["xy"],previous["xy"]))
        if math.dist(near["xy"],previous["xy"]) < 180 and near["reachability"]>.25 and near["total_score"]>best["total_score"]-.09: best=near
        retarget=math.dist(best["xy"],previous["xy"])>=150 or best["reachability"]<.25
        # Candidate IDs identify tracked zones, not their instantaneous score rank.
        # Preserve the selected ID while tracking; assign a different ID on retarget.
        if not retarget and best["id"] != previous["id"]:
            displaced=next(c for c in cands if c["id"]==previous["id"])
            displaced["id"],best["id"]=best["id"],previous["id"]
        elif retarget and best["id"] == previous["id"]:
            replacement=next(c for c in cands if c["id"]!=previous["id"])
            replacement["id"],best["id"]=best["id"],replacement["id"]
    if index==0: state="SEARCHING"; stable_count=0
    elif retarget: state="RETARGETING"; stable_count=0
    elif index==1: state="CANDIDATE_FOUND"; stable_count=1
    else:
        stable_count+=1
        state="TRACKING" if stable_count<3 else "TARGET_LOCKED"
    if index>=total-2: state="FINAL_APPROACH"
    why=(f"{best['id']} selected for its {best['terrain_score']:.0%} terrain quality, {best['clearance_score']:.0%} clearance, and {best['reachability']:.0%} reachability.")
    if retarget: why=f"The previous target lost relative reachability or score. OpenLander retargeted to {best['id']}."
    return best,state,stable_count,retarget,why


def color_map(a, cmap=cv2.COLORMAP_TURBO): return cv2.applyColorMap((normalize(a)*255).astype(np.uint8),cmap)


def render_views(rgb, depth, sem, maps, motion, cands, selected, state, radius, envelope, scenario):
    h,w=rgb.shape[:2]; final=rgb.copy(); overlay=final.copy()
    overlay[maps["hazard"]>.5]=(30,35,220); cv2.addWeighted(overlay,.32,final,.68,0,final)
    cx,cy,rx,ry=envelope; cv2.ellipse(final,(int(cx),int(cy)),(int(rx),int(ry)),0,0,360,(80,210,255),3)
    for c in cands:
        x,y=c["xy"]; chosen=c is selected
        color=(64,245,120) if chosen else (255,190,55); thick=5 if chosen else 2
        cv2.circle(final,(x,y),radius,color,thick); cv2.circle(final,(x,y),7,color,-1)
        cv2.putText(final,f"{c['id']} {c['total_score']:.2f}",(x+12,y-12),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    sx,sy=selected["xy"]; center=(w//2,h//2)
    pts=np.array([center,((2*center[0]+sx)//3,(2*center[1]+sy)//3),((center[0]+2*sx)//3,(center[1]+2*sy)//3),(sx,sy)],np.int32)
    cv2.polylines(final,[pts],False,(235,235,255),2,cv2.LINE_AA)
    cv2.arrowedLine(final,center,(int(center[0]+motion["vehicle_dx_px_s"]*.35),int(center[1]+motion["vehicle_dy_px_s"]*.35)),(0,180,255),3,tipLength=.25)
    cv2.rectangle(final,(0,0),(w,62),(9,16,25),-1); cv2.putText(final,f"OPENLANDER  //  {state}",(22,41),cv2.FONT_HERSHEY_SIMPLEX,.8,(235,245,250),2,cv2.LINE_AA)
    perception=color_map(np.maximum.reduce([sem["safe ground"],sem["flat open ground"]])-np.maximum.reduce([sem["rock"],sem["crater"],sem["building"]]),cv2.COLORMAP_VIRIDIS)
    flow=rgb.copy()
    for x0,y0,x1,y1 in motion["tracks"]: cv2.arrowedLine(flow,(int(x0),int(y0)),(int(x1),int(y1)),(40,220,255),2,tipLength=.25)
    heat=color_map(maps["score"]); heat=cv2.addWeighted(rgb,.35,heat,.65,0)
    depth_view=color_map(depth,cv2.COLORMAP_MAGMA)
    dash=np.zeros((h,w,3),np.uint8); thumbs=[rgb,depth_view,perception,heat]
    tw,th=w//2,h//2
    for i,img in enumerate(thumbs): dash[(i//2)*th:(i//2+1)*th,(i%2)*tw:(i%2+1)*tw]=cv2.resize(img,(tw,th))
    return {"annotated":final,"depth":depth_view,"perception":perception,"heatmap":heat,"flow":flow,"dashboard":dash}


def analyze_scenario(
    folder: str | Path,
    progress: Callable[[str],None] | None=None,
    *,
    models=None,
    device: str | None=None,
    evaluation_set: str | None=None,
) -> str:
    """Analyze one scenario, optionally reusing already-loaded batch models."""
    source,scenario,metadata,paths,(h,w)=validate_scenario(folder)
    def note(x):
        if progress: progress(x)
    stamp=datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")[:-3]
    run_name=f"{scenario.get('scenario_name',source.name)}_{stamp}"
    final_dir=ROOT/"runs"/run_name
    run_dir=ROOT/"runs"/f".{run_name}.partial"
    run_dir.mkdir(parents=True)
    shutil.copytree(source,run_dir/"raw")
    for d in ["annotated","depth","perception","heatmap","flow","dashboard","maps"]: (run_dir/d).mkdir()
    detected_device,device_label=compute_device()
    device=device or detected_device
    device_label={"mps":"Apple MPS","cuda":"CUDA","cpu":"CPU"}.get(device,device)
    owns_models=models is None
    started=time.perf_counter()
    if owns_models: models=load_models(device,note)
    records=[]; events=[]; prev_gray=None; prev_selected=None; stable=0; retargets=0; frames_for_gif=[]
    try:
        for i,(row,path) in enumerate(zip(metadata.to_dict("records"),paths)):
            tic=time.perf_counter(); note(f"Frame {i+1} / {len(paths)} — Depth + perception…")
            pil=Image.open(path).convert("RGB"); rgb=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR); gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            depth,sem,detections=infer_ai(pil,models,device)
            dt=1 if i==0 else float(row["timestamp_s"])-float(metadata.iloc[i-1].timestamp_s)
            motion=estimate_motion(prev_gray,gray,dt); note(f"Frame {i+1} / {len(paths)} — Fusion + decision…")
            maps,radius,envelope=fuse_maps(depth,sem,detections,motion,row,scenario,prev_selected["xy"] if prev_selected else None)
            cands=candidates(maps); selected,state,stable,retarget,why=decide(cands,prev_selected,stable,i,len(paths))
            if retarget: retargets+=1
            if not records or records[-1]["state"]!=state: events.append({"timestamp_s":float(row["timestamp_s"]),"event":state+(f" → {selected['id']}" if selected else "")})
            views=render_views(rgb,depth,sem,maps,motion,cands,selected,state,radius,envelope,scenario)
            filename=f"{i:06d}.png"
            for d,img in views.items(): cv2.imwrite(str(run_dir/d/filename),img)
            # Quarter-scale arrays preserve local replay inspection while keeping
            # dense 150-frame runs compact.
            map_w=max(1,min(320,w)); map_h=max(1,round(h*map_w/w))
            compact_maps={k:cv2.resize(v,(map_w,map_h),interpolation=cv2.INTER_AREA).astype(np.float16) for k,v in maps.items()}
            np.savez_compressed(run_dir/"maps"/f"{i:06d}.npz",**compact_maps)
            control={"label":"Simulated control reference","path":[[w//2,h//2],[(2*(w//2)+selected['xy'][0])//3,(2*(h//2)+selected['xy'][1])//3],selected["xy"]]}
            telemetry={k:(None if pd.isna(v) else v) for k,v in row.items() if k not in ({"frame_id","filename","timestamp_s"}|DEBUG_METADATA)}
            rec={"frame_id":int(row["frame_id"]),"filename":filename,"timestamp_s":float(row["timestamp_s"]),"state":state,"selected_candidate":selected["id"],"selected_xy":selected["xy"],"score":selected["total_score"],"footprint_radius_px":radius,"reachable_envelope":[round(float(x),2) for x in envelope],"drift":{k:v for k,v in motion.items() if k!="tracks"},"telemetry":telemetry,"detections":detections,"candidates":cands,"explanation":why,"control_reference":control,"analysis_time_ms":round((time.perf_counter()-tic)*1000,1)}
            records.append(rec); frames_for_gif.append(cv2.cvtColor(cv2.resize(views["annotated"],(640,640)),cv2.COLOR_BGR2RGB)); prev_gray=gray; prev_selected=selected
            note(f"Frame {i+1} / {len(paths)} — saved")
    finally:
        if owns_models: del models
        gc.collect()
        if device=="mps":
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
    elapsed=time.perf_counter()-started
    locked=next((r["timestamp_s"] for r in records if r["state"]=="TARGET_LOCKED"),None)
    result={"scenario":scenario,"events":events,"frames":records,"mission_summary":{"result":"SAFE LANDING TARGET IDENTIFIED","final_landing_zone":records[-1]["selected_candidate"],"final_safety_score":records[-1]["score"],"target_locked_at_s":locked,"retarget_events":retargets,"average_analysis_time_ms":round(sum(r["analysis_time_ms"] for r in records)/len(records),1),"total_analysis_time_s":round(elapsed,2),"final_drift":records[-1]["drift"]}}
    run={"openlander_version":VERSION,"scenario_name":scenario.get("scenario_name",source.name),"title":scenario.get("title",source.name),"created_at":datetime.now().isoformat(timespec="seconds"),"device":device,"device_label":device_label,"frame_count":len(records),"image_width":w,"image_height":h,"models":{"depth":"depth-anything/Depth-Anything-V2-Small-hf","detection":"Ultralytics YOLO11n","segmentation":"CIDAS/clipseg-rd64-refined"},"analysis_time_s":round(elapsed,2)}
    if evaluation_set: run["evaluation_set"]=evaluation_set
    (run_dir/"run.json").write_text(json.dumps(run,indent=2,default=_json_default)+"\n")
    (run_dir/"result.json").write_text(json.dumps(result,indent=2,default=_json_default)+"\n")
    summary=result["mission_summary"]
    (run_dir/"summary.txt").write_text("OPENLANDER MISSION COMPLETE\n\n"+"\n".join(f"{k.replace('_',' ').title()}: {v}" for k,v in summary.items())+"\n")
    timestamps=[r["timestamp_s"] for r in records]
    frame_duration=(timestamps[-1]-timestamps[0])/(len(timestamps)-1) if len(timestamps)>1 else 1/30
    imageio.mimsave(run_dir/"landing.gif",frames_for_gif,duration=max(.02,frame_duration),loop=0)
    run_dir.rename(final_dir)
    note(f"Mission saved: {final_dir}")
    return str(final_dir)


def frame_view(run_dir: str | Path, frame_index: int, view: str) -> str:
    folder,run,result=validate_run(run_dir); frame=result["frames"][max(0,min(int(frame_index),len(result["frames"])-1))]
    sub=VIEW_DIRS.get(view,"annotated")
    return str(folder/sub/frame["filename"])


def inspect_point(run_dir: str | Path, frame_index: int, x: float, y: float) -> str:
    folder,run,result=validate_run(run_dir); frame=result["frames"][int(frame_index)]
    maps=np.load(folder/"maps"/f"{int(frame_index):06d}.npz"); h,w=maps["score"].shape
    display_w=float(run.get("image_width",w)); display_h=float(run.get("image_height",h))
    display_x,display_y=int(x),int(y)
    x=int(np.clip(x/display_w*w,0,w-1)); y=int(np.clip(y/display_h*h,0,h-1)); vals={k:float(maps[k][y,x]) for k in maps.files}
    if vals["reachable"]<.25: reason="Outside the current reachable envelope."
    elif vals["hazard"]>.45: reason="A perceived hazard overlaps the landing footprint."
    elif vals["clearance"]<.5: reason="Insufficient object and terrain clearance."
    else: reason=f"Valid area, but {frame['selected_candidate']} has higher fused reachability and clearance."
    verdict="REJECTED" if vals["score"]<.45 or vals["reachable"]<.25 or vals["hazard"]>.45 else "SAFE BUT LOWER SCORE"
    return f"### WHY NOT LAND HERE?\n\n**{verdict}** at ({display_x}, {display_y})\n\nTerrain `{vals['terrain']:.0%}` · Clearance `{vals['clearance']:.0%}` · Reachability `{vals['reachable']:.0%}` · Facility preference `{vals['facility']:.0%}`\n\nPrimary reason: {reason}"


def export_video(run_dir: str | Path, view: str="Final Decision", fps: int=30) -> str:
    """Export a presentation replay from cached frames; never loads AI models."""
    folder,_,result=validate_run(run_dir)
    subdir={"Final Decision":"annotated","Debug Dashboard":"dashboard"}.get(view)
    if not subdir: raise ValueError("Export view must be Final Decision or Debug Dashboard.")
    source=folder/subdir
    if not source.is_dir() or len(list(source.glob("*.png"))) < len(result["frames"]):
        raise ValueError(f"Cached {view} frames are incomplete.")
    stem="landing_replay" if view=="Final Decision" else "dashboard_replay"
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg:
        output=folder/f"{stem}.mp4"
        command=[ffmpeg,"-y","-loglevel","error","-framerate",str(fps),"-i",str(source/"%06d.png"),
                 "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-preset","medium","-crf","18",
                 "-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
        completed=subprocess.run(command,capture_output=True,text=True)
        if completed.returncode==0 and output.is_file(): return str(output)
    output=folder/f"{stem}.gif"
    images=(imageio.imread(source/f["filename"]) for f in result["frames"])
    imageio.mimsave(output,images,duration=1/max(1,fps),loop=0)
    return str(output)
