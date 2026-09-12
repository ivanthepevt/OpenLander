"""Polished local Gradio mission-control UI for OpenLander."""
from __future__ import annotations

import html
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except ImportError:
    pass

import gradio as gr

import engine

ROOT=Path(__file__).resolve().parent
VIEWS=list(engine.VIEW_DIRS)

CSS="""
:root{--ol-bg:#071018;--ol-panel:#0d1923;--ol-line:#233746;--ol-cyan:#55d9ff;--ol-green:#63f2a6;--ol-text:#eaf5fb;--ol-muted:#8da7b8}
.gradio-container{background:radial-gradient(circle at 50% -20%,#173449,#071018 48%)!important;color:var(--ol-text)!important;max-width:1600px!important}
.gradio-container .form,.gradio-container .block:not(.image-container){background:#0d1923!important;border-color:#233746!important}
.gradio-container input,.gradio-container textarea{background:#111f2b!important;color:#eaf5fb!important;border-color:#2b4555!important}
.gradio-container label span,.gradio-container .block-info{color:#9db4c3!important}
.ol-hero{padding:24px 28px;border:1px solid #254050;background:linear-gradient(125deg,rgba(15,37,51,.98),rgba(7,16,24,.9));border-radius:14px;margin-bottom:12px}
.ol-kicker{font:600 11px ui-monospace;letter-spacing:.24em;color:var(--ol-cyan)}
.ol-title{font:700 32px system-ui;margin:6px 0;color:#f1faff!important}.ol-sub,.ol-sub b{color:#a9becb!important;max-width:850px}
.ol-panel{border:1px solid var(--ol-line)!important;background:rgba(13,25,35,.9)!important;border-radius:12px!important;color:#dcecf4!important}.ol-panel h3,.ol-panel p{color:#dcecf4!important}
.ol-readout{font-family:ui-monospace,monospace;color:var(--ol-text)}
.ol-readout>div:not(.ol-label){color:#eaf5fb!important}
.ol-label{font-size:10px;letter-spacing:.17em;color:#7f9bad;margin-top:12px}.ol-value{font-size:18px;color:#eefaff}.ol-accent{color:var(--ol-green)}
.candidate{padding:10px 12px;margin:7px 0;border:1px solid #243b4a;border-radius:8px;background:#0a151e;color:#dcecf4!important}.candidate.selected{border-color:#48da9a;box-shadow:inset 3px 0 #48da9a}
.bar{height:5px;background:#1b2c37;border-radius:9px;overflow:hidden;margin:4px 0 7px}.bar i{display:block;height:100%;background:linear-gradient(90deg,#37abd1,#59e1aa)}
.event{color:#9eb7c7;border-left:2px solid #284858;padding:3px 9px;margin:2px}.event.current{color:#fff;border-color:#58dca2;background:#10231f}
footer{display:none!important}
"""


def escape(x): return html.escape(str(x if x is not None else "—"))


def load_data(run_path):
    folder,run,result=engine.validate_run(run_path)
    return str(folder),run,result


def decision_html(run,result,i):
    f=result["frames"][i]; s=result["scenario"]; t=f["telemetry"]; d=f["drift"]
    velocity=" / ".join(str(t.get(x,"—")) for x in ("velocity_x_mps","velocity_y_mps","velocity_z_mps"))
    return f'''<div class="ol-readout">
    <div class="ol-label">SCENARIO · {escape(s.get('mission_mode','SAFETY'))}</div><div class="ol-value">{escape(s.get('title',s.get('scenario_name')))}</div>
    <div class="ol-label">COMPUTE DEVICE</div><div>{escape(run.get('device_label',run.get('device','unknown')))}</div>
    <div class="ol-label">FLIGHT</div><div>T+ {f['timestamp_s']:.2f}s &nbsp; ALT {escape(t.get('altitude_m'))} m<br>VEL XYZ {escape(velocity)} m/s</div>
    <div class="ol-label">OPENLANDER STATE</div><div class="ol-value ol-accent">{escape(f['state'])}</div>
    <div class="ol-label">SELECTED LANDING ZONE</div><div class="ol-value">{escape(f['selected_candidate'])} &nbsp; {f['score']:.0%}</div>
    <div class="ol-label">DRIFT</div><div>{d['visual_drift_speed']:.1f} px/s &nbsp; ↗ ({d['vehicle_dx_px_s']:.1f}, {d['vehicle_dy_px_s']:.1f})</div>
    <div class="ol-label">WHY THIS ZONE?</div><div>{escape(f['explanation'])}</div>
    <div class="ol-label">SIMULATED CONTROL REFERENCE</div><div>Image-space PID-like correction cue. Not flight-certified.</div></div>'''


def candidate_html(result,i):
    f=result["frames"][i]; chunks=[]
    for c in f["candidates"]:
        chosen=c["id"]==f["selected_candidate"]
        metrics=[("Terrain",c["terrain_score"]),("Clearance",c["clearance_score"]),("Reachability",c["reachability"])]
        bars="".join(f'<small>{n} {v:.0%}</small><div class="bar"><i style="width:{v*100:.0f}%"></i></div>' for n,v in metrics)
        chunks.append(f'<div class="candidate {"selected" if chosen else ""}"><b>{c["id"]}</b> &nbsp; {c["total_score"]:.0%} {"· SELECTED" if chosen else ""}{bars}</div>')
    return '<div class="ol-label">CANDIDATE BATTLE</div>'+"".join(chunks)


def timeline_html(result,i):
    now=result["frames"][i]["timestamp_s"]
    rows=[]
    for event in result["events"]:
        rows.append(f'<div class="event {"current" if event["timestamp_s"]<=now else ""}">{event["timestamp_s"]:05.2f}s &nbsp; {escape(event["event"])}</div>')
    return '<div class="ol-label">DECISION TIMELINE</div>'+"".join(rows)


def summary_md(result,i):
    if i<len(result["frames"])-1: return ""
    s=result["mission_summary"]
    return f"""## MISSION COMPLETE

**{s['result']}**

Final landing zone: **{s['final_landing_zone']}** · Safety score: **{s['final_safety_score']:.0%}** · Locked at: **{s['target_locked_at_s'] if s['target_locked_at_s'] is not None else 'final approach'} s**

Retargets: {s['retarget_events']} · Average analysis: {s['average_analysis_time_ms']} ms/frame · Total: {s['total_analysis_time_s']} s

**REPLAY MISSION** with Play or the timeline controls.
"""


def render(run_path,i,view):
    folder,run,result=load_data(run_path); i=max(0,min(int(i or 0),len(result["frames"])-1))
    return engine.frame_view(folder,i,view),decision_html(run,result,i),candidate_html(result,i),timeline_html(result,i),summary_md(result,i),i


def open_run(selected,manual):
    path=(manual or "").strip() or selected
    if not path: raise gr.Error("Choose a processed run or enter its folder path.")
    folder,run,result=load_data(path)
    first=render(folder,0,"Final Decision")
    return (folder,gr.update(maximum=len(result["frames"])-1,value=0),*first[:-1],f"Loaded cached run — no AI inference: {folder}")


def analyze(selected,manual,progress=gr.Progress()):
    path=(manual or "").strip() or selected
    if not path: raise gr.Error("Choose a raw scenario or enter its folder path.")
    progress(0,desc="Validating scenario…")
    messages=[]
    def note(msg):
        messages.append(msg); progress(None,desc=msg)
    try: folder=engine.analyze_scenario(path,note)
    except Exception as e: raise gr.Error(str(e)) from e
    _,run,result=load_data(folder); first=render(folder,0,"Final Decision")
    return (folder,gr.update(maximum=len(result["frames"])-1,value=0),*first[:-1],f"Analysis complete: {folder}")


def move(run_path,i,delta,view): return render(run_path,int(i)+delta,view)
def select_view(run_path,i,view): return render(run_path,i,view)
def toggle_play(playing):
    active=not playing
    return active,("PAUSE" if active else "PLAY"),gr.update(active=active)
def tick(run_path,i,playing,speed,view):
    if not run_path or not playing: return render(run_path,i,view) if run_path else (None,"","","","",i)
    _,_,result=load_data(run_path)
    return render(run_path,(int(i)+1)%len(result["frames"]),view)
def playback_interval(speed): return {"0.5x":1.3,"1x":.65,"2x":.325}[speed]
def inspect(run_path,i,evt: gr.SelectData):
    if not run_path: return "Load a mission first."
    xy=evt.index
    if not isinstance(xy,(list,tuple)) or len(xy)<2: return "Click within the image to inspect a landing point."
    return engine.inspect_point(run_path,int(i),float(xy[0]),float(xy[1]))


device,device_label=engine.compute_device()
scenarios=engine.list_folders("scenario"); runs=engine.list_folders("run")

with gr.Blocks(title="OpenLander Mission Control") as demo:
    run_state=gr.State(""); playing=gr.State(False)
    gr.HTML(f'''<div class="ol-hero"><div class="ol-kicker">AUTONOMOUS LANDING MISSION CONTROL</div><div class="ol-title">OPENLANDER</div><div class="ol-sub">Real AI perception, temporal drift, reachable-envelope reasoning, and explainable landing decisions. Ready compute device: <b>{device_label}</b>.</div></div>''')
    with gr.Tabs():
        with gr.Tab("ANALYZE RAW SCENARIO"):
            with gr.Row():
                scenario_dd=gr.Dropdown(scenarios,value=scenarios[0] if scenarios else None,label="Scenario folder",scale=2)
                scenario_path=gr.Textbox(label="Or compatible folder path",placeholder="/path/to/scenario",scale=2)
                analyze_btn=gr.Button("ANALYZE SCENARIO",variant="primary",scale=1)
        with gr.Tab("OPEN PROCESSED RUN"):
            with gr.Row():
                run_dd=gr.Dropdown(runs,value=runs[0] if runs else None,label="Processed run",scale=2)
                run_path=gr.Textbox(label="Or compatible folder path",placeholder="/path/to/run",scale=2)
                open_btn=gr.Button("OPEN PROCESSED RUN",variant="primary",scale=1)
    status=gr.Textbox(label="Mission status",interactive=False)
    with gr.Row():
        with gr.Column(scale=7,min_width=600):
            main_image=gr.Image(label="FINAL DECISION",type="filepath",interactive=False,height=720)
            with gr.Row():
                prev_btn=gr.Button("◀ PREVIOUS"); play_btn=gr.Button("PLAY",variant="primary"); next_btn=gr.Button("NEXT ▶")
                speed=gr.Radio(["0.5x","1x","2x"],value="1x",label="Speed")
            timeline=gr.Slider(0,9,value=0,step=1,label="MISSION TIMELINE")
            view=gr.Radio(VIEWS,value="Final Decision",label="VIEW MODE")
        with gr.Column(scale=4,min_width=390):
            decision=gr.HTML(elem_classes="ol-panel")
            battle=gr.HTML(elem_classes="ol-panel")
            why=gr.Markdown("### WHY NOT LAND HERE?\n\nClick the mission image to inspect any saved score-map location.",elem_classes="ol-panel")
    with gr.Row():
        events=gr.HTML(elem_classes="ol-panel")
        summary=gr.Markdown(elem_classes="ol-panel")
    timer=gr.Timer(.65,active=False)
    outputs=[main_image,decision,battle,events,summary,timeline]
    panel_outputs=outputs[:-1]
    analyze_btn.click(analyze,[scenario_dd,scenario_path],[run_state,timeline,main_image,decision,battle,events,summary,status])
    open_btn.click(open_run,[run_dd,run_path],[run_state,timeline,main_image,decision,battle,events,summary,status])
    prev_btn.click(lambda r,i,v:move(r,i,-1,v),[run_state,timeline,view],outputs)
    next_btn.click(lambda r,i,v:move(r,i,1,v),[run_state,timeline,view],outputs)
    timeline.change(lambda r,i,v:render(r,i,v)[:-1],[run_state,timeline,view],panel_outputs)
    view.change(lambda r,i,v:select_view(r,i,v)[:-1],[run_state,timeline,view],panel_outputs)
    play_btn.click(toggle_play,[playing],[playing,play_btn,timer])
    speed.change(playback_interval,speed,timer)
    timer.tick(tick,[run_state,timeline,playing,speed,view],outputs)
    main_image.select(inspect,[run_state,timeline],why)

if __name__ == "__main__":
    (ROOT/"scenarios").mkdir(exist_ok=True); (ROOT/"runs").mkdir(exist_ok=True); (ROOT/"models_cache").mkdir(exist_ok=True)
    demo.launch(inbrowser=True,show_error=True,css=CSS,theme=gr.themes.Base(primary_hue="cyan",neutral_hue="slate"))
