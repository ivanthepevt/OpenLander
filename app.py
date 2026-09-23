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
:root{--ol-bg:#060d13;--ol-card:#0d1821;--ol-card2:#111f2a;--ol-line:#2d4656;--ol-cyan:#55d9ff;--ol-green:#5af0ad;--ol-text:#f2f8fb;--ol-secondary:#b6c8d3;--ol-muted:#8fa8b7}
body,.gradio-container{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}
.gradio-container{background:radial-gradient(circle at 50% -18%,#17374a,#060d13 48%)!important;color:var(--ol-text)!important;max-width:1640px!important;padding:22px!important}
.gradio-container .form,.gradio-container .block:not(.image-container){background:var(--ol-card)!important;border-color:var(--ol-line)!important}
.gradio-container input,.gradio-container textarea{background:var(--ol-card2)!important;color:var(--ol-text)!important;border-color:#365365!important;font-size:14px!important}
.gradio-container label,.gradio-container label *,.gradio-container .block-info{color:#d4e2e9!important;font-weight:650!important}
.gradio-container label[data-testid$="-radio-label"]{background:#101d27!important;border:1px solid #355162!important;color:#bfd0d9!important}
.gradio-container label[data-testid$="-radio-label"] span{color:#bfd0d9!important}
.gradio-container label[data-testid$="-radio-label"].selected{background:#123849!important;border-color:var(--ol-cyan)!important;box-shadow:inset 0 0 0 1px rgba(89,211,240,.25)}
.gradio-container label[data-testid$="-radio-label"].selected span{color:#f2fbff!important}
.gradio-container label[data-testid="block-label"]{background:#0d202b!important;border:1px solid #355667!important;color:#dff8ff!important}
.gradio-container label[data-testid="block-label"] *{color:#62d9f6!important}
.gradio-container button[role=tab]{color:#a9c0cc!important}.gradio-container button[role=tab][aria-selected=true]{color:var(--ol-cyan)!important}
.gradio-container button{border-radius:8px!important;font-weight:750!important;letter-spacing:.025em!important;border-color:#38576a!important;min-height:42px!important}
.gradio-container button.primary{background:#24a9cf!important;color:#041017!important;border-color:#62d9f6!important}
.gradio-container button:hover{border-color:var(--ol-cyan)!important;filter:brightness(1.08)}
.gradio-container input[type=range]{accent-color:var(--ol-cyan)!important}
.ol-hero{padding:23px 28px;border:1px solid #315064;background:linear-gradient(125deg,#112633,#09131b);border-radius:14px;margin-bottom:14px;box-shadow:0 15px 45px rgba(0,0,0,.25)}
.ol-kicker{font:750 11px ui-monospace,SFMono-Regular,monospace;letter-spacing:.22em;color:var(--ol-cyan)!important}
.ol-title{font-size:34px;font-weight:800;letter-spacing:.025em;margin:5px 0 3px;color:#fff!important}.ol-sub,.ol-sub b{color:#c0d2dc!important;max-width:900px;line-height:1.5}.ol-sub b{color:var(--ol-green)!important}
.ol-panel{border:1px solid var(--ol-line)!important;background:rgba(13,24,33,.97)!important;border-radius:12px!important;color:var(--ol-text)!important;padding:2px!important;box-shadow:0 10px 28px rgba(0,0,0,.16)}
.ol-panel h2,.ol-panel h3,.ol-panel p,.ol-panel strong{color:var(--ol-text)!important}.ol-panel code{color:var(--ol-cyan)!important;background:#08131a!important}
.ol-readout{color:var(--ol-text);padding:14px 16px;line-height:1.45}.ol-eyebrow,.ol-label{font-size:10px;font-weight:800;letter-spacing:.16em;color:#9fb8c6!important;text-transform:uppercase}.ol-heading{font-size:20px;font-weight:780;color:#fff!important;margin:3px 0 13px}.ol-badge{display:inline-block;padding:3px 7px;margin-left:6px;border:1px solid #3f687a;border-radius:99px;color:var(--ol-cyan)!important;background:#0b2530}
.ol-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 12px}.ol-metric{background:#09141c;border:1px solid #223b4a;border-radius:8px;padding:9px 10px}.ol-metric .ol-label{display:block;margin-bottom:3px}.ol-value{font:700 16px ui-monospace,SFMono-Regular,monospace;color:#fff!important}.ol-value-lg{font-size:21px}.ol-accent{color:var(--ol-green)!important}.ol-copy{color:#d5e3ea!important;margin-top:5px}.ol-separator{height:1px;background:#233b49;margin:12px 0}
.candidate{padding:11px 12px;margin:8px 10px;border:1px solid #294657;border-radius:9px;background:#09151d;color:#dcecf4!important}.candidate.selected{border-color:var(--ol-green);box-shadow:inset 4px 0 var(--ol-green),0 0 0 1px rgba(90,240,173,.12);background:#0d211e}.candidate-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.candidate-head b{font-size:16px;color:#fff!important}.selected-tag{font-size:10px;font-weight:800;letter-spacing:.1em;color:#051610!important;background:var(--ol-green);padding:3px 6px;border-radius:4px}.metric-row{display:flex;justify-content:space-between;font-size:11px}.metric-row span,.metric-row b{color:#c8d9e1!important}.bar{height:6px;background:#1b2d38;border-radius:9px;overflow:hidden;margin:4px 0 8px}.bar i{display:block;height:100%;background:linear-gradient(90deg,#3bb8dd,#5af0ad)}
.panel-title{padding:12px 13px 3px;font-size:10px;font-weight:800;letter-spacing:.16em;color:#a6bfcc!important}.event{color:#b6cad5!important;border-left:2px solid #355465;padding:6px 10px;margin:3px 10px;font:600 12px ui-monospace,SFMono-Regular,monospace}.event.current{color:#fff!important;border-color:var(--ol-green);background:#10251f}.event-time{color:var(--ol-cyan)!important;margin-right:8px}
#event-panel{max-height:300px;overflow:auto}.export-card{padding:5px}.export-card .wrap{gap:8px!important}
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
    <div class="ol-eyebrow">SCENARIO <span class="ol-badge">{escape(s.get('mission_mode','SAFETY'))}</span></div><div class="ol-heading">{escape(s.get('title',s.get('scenario_name')))}</div>
    <div class="ol-grid">
      <div class="ol-metric"><span class="ol-label">MISSION TIME</span><div class="ol-value ol-value-lg">T+ {f['timestamp_s']:.3f}s</div></div>
      <div class="ol-metric"><span class="ol-label">FRAME</span><div class="ol-value">{i+1:03d} / {len(result['frames']):03d}</div></div>
      <div class="ol-metric"><span class="ol-label">ALTITUDE</span><div class="ol-value">{escape(t.get('altitude_m'))} m</div></div>
      <div class="ol-metric"><span class="ol-label">COMPUTE</span><div class="ol-value">{escape(run.get('device_label',run.get('device','unknown')))}</div></div>
    </div>
    <div class="ol-label">OPENLANDER STATE</div><div class="ol-value ol-value-lg ol-accent">{escape(f['state'])}</div>
    <div class="ol-separator"></div><div class="ol-label">SELECTED LANDING ZONE</div><div class="ol-value ol-value-lg">{escape(f['selected_candidate'])} &nbsp; {f['score']:.0%}</div>
    <div class="ol-label" style="margin-top:10px">VELOCITY XYZ / DRIFT</div><div class="ol-copy">{escape(velocity)} m/s &nbsp; · &nbsp; {d['visual_drift_speed']:.1f} px/s</div>
    <div class="ol-separator"></div><div class="ol-label">WHY THIS ZONE?</div><div class="ol-copy">{escape(f['explanation'])}</div>
    <div class="ol-label" style="margin-top:11px">SIMULATED CONTROL REFERENCE</div><div class="ol-copy">Image-space correction cue · not flight-certified.</div></div>'''


def candidate_html(result,i):
    f=result["frames"][i]; chunks=[]
    for c in f["candidates"]:
        chosen=c["id"]==f["selected_candidate"]
        metrics=[("Terrain",c["terrain_score"]),("Clearance",c["clearance_score"]),("Reachability",c["reachability"])]
        bars="".join(f'<div class="metric-row"><span>{n}</span><b>{v:.0%}</b></div><div class="bar"><i style="width:{v*100:.0f}%"></i></div>' for n,v in metrics)
        tag='<span class="selected-tag">SELECTED</span>' if chosen else ""
        chunks.append(f'<div class="candidate {"selected" if chosen else ""}"><div class="candidate-head"><b>{c["id"]} &nbsp; {c["total_score"]:.0%}</b>{tag}</div>{bars}</div>')
    return '<div class="panel-title">CANDIDATE BATTLE</div>'+"".join(chunks)


def timeline_html(result,i):
    now=result["frames"][i]["timestamp_s"]
    rows=[]
    for event in result["events"]:
        rows.append(f'<div class="event {"current" if event["timestamp_s"]<=now else ""}"><span class="event-time">{event["timestamp_s"]:06.3f}s</span>{escape(event["event"])}</div>')
    return '<div class="panel-title">DECISION TIMELINE · MAJOR EVENTS</div>'+"".join(rows)


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
    return active,("PAUSE" if active else "PLAY"),gr.update(active=active),0.0
def tick(run_path,i,playing,speed,view,clock):
    if not run_path or not playing:
        frame=render(run_path,i,view) if run_path else (None,"","","","",i)
        return clock,*frame
    _,_,result=load_data(run_path)
    timestamps=[float(f["timestamp_s"]) for f in result["frames"]]
    deltas=sorted(b-a for a,b in zip(timestamps,timestamps[1:]) if b>a)
    frame_dt=deltas[len(deltas)//2] if deltas else 1/30
    clock=float(clock or 0)+.1*{"0.5x":.5,"1x":1.,"2x":2.}[speed]
    advance=int(clock/frame_dt)
    if advance: clock-=advance*frame_dt
    return clock,*render(run_path,(int(i)+advance)%len(result["frames"]),view)
def inspect(run_path,i,evt: gr.SelectData):
    if not run_path: return "Load a mission first."
    xy=evt.index
    if not isinstance(xy,(list,tuple)) or len(xy)<2: return "Click within the image to inspect a landing point."
    return engine.inspect_point(run_path,int(i),float(xy[0]),float(xy[1]))


def export_cached(run_path,mode,progress=gr.Progress()):
    if not run_path: raise gr.Error("Open a processed run before exporting.")
    yield "Export started — encoding cached frames only…",None
    progress(0,desc="Encoding cached replay…")
    try: output=engine.export_video(run_path,mode,30)
    except Exception as exc: raise gr.Error(str(exc)) from exc
    progress(1,desc="Export finished")
    yield f"Export finished: {output}",output


device,device_label=engine.compute_device()
scenarios=engine.list_folders("scenario"); runs=engine.list_folders("run")

with gr.Blocks(title="OpenLander Mission Control") as demo:
    run_state=gr.State(""); playing=gr.State(False); play_clock=gr.State(0.0)
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
            with gr.Group(elem_classes="export-card"):
                with gr.Row():
                    export_mode=gr.Radio(["Final Decision","Debug Dashboard"],value="Final Decision",label="EXPORT VIEW",scale=2)
                    export_btn=gr.Button("EXPORT VIDEO",variant="primary",scale=1)
                export_status=gr.Markdown("Export uses cached frames only.")
                export_file=gr.File(label="Exported replay",interactive=False)
        with gr.Column(scale=4,min_width=390):
            decision=gr.HTML(elem_classes="ol-panel")
            battle=gr.HTML(elem_classes="ol-panel")
            why=gr.Markdown("### WHY NOT LAND HERE?\n\nClick the mission image to inspect any saved score-map location.",elem_classes="ol-panel")
    with gr.Row():
        events=gr.HTML(elem_classes="ol-panel",elem_id="event-panel")
        summary=gr.Markdown(elem_classes="ol-panel")
    timer=gr.Timer(.1,active=False)
    outputs=[main_image,decision,battle,events,summary,timeline]
    panel_outputs=outputs[:-1]
    analyze_btn.click(analyze,[scenario_dd,scenario_path],[run_state,timeline,main_image,decision,battle,events,summary,status])
    open_btn.click(open_run,[run_dd,run_path],[run_state,timeline,main_image,decision,battle,events,summary,status])
    prev_btn.click(lambda r,i,v:move(r,i,-1,v),[run_state,timeline,view],outputs)
    next_btn.click(lambda r,i,v:move(r,i,1,v),[run_state,timeline,view],outputs)
    timeline.input(lambda r,i,v:render(r,i,v)[:-1],[run_state,timeline,view],panel_outputs)
    view.change(lambda r,i,v:select_view(r,i,v)[:-1],[run_state,timeline,view],panel_outputs)
    play_btn.click(toggle_play,[playing],[playing,play_btn,timer,play_clock])
    timer.tick(tick,[run_state,timeline,playing,speed,view,play_clock],[play_clock,*outputs])
    main_image.select(inspect,[run_state,timeline],why)
    export_btn.click(export_cached,[run_state,export_mode],[export_status,export_file])

if __name__ == "__main__":
    (ROOT/"scenarios").mkdir(exist_ok=True); (ROOT/"runs").mkdir(exist_ok=True); (ROOT/"models_cache").mkdir(exist_ok=True)
    demo.launch(inbrowser=True,show_error=True,css=CSS,theme=gr.themes.Base(primary_hue="cyan",neutral_hue="slate"))
