"""Render a QC report and its recordings as one self-contained HTML page.

The console report tells you *that* the IMU lost nine seconds.  This page tells
you what every other sensor was doing at that moment, which marker it landed
between, and what the camera saw — because every stream is drawn against one
shared time axis anchored on RUN_START.

Findings split in two on the way in.  A finding carrying ``spans`` is
time-anchored and becomes a band or a flag on the timeline at the instant it
happened; one without (a MAD fraction, a dead channel) is a property of the
whole stream and goes in the card header, where it cannot masquerade as an
event that occurred at t=0.

The output has no external references — data URIs and inline script — so it
opens from ``file://`` and can be handed to someone else as a single file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .qc_payload import FRAME_FPS, FRAME_W, JPEG_Q
from .qc_streams import extractor_for


@dataclass
class Options:
    frames: bool = True
    fps: float = FRAME_FPS
    thumb_w: int = FRAME_W
    jpeg_q: int = JPEG_Q
    max_thumbs: int = 400
    filter: bool = True                # 嵌入滤波副本(原始数据永不修改)
    filter_presets: dict | None = None  # checker.yaml "filter:" 节覆盖
    _wanted: dict = field(default_factory=dict)

    def want_times(self, stream: str) -> list[float]:
        """Instants this stream was flagged at — extra thumbnails go here."""
        return self._wanted.get(stream, [])


# =============================================================================
# Payload
# =============================================================================

def _epoch(report: dict) -> float:
    w = report.get("window")
    if w:
        return float(w["t0"])
    starts = [s["t0"] for st in report.get("streams", {}).values()
              for s in st.get("series", {}).values() if s.get("t0")]
    return float(min(starts)) if starts else 0.0


def _split_findings(findings: list, t0: float):
    """Time-anchored findings become events; the rest stay stream-level."""
    events, notes = [], []
    for f in findings:
        base = {"level": f.get("level", "INFO"), "check": f.get("check", ""),
                "msg": f.get("message", ""), "subject": f.get("subject", ""),
                "thr": f.get("threshold"), "obs": f.get("observed")}
        spans = f.get("spans") or []
        if spans:
            for sp in spans:
                events.append({**base, "t": float(sp["t"]) - t0,
                               "dur": float(sp.get("dur", 0.0)),
                               "detail": sp.get("msg", "")})
        else:
            notes.append(base)
    return events, notes


def _markers(report: dict) -> list[dict]:
    st = report.get("streams", {}).get("marker")
    if not st:
        return []
    items = (st.get("stats", {}).get("markers") or {}).get("items") or []
    return [{"t": float(m["t_offset"]), "name": m.get("name", ""),
             "code": m.get("code")} for m in items]


def _summary(st: dict) -> str:
    parts = []
    for label, s in list(st.get("series", {}).items())[:3]:
        if s.get("n"):
            parts.append(f"{label} {s['n']} 样本 @{s.get('rate', 0):.1f}/s")
    return " · ".join(parts)


def _stats(st: dict) -> list[dict]:
    """The checks' own numbers, flattened for display.

    These matter even when nothing tripped a threshold: a frozen fraction of
    0.25 raises no finding (no single run exceeds the limit) but is exactly
    what a reader wants to see before trusting the recording.
    """
    out = []
    for check, values in sorted(st.get("stats", {}).items()):
        if not isinstance(values, dict):
            continue
        bits = [(k, v) for k, v in values.items()
                if isinstance(v, (int, float, str, bool)) and v is not None]
        if bits:
            out.append({"check": check, "bits": [[k, _num(v)] for k, v in bits]})
    return out


def _num(v):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    return round(v, 4) if abs(v) < 1000 else round(v, 1)


def build_payload(report: dict, session_dir: Path | str | None = None,
                  opt: Options | None = None) -> dict:
    """Assemble everything the page draws, already on the RUN_START clock."""
    opt = opt or Options()
    root = Path(session_dir or report.get("session_dir", "."))
    t0 = _epoch(report)

    # Collect flagged instants first so the video pass can put a thumbnail on
    # each of them rather than only on its fixed cadence.
    for name, st in report.get("streams", {}).items():
        ts = [float(sp["t"]) - t0
              for f in st.get("findings", []) for sp in (f.get("spans") or [])]
        if ts:
            opt._wanted[name] = ts

    streams, lo, hi = [], [], []
    for name, st in report.get("streams", {}).items():
        events, notes = _split_findings(st.get("findings", []), t0)
        entry = {"name": name, "level": st.get("level", "INFO"),
                 "summary": _summary(st), "stats": _stats(st),
                 "rows": [], "thumbs": [], "events": events, "notes": notes}
        for s in st.get("series", {}).values():
            if s.get("t0") is not None:
                lo.append(s["t0"] - t0)
                hi.append(s["t1"] - t0)

        fn = extractor_for(name)
        d = root / name
        if fn is not None and d.is_dir():
            try:
                entry["rows"], entry["thumbs"] = _extract(fn, d, t0, opt)
            except Exception as exc:                      # noqa: BLE001
                # One unreadable stream must never cost the whole page.
                entry["notes"].append({"level": "WARN", "check": "Render",
                                       "msg": f"该流渲染失败: {exc}",
                                       "subject": "", "thr": None, "obs": None})
        if len(entry["thumbs"]) > opt.max_thumbs:
            keep = np.linspace(0, len(entry["thumbs"]) - 1, opt.max_thumbs)
            entry["thumbs"] = [entry["thumbs"][int(i)] for i in keep]
        streams.append(entry)

    win = report.get("window")
    window = [win["t0"] - t0, win["t1"] - t0] if win else None
    span_lo = min(lo) if lo else 0.0
    span_hi = max(hi) if hi else 1.0

    problems = sorted(
        ({"stream": s["name"], **e} for s in streams for e in s["events"]),
        key=lambda p: p["t"])
    problems += [{"stream": s["name"], "t": None, "dur": 0, **n}
                 for s in streams for n in s["notes"]]

    return {"session": str(root), "level": report.get("level", "INFO"),
            "span": [span_lo, span_hi], "window": window,
            "view": window or [span_lo, span_hi],
            "markers": _markers(report), "streams": streams,
            "problems": problems}


def _extract(fn, d: Path, t0: float, opt: Options):
    npz = next(iter(sorted(d.glob("*.npz"))), None)
    if npz is None:
        return fn({}, d, t0, opt)
    with np.load(npz, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return fn(data, d, t0, opt)


# =============================================================================
# Page
# =============================================================================

_TOKEN = "/*__QC_PAYLOAD__*/"


def render_html(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return _TEMPLATE.replace(_TOKEN, blob, 1)


def build_page(report: dict, session_dir: Path | str | None = None,
               opt: Options | None = None) -> str:
    return render_html(build_payload(report, session_dir, opt))


_TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>会话质量检查</title>
<style>
:root{
  color-scheme:light;
  --surface-1:#fcfcfb; --page:#f9f9f7;
  --ink-1:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#eda100;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --ink-1:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
}}
:root[data-theme=dark]{
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --ink-1:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink-1);
  font:13px/1.5 ui-sans-serif,system-ui,"Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:12px;
  padding:8px 16px;background:var(--surface-1);border-bottom:1px solid var(--border)}
header h1{font-size:14px;margin:0;font-weight:600}
.meta{color:var(--ink-2);font-size:12px}
button{font:inherit;padding:3px 10px;border:1px solid var(--border);border-radius:6px;
  background:var(--surface-1);color:var(--ink-1);cursor:pointer}
button:hover{border-color:var(--axis)}
#toolbar{position:sticky;top:41px;z-index:9;display:flex;align-items:center;gap:8px;
  padding:6px 16px;background:var(--surface-1);border-bottom:1px solid var(--border);
  font-size:12px;color:var(--ink-2);flex-wrap:wrap}
.badge{display:inline-flex;align-items:center;gap:4px;padding:1px 7px;border-radius:999px;
  font-size:11px;font-weight:600;border:1px solid}
.b-INFO{color:var(--ink-2);border-color:var(--axis)}
.b-WARN{color:#8a5d00;border-color:var(--warning);background:#fab2191f}
.b-ERROR{color:var(--critical);border-color:var(--critical);background:#d03b3b1f}
:root[data-theme=dark] .b-WARN{color:var(--warning)}
#master-wrap{padding:8px 16px 0}
#master{width:100%;display:block;background:var(--surface-1);
  border:1px solid var(--border);border-radius:8px;cursor:crosshair}
#layout{display:flex;gap:12px;padding:12px 16px 32px;align-items:flex-start}
#problems{flex:0 0 320px;position:sticky;top:92px;max-height:calc(100vh - 120px);
  overflow-y:auto;background:var(--surface-1);border:1px solid var(--border);
  border-radius:8px;padding:8px}
#problems h2{font-size:12px;margin:2px 4px 8px;color:var(--ink-2);font-weight:600}
.p{display:block;width:100%;text-align:left;border:0;background:none;padding:5px 6px;
  border-radius:6px;cursor:pointer;border-left:3px solid transparent;margin-bottom:2px}
.p:hover{background:var(--page)}
.p .pt{color:var(--muted);font-variant-numeric:tabular-nums;font-size:11px}
.p .pm{color:var(--ink-1);font-size:12px}
.p.lv-WARN{border-left-color:var(--warning)}
.p.lv-ERROR{border-left-color:var(--critical)}
.p.flash{background:var(--page);outline:1px solid var(--axis)}
#cards{flex:1;min-width:0;display:flex;flex-direction:column;gap:12px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
  overflow:hidden}
.card-head{display:flex;align-items:baseline;gap:8px;padding:7px 10px;
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.card-head strong{font-size:13px}
.card-head .sum{color:var(--ink-2);font-size:12px}
.card-head .filt{margin-left:auto;font:11px system-ui;color:var(--ink-2);
  background:none;border:1px solid var(--axis);border-radius:6px;
  padding:1px 8px;cursor:pointer}
.card-head .filt.on{color:var(--good);border-color:var(--good)}
.notes{padding:6px 10px;display:flex;flex-direction:column;gap:3px;font-size:12px}
.note{color:var(--ink-2)}
.note b{color:var(--ink-1);font-weight:600}
.stats{padding:4px 10px 8px;font-size:11px;color:var(--ink-2);
  display:flex;flex-wrap:wrap;gap:2px 14px}
.stats .st{white-space:nowrap}
.stats .st i{font-style:normal;color:var(--muted)}
.stats .st b{font-weight:600;color:var(--ink-1);font-variant-numeric:tabular-nums}
.mk{padding:4px 10px 10px;display:grid;
  grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:2px 12px;font-size:12px}
.mk div{display:flex;gap:8px;color:var(--ink-2)}
.mk b{color:var(--ink-1);font-weight:600;min-width:96px}
.mk span{font-variant-numeric:tabular-nums;color:var(--muted)}
.frames{display:flex;gap:4px;overflow-x:auto;padding:6px 10px}
.frames figure{margin:0;flex:0 0 auto;cursor:pointer}
.frames img{display:block;width:150px;border:2px solid transparent;border-radius:4px}
.frames figure.flag img{border-color:var(--critical)}
.frames figure.sel img{border-color:var(--series-1)}
.frames figcaption{font-size:10px;color:var(--muted);text-align:center;
  font-variant-numeric:tabular-nums}
.card canvas{width:100%;display:block}
#tip{position:fixed;z-index:50;pointer-events:none;display:none;max-width:340px;
  background:var(--surface-1);border:1px solid var(--axis);border-radius:6px;
  padding:5px 8px;font-size:12px;box-shadow:0 2px 10px rgba(0,0,0,.16)}
#tip .k{color:var(--muted)}
</style></head><body>

<header>
  <h1>会话质量检查</h1>
  <span class="meta" id="sess"></span>
  <span id="lvl"></span>
  <button id="theme" style="margin-left:auto">主题</button>
</header>

<div id="toolbar">
  <span>视窗</span>
  <input id="ra" size="7" inputmode="decimal"> –
  <input id="rb" size="7" inputmode="decimal">
  <button id="apply">应用</button>
  <button id="all">显示全部</button>
  <button id="win">运行窗口</button>
  <span id="ph" style="margin-left:auto;font-variant-numeric:tabular-nums"></span>
</div>

<div id="master-wrap"><canvas id="master" height="200"></canvas></div>

<div id="layout">
  <aside id="problems"><h2>问题清单</h2><div id="plist"></div></aside>
  <main id="cards"></main>
</div>
<div id="tip"></div>

<script>
const DATA = /*__QC_PAYLOAD__*/;
const PL = 84, PR = 14, TOPB = 16;      // plot padding: label gutter, right, top band

/* ---------- binary payload -> typed arrays ---------- */
function bytes(b64){const s=atob(b64),a=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a;}
function f32(b64){const b=bytes(b64);return new Float32Array(b.buffer,b.byteOffset,b.length/4);}
function sTime(s){
  if(s._t)return s._t;
  if(s.tstride){                        // uniform series shipped as a stride
    const p=f32(s.t), n=sVal(s).length, a=new Float32Array(n);
    for(let i=0;i<n;i++)a[i]=p[0]+i*s.tstride;return s._t=a;
  }
  return s._t=f32(s.t);
}
function sVal(s){
  if(s._y)return s._y;
  const b=bytes(s.y), q=new Int16Array(b.buffer,b.byteOffset,b.length/2);
  const y=new Float32Array(q.length), k=(s.hi-s.lo)/32767;
  for(let i=0;i<q.length;i++)y[i]=q[i]===-32768?NaN:q[i]*k+s.lo;
  return s._y=y;
}
function sValF(s){
  // Filtered display copy — decoded lazily so a page that never toggles
  // the filter never pays the atob() for it.
  if(s._yf)return s._yf;
  if(!s.yf)return null;
  const b=bytes(s.yf), q=new Int16Array(b.buffer,b.byteOffset,b.length/2);
  const y=new Float32Array(q.length), k=(s.fhi-s.flo)/32767;
  for(let i=0;i<q.length;i++)y[i]=q[i]===-32768?NaN:q[i]*k+s.flo;
  return s._yf=y;
}

/* ---------- theme-aware colours: read at paint time, never baked in ---------- */
function cv(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
const C={
  slot:i=>cv('--series-'+(((i-1)%4)+1)),
  level:l=>l==='ERROR'?cv('--critical'):l==='WARN'?cv('--warning'):cv('--muted'),
  ink:()=>cv('--ink-1'), ink2:()=>cv('--ink-2'), muted:()=>cv('--muted'),
  grid:()=>cv('--grid'), axis:()=>cv('--axis'), surf:()=>cv('--surface-1')
};

/* ---------- shared view state: one axis for master and every card ---------- */
const SPAN=DATA.span, S={x0:DATA.view[0], x1:DATA.view[1], ph:DATA.view[0], filt:{}};
const cards=[];
function xPx(w,t){return (t-S.x0)/(S.x1-S.x0)*w;}
function tAt(w,px){return S.x0+px/w*(S.x1-S.x0);}
function setView(a,b){
  if(b-a<0.02){const m=(a+b)/2;a=m-0.01;b=m+0.01;}
  S.x0=a;S.x1=b;ra.value=a.toFixed(2);rb.value=b.toFixed(2);renderAll();
}
function fit(c,h){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;
  c.width=Math.max(1,r.width*d);c.height=h*d;c.style.height=h+'px';
  const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,h];}
function niceStep(span,px){const target=span/Math.max(2,px/90);
  for(const s of [.05,.1,.2,.5,1,2,5,10,20,30,60,120,300])if(s>=target)return s;return 600;}

/* ---------- shared chrome: grid, event bands, marker lines ---------- */
function drawGrid(x,w,h){
  const st=niceStep(S.x1-S.x0,w);
  x.strokeStyle=C.grid();x.fillStyle=C.muted();x.lineWidth=1;x.font='10px system-ui';
  for(let t=Math.ceil(S.x0/st)*st;t<=S.x1;t+=st){
    const px=Math.round(PL+xPx(w,t))+.5;
    x.beginPath();x.moveTo(px,TOPB);x.lineTo(px,h-13);x.stroke();
    x.fillText(t.toFixed(st<1?2:0)+'s',px+2,h-3);
  }
}
function eventBands(e,h,geom){
  // An event belongs to the rows its finding named, not to the whole card:
  // an IMU dropout drawn over the EMG channels claims those went missing
  // too.  Scoping also stops dozens of full-height bands from smearing into
  // one another on a stream with many findings.
  if(e.subject&&geom&&geom.length){
    const hit=geom.filter(g=>g.row.src===e.subject);
    if(hit.length)return hit.map(g=>[g.y-1,g.h+4]);
  }
  return [[TOPB,h-TOPB-13]];
}
function drawEvents(x,w,h,events,geom){
  x.font='10px system-ui';
  for(const e of events){
    const dur=e.dur||0;
    if(e.t+dur<S.x0||e.t>S.x1)continue;
    const col=C.level(e.level), a=PL+xPx(w,e.t);
    for(const [by,bh] of eventBands(e,h,geom)){
      if(dur>0.15){
        const b=PL+xPx(w,Math.min(e.t+dur,S.x1)), wd=Math.max(b-a,2);
        x.fillStyle=col+'2e';x.fillRect(a,by,wd,bh);
        x.strokeStyle=col;x.lineWidth=1;
        x.strokeRect(Math.round(a)+.5,by+.5,wd,bh-1);
        if(wd>44){        // the band's own extent, so it can be read exactly
          x.fillStyle=col;
          x.fillText(dur.toFixed(2)+'s',a+3,by+11);
        }
      }else{
        x.fillStyle=col;x.beginPath();
        x.moveTo(a-4,by);x.lineTo(a+4,by);x.lineTo(a,by+8);
        x.closePath();x.fill();
        x.fillRect(a-.5,by,1,bh);
      }
    }
  }
}
function drawMarkers(x,w,h,labels){
  x.strokeStyle=C.axis();x.fillStyle=C.muted();x.lineWidth=1;x.font='10px system-ui';
  let lastLabel=-1e9;
  for(const m of DATA.markers){
    if(m.t<S.x0||m.t>S.x1)continue;
    const px=Math.round(PL+xPx(w,m.t))+.5;
    x.setLineDash([3,3]);x.beginPath();x.moveTo(px,TOPB);x.lineTo(px,h-13);x.stroke();
    x.setLineDash([]);
    // Adjacent markers can share a tenth of a second (RUN_START, FIX_ON);
    // stacking their labels just makes a smear.  The line still marks every
    // one — the tooltip names whichever you point at.
    if(labels && px-lastLabel>=11){
      lastLabel=px;
      x.save();x.translate(px+3,TOPB+2);x.rotate(-Math.PI/2);
      x.textAlign='right';x.fillText(m.name,0,0);x.restore();
    }
  }
}

/* ---------- master timeline ---------- */
function renderMaster(){
  const n=DATA.streams.length, h=Math.max(120,34+n*16+18);
  const [x,w0]=fit(master,h), w=w0-PL-PR;
  x.clearRect(0,0,w0,h);
  drawGrid(x,w,h);
  if(DATA.window){                       // the run window, as context
    const a=PL+xPx(w,DATA.window[0]), b=PL+xPx(w,DATA.window[1]);
    x.fillStyle=C.grid();x.globalAlpha=.5;x.fillRect(a,TOPB,b-a,h-TOPB-13);
    x.globalAlpha=1;
  }
  x.font='10px system-ui';
  DATA.streams.forEach((st,i)=>{
    const y=TOPB+4+i*16;
    x.fillStyle=C.ink2();x.textAlign='right';x.fillText(st.name,PL-6,y+9);
    x.textAlign='left';
    const b=streamBounds(st);
    if(b){
      const a=PL+xPx(w,b[0]), e=PL+xPx(w,b[1]);
      // The bar body must read as "this stream covers here" at a glance,
      // so it takes the axis tone rather than the near-invisible gridline.
      x.fillStyle=C.axis();x.globalAlpha=.55;
      x.fillRect(a,y,Math.max(e-a,1),11);x.globalAlpha=1;
      x.fillStyle=C.level(st.level);x.fillRect(a,y,Math.max(e-a,1),3);
    }
    for(const ev of st.events){
      if(ev.t<S.x0||ev.t>S.x1)continue;
      x.fillStyle=C.level(ev.level);
      x.fillRect(PL+xPx(w,ev.t)-1,y,Math.max(2,xPx(w,ev.dur||0)),11);
    }
  });
  drawMarkers(x,w,h,true);
  x.strokeStyle=C.level('ERROR');x.lineWidth=1;
  const p=Math.round(PL+xPx(w,S.ph))+.5;
  x.beginPath();x.moveTo(p,TOPB);x.lineTo(p,h-13);x.stroke();
}
function streamBounds(st){
  let a=Infinity,b=-Infinity;
  for(const r of st.rows)for(const s of (r.ser||[])){
    const t=sTime(s);if(t.length){a=Math.min(a,t[0]);b=Math.max(b,t[t.length-1]);}}
  return isFinite(a)?[a,b]:null;
}

/* ---------- one stream card ---------- */
function renderCard(card){
  const st=card.st, rows=st.rows;
  if(!rows.length)return;
  const useF=!!S.filt[st.name];
  let h=TOPB+13;for(const r of rows)h+=r.h+6;
  const [x,w0]=fit(card.cv,h), w=w0-PL-PR;
  x.clearRect(0,0,w0,h);
  drawGrid(x,w,h);

  let y=TOPB+2;
  card.geom=[];
  for(const r of rows){
    x.fillStyle=C.ink2();x.font='11px system-ui';x.textAlign='right';
    x.fillText(r.label,PL-6,y+12);x.textAlign='left';
    x.strokeStyle=C.grid();x.beginPath();
    x.moveTo(PL,y+r.h+3.5);x.lineTo(PL+w,y+r.h+3.5);x.stroke();
    drawLines(x,r,PL,y,w,r.h,useF);
    card.geom.push({row:r,y:y,h:r.h});
    y+=r.h+6;
  }
  drawEvents(x,w,h,st.events,card.geom);
  drawMarkers(x,w,h,false);
  x.strokeStyle=C.level('ERROR');x.globalAlpha=.6;x.lineWidth=1;
  const p=Math.round(PL+xPx(w,S.ph))+.5;
  x.beginPath();x.moveTo(p,TOPB);x.lineTo(p,h-13);x.stroke();x.globalAlpha=1;
}

function visibleRange(r,useF){
  // Rescale y to what is on screen: a fixed build-time range flattens the
  // trace to a line as soon as you zoom into a quiet stretch.
  let lo=Infinity,hi=-Infinity;
  for(const s of r.ser){
    const t=sTime(s),v=useF&&s.yf?sValF(s):sVal(s);
    for(let i=0;i<t.length;i++){
      if(t[i]<S.x0||t[i]>S.x1)continue;
      const y=v[i];if(!isFinite(y))continue;
      if(y<lo)lo=y;if(y>hi)hi=y;
    }
  }
  if(!isFinite(lo)){lo=r.ser[0].lo;hi=r.ser[0].hi;}
  if(hi-lo<1e-12){const m=(hi+lo)/2||0;lo=m-.5;hi=m+.5;}
  return [lo,hi];
}
function drawLines(x,r,ox,oy,w,h,useF){
  const [lo,hi]=visibleRange(r,useF), k=h-8;
  r._vis=[lo,hi];
  x.lineWidth=2;x.lineJoin='round';x.lineCap='round';
  r.ser.forEach((s,si)=>{
    const t=sTime(s),v=useF&&s.yf?sValF(s):sVal(s);
    x.strokeStyle=C.slot(s.slot);x.beginPath();
    let on=false;
    for(let i=0;i<t.length;i++){
      if(t[i]<S.x0||t[i]>S.x1){on=false;continue;}
      const py=v[i];
      if(!isFinite(py)){on=false;continue;}
      const px=ox+xPx(w,t[i]), yy=oy+4+(1-(py-lo)/(hi-lo))*k;
      on?x.lineTo(px,yy):(x.moveTo(px,yy),on=true);
    }
    x.stroke();
    if(r.ser.length>1){       // direct labels: the light-mode contrast relief
      x.fillStyle=C.slot(s.slot);x.font='10px system-ui';
      x.fillText(s.label,ox+w+3,oy+11+si*11);
    }
  });
  x.fillStyle=C.muted();x.font='10px system-ui';x.textAlign='right';
  x.fillText(fmt(lo)+'~'+fmt(hi)+(r.unit?' '+r.unit:''),ox+w-2,oy+10);
  x.textAlign='left';
}
function fmt(v){const a=Math.abs(v);
  return a>=1e4||(a<1e-3&&a>0)?v.toExponential(1):v.toPrecision(3).replace(/\.?0+$/,'');}

/* ---------- tooltip: the value under the cursor, which is the first thing
     anyone QC-ing a signal wants to know ---------- */
const tip=document.getElementById('tip');
function showTip(cx,cy,html){tip.innerHTML=html;tip.style.display='block';
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.min(cx+14,innerWidth-r.width-8)+'px';
  tip.style.top=Math.min(cy+14,innerHeight-r.height-8)+'px';}
function hideTip(){tip.style.display='none';}

function hoverCard(card,ev){
  const box=card.cv.getBoundingClientRect(), w=box.width-PL-PR;
  const t=tAt(w,ev.clientX-box.left-PL);
  if(t<S.x0||t>S.x1){hideTip();return;}
  const yy=ev.clientY-box.top;
  const g=(card.geom||[]).find(q=>yy>=q.y&&yy<=q.y+q.h);
  const near=(S.x1-S.x0)*8/w;
  let out='<div class="k">+'+t.toFixed(3)+'s</div>';

  if(g){
    const useF=!!S.filt[card.st.name];
    for(const s of g.row.ser){
      const tt=sTime(s),vv=useF&&s.yf?sValF(s):sVal(s);
      let bi=-1,bd=Infinity;
      for(let i=0;i<tt.length;i++){const d=Math.abs(tt[i]-t);
        if(d<bd){bd=d;bi=i;}}
      if(bi>=0&&bd<near*3)
        out+='<div><b style="color:'+C.slot(s.slot)+'">■</b> '+
          (s.label||g.row.label)+' <b>'+fmt(vv[bi])+'</b> '+(s.unit||'')+'</div>';
    }
  }
  for(const e of card.st.events){
    if(t>=e.t-near&&t<=e.t+(e.dur||0)+near)
      out+='<div style="color:'+C.level(e.level)+'">'+e.check+': '+e.msg+'</div>';
  }
  for(const m of DATA.markers)
    if(Math.abs(m.t-t)<near)out+='<div class="k">标记 '+m.name+'</div>';
  showTip(ev.clientX,ev.clientY,out);
}

/* ---------- build DOM ---------- */
const master=document.getElementById('master');
const ra=document.getElementById('ra'), rb=document.getElementById('rb');
document.getElementById('sess').textContent=DATA.session;
document.getElementById('lvl').innerHTML=
  '<span class="badge b-'+DATA.level+'">'+({INFO:'提示',WARN:'警告',ERROR:'错误'}[DATA.level]||DATA.level)+'</span>';

for(const st of DATA.streams){
  const el=document.createElement('section');el.className='card';
  const zh={INFO:'提示',WARN:'警告',ERROR:'错误'}[st.level]||st.level;
  const hasF=st.rows.some(r=>r.ser.some(s=>s.yf));
  let html='<div class="card-head"><strong>'+st.name+'</strong>'+
    '<span class="badge b-'+st.level+'">'+zh+'</span>'+
    '<span class="sum">'+(st.summary||'')+'</span>'+
    (hasF?'<button class="filt" title="切换原始/滤波显示">原始/滤波</button>':'')+
    '</div>';
  if(st.notes.length){
    html+='<div class="notes">'+st.notes.map(n=>{
      const bits=[];
      if(n.subject)bits.push(n.subject);
      if(n.thr!=null)bits.push('阈值 '+fmt(n.thr));
      if(n.obs!=null)bits.push('实测 '+fmt(n.obs));
      return '<div class="note"><b style="color:'+C.level(n.level)+'">●</b> '+
        n.check+' — '+n.msg+(bits.length?' <span class="k">('+bits.join(' · ')+')</span>':'')+'</div>';
    }).join('')+'</div>';
  }
  if(st.stats && st.stats.length){
    html+='<div class="stats">'+st.stats.map(g=>
      '<span class="st"><i>'+g.check+'</i> '+
      g.bits.map(([k,v])=>k+'=<b>'+v+'</b>').join(' ')+'</span>').join('')+'</div>';
  }
  if(st.name.startsWith('marker') && DATA.markers.length){
    html+='<div class="mk">'+DATA.markers.map(m=>
      '<div><b>'+m.name+'</b><span>+'+m.t.toFixed(2)+'s</span>'+
      '<span>code '+m.code+'</span></div>').join('')+'</div>';
  }
  if(st.thumbs.length){
    html+='<div class="frames">'+st.thumbs.map((f,i)=>
      '<figure data-t="'+f.t+'" class="'+(f.flag?'flag':'')+'">'+
      '<img loading="lazy" src="data:image/jpeg;base64,'+f.b64+'">'+
      '<figcaption>+'+f.t.toFixed(1)+'s</figcaption></figure>').join('')+'</div>';
  }
  if(st.rows.length)html+='<canvas></canvas>';
  el.innerHTML=html;
  const fb=el.querySelector('.filt');
  if(fb)fb.onclick=()=>{
    const on=!(S.filt[st.name]||false);
    S.filt[st.name]=on;
    fb.classList.toggle('on',on);
    renderAll();
  };
  document.getElementById('cards').appendChild(el);
  const cv0=el.querySelector('canvas');
  const card={st:st,cv:cv0,el:el};
  if(cv0){
    cv0.addEventListener('mousemove',e=>hoverCard(card,e));
    cv0.addEventListener('mouseleave',hideTip);
    cv0.addEventListener('click',e=>{
      const b=cv0.getBoundingClientRect();
      S.ph=tAt(b.width-PL-PR,e.clientX-b.left-PL);renderAll();});
  }
  el.querySelectorAll('.frames figure').forEach(f=>f.addEventListener('click',()=>{
    el.querySelectorAll('.frames figure').forEach(o=>o.classList.remove('sel'));
    f.classList.add('sel');S.ph=parseFloat(f.dataset.t);renderAll();}));
  cards.push(card);
}

const plist=document.getElementById('plist');
plist.innerHTML=DATA.problems.map((p,i)=>
  '<button class="p lv-'+p.level+'" data-i="'+i+'">'+
  '<span class="pt">'+(p.t==null?'全局':'+'+p.t.toFixed(2)+'s')+' · '+p.stream+'</span><br>'+
  '<span class="pm">'+p.msg+'</span></button>').join('') || '<div class="note">无</div>';
plist.querySelectorAll('.p').forEach(b=>b.addEventListener('click',()=>{
  const p=DATA.problems[+b.dataset.i];
  if(p.t!=null){
    const pad=Math.max(1,(p.dur||0));
    setView(p.t-pad,p.t+(p.dur||0)+pad);S.ph=p.t;
  }
  const card=cards.find(c=>c.st.name===p.stream);
  if(card)card.el.scrollIntoView({behavior:'smooth',block:'center'});
  plist.querySelectorAll('.p').forEach(o=>o.classList.remove('flash'));
  b.classList.add('flash');renderAll();
}));

/* ---------- master interaction: drag to pan, wheel to zoom ---------- */
let drag=null;
master.addEventListener('pointerdown',e=>{
  const b=master.getBoundingClientRect();
  drag={x:e.clientX,x0:S.x0,x1:S.x1,w:b.width-PL-PR,moved:false};
  master.setPointerCapture(e.pointerId);
});
master.addEventListener('pointermove',e=>{
  if(!drag)return;
  const dt=(e.clientX-drag.x)/drag.w*(drag.x1-drag.x0);
  if(Math.abs(e.clientX-drag.x)>2)drag.moved=true;
  setView(drag.x0-dt,drag.x1-dt);
});
master.addEventListener('pointerup',e=>{
  if(drag&&!drag.moved){
    const b=master.getBoundingClientRect();
    S.ph=tAt(b.width-PL-PR,e.clientX-b.left-PL);renderAll();
  }
  drag=null;
});
master.addEventListener('wheel',e=>{
  e.preventDefault();
  const b=master.getBoundingClientRect(), w=b.width-PL-PR;
  const t=tAt(w,e.clientX-b.left-PL), k=e.deltaY>0?1.3:1/1.3;
  setView(t-(t-S.x0)*k, t+(S.x1-t)*k);
},{passive:false});

document.getElementById('apply').onclick=()=>{
  const a=parseFloat(ra.value),b=parseFloat(rb.value);
  if(isFinite(a)&&isFinite(b)&&b>a)setView(a,b);};
document.getElementById('all').onclick=()=>setView(SPAN[0],SPAN[1]);
document.getElementById('win').onclick=()=>setView(DATA.view[0],DATA.view[1]);
document.getElementById('theme').onclick=()=>{
  const dark=document.documentElement.getAttribute('data-theme')==='dark';
  document.documentElement.setAttribute('data-theme',dark?'light':'dark');
  renderAll();                                   // canvases do not restyle themselves
};
addEventListener('resize',()=>renderAll());

function renderAll(){
  renderMaster();
  for(const c of cards)if(c.cv)try{renderCard(c);}catch(err){console.error(c.st.name,err);}
  document.getElementById('ph').textContent='播放头 +'+S.ph.toFixed(3)+'s';
}
setView(DATA.view[0],DATA.view[1]);
</script></body></html>
"""
