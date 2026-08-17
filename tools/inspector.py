#!/usr/bin/env python3
"""Build a single self-contained HTML page from the run files.

    python3 tools/inspector.py > inspector.html

One file. No server, no port, no process, no CDN, no external asset. It opens
from `file://` and it can be dropped on any static host. That is deliberate:
the submission needs a link that is still reachable weeks later, and a page
with a process behind it is a page that can be down when a judge opens it.

What it shows is the argument of the project, made clickable. Every question
carries the evidence that was put in front of the model, the answer or the
refusal, the judge's verdict, and the retrieval report -- how many candidates
the gate produced, how many survived the as-of filter, how many were admitted
by the tolerance. A reader can pick any false refusal and see for themselves
that the gold session was sitting in the evidence when the answerer declined.

Gold sessions are marked in the evidence list, so "the evidence was there and
it refused anyway" is visible rather than asserted.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os

STATE = os.environ.get("DOCKET_STATE", os.path.expanduser("~/docket/state"))


def load(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("question_id"):
                rows[row["question_id"]] = row
    return rows


def when(ts):
    if ts is None:
        return ""
    try:
        return datetime.datetime.fromtimestamp(
            int(ts), datetime.timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return str(ts)


def build(answers, judged):
    """One record per question, small enough to embed 500 of them."""
    out = []
    for qid, a in answers.items():
        j = judged.get(qid, {})
        gold = set(a.get("gold_sessions") or [])
        ev = []
        for e in (a.get("evidence") or []):
            sess = e.get("session")
            ev.append({
                "n": e.get("n"),
                "t": e.get("triple"),
                "s": sess,
                "g": 1 if sess in gold else 0,
                "d": when(e.get("said_at")),
                "m": list((e.get("matched_terms") or {}))[:6],
                "u": e.get("used"),
            })
        r = a.get("retrieval") or {}
        out.append({
            "id": qid,
            "c": a.get("question_type") or "unknown",
            "abs": 1 if a.get("is_abstention") else 0,
            "q": a.get("question") or "",
            "at": when(a.get("asked_at")),
            "gold": "" if a.get("is_abstention") else str(a.get("gold") or ""),
            "ans": a.get("answer"),
            "st": a.get("status"),
            "why": a.get("reason") or "",
            "v": j.get("verdict") or "unjudged",
            "k": j.get("kind") or "",
            "jw": j.get("why") or "",
            "jm": j.get("judge_model") or "",
            "ev": ev,
            "r": {k: r.get(k) for k in
                  ("hits", "candidates", "kept", "dropped_future",
                   "dropped_missing", "admitted_by_tolerance", "mode")
                  if r.get(k) is not None},
        })
    out.sort(key=lambda x: (x["c"], x["id"]))
    return out


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>docket -- the register</title>
<style>
:root{--bg:#12100e;--card:#1b1815;--line:#2e2924;--ink:#efeae1;--dim:#a49c90;
--ok:#8fbf8f;--bad:#cf8071;--warn:#d6a85f;--cool:#8fa8c9;--gold:#c99b5f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;padding:28px 20px 80px}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px;letter-spacing:.02em}
h1 b{color:var(--gold)}
.sub{color:var(--dim);margin:0 0 22px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 22px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:9px 13px;min-width:132px}
.stat span{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;
letter-spacing:.08em}
.stat b{font-size:19px;font-weight:600}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
button{background:var(--card);color:var(--ink);border:1px solid var(--line);
border-radius:5px;padding:6px 11px;cursor:pointer;font:inherit;font-size:13px}
button.on{border-color:var(--gold);color:var(--gold)}
.row{background:var(--card);border:1px solid var(--line);border-left-width:3px;
border-radius:5px;margin:0 0 7px;padding:10px 13px;cursor:pointer}
.row:hover{border-color:var(--dim)}
.row .q{display:block}
.row .meta{color:var(--dim);font-size:12px;margin-top:3px}
.correct{border-left-color:var(--ok)} .incorrect{border-left-color:var(--bad)}
.false_refusal{border-left-color:var(--warn)}
.retrieval_miss{border-left-color:var(--cool)}
.unjudged{border-left-color:var(--line)}
.tag{font-size:11px;padding:1px 6px;border-radius:3px;border:1px solid var(--line);
color:var(--dim);margin-right:6px}
.detail{display:none;margin-top:11px;border-top:1px solid var(--line);padding-top:11px}
.detail.open{display:block}
.kv{margin:0 0 9px}
.kv i{color:var(--dim);font-style:normal;display:inline-block;min-width:112px}
.ev{border-top:1px dashed var(--line);padding:7px 0;font-size:13px}
.ev.gold{background:rgba(201,155,95,.09)}
.ev .g{color:var(--gold)}
.ev .m{color:var(--dim);font-size:11px}
.rep{color:var(--dim);font-size:12px;margin-top:9px}
.none{color:var(--dim);padding:22px;text-align:center}
.count{color:var(--dim);font-size:12px;margin:0 0 10px}
.count b{color:var(--gold);font-weight:600}
</style>
<div class=wrap>
<h1>docket <b>&mdash; the register</b></h1>
<p class=sub>Every question, the evidence put in front of the model, and what it
did with it. Gold sessions are marked, so a refusal with the answer present is
visible rather than asserted.</p>
<div class=stats id=stats></div>
<div class=bar id=cats></div>
<div class=bar id=kinds></div>
<p class=count id=count></p>
<div id=list></div>
</div>
<script id=data type="application/json">__DATA__</script>
<script>
var ROWS = JSON.parse(document.getElementById('data').textContent);
var cat = 'all', kind = 'all';
function esc(s){var d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML}
function kindOf(r){ if(r.abs) return r.v==='correct'?'abstained':'hallucinated';
  if(r.k==='false_refusal'||r.k==='retrieval_miss') return r.k;
  return r.v; }
function stats(){
  var real=ROWS.filter(function(r){return !r.abs});
  var g=real.filter(function(r){return r.v==='correct'||r.v==='incorrect'});
  var c=real.filter(function(r){return r.v==='correct'}).length;
  var fr=real.filter(function(r){return r.k==='false_refusal'}).length;
  var rm=real.filter(function(r){return r.k==='retrieval_miss'}).length;
  var ab=ROWS.filter(function(r){return r.abs});
  var abok=ab.filter(function(r){return r.v==='correct'}).length;
  var ansd=g.length-fr-rm;
  return [['accuracy',(c/g.length).toFixed(4)],
          ['correct',c+' / '+g.length],
          ['precision when answering',(c/ansd).toFixed(4)],
          ['refused, evidence present',fr],
          ['refused, nothing retrieved',rm],
          ['abstention',abok+' / '+ab.length]];
}
document.getElementById('stats').innerHTML = stats().map(function(s){
  return '<div class=stat><span>'+s[0]+'</span><b>'+s[1]+'</b></div>'}).join('');
function bar(el,vals,cur,set){
  document.getElementById(el).innerHTML = vals.map(function(v){
    return '<button data-v="'+v+'" class="'+(v===cur?'on':'')+'">'+v+'</button>'}).join('');
  document.getElementById(el).onclick=function(e){
    if(e.target.tagName!=='BUTTON')return; set(e.target.getAttribute('data-v')); draw();}
}
function draw(){
  bar('cats',['all'].concat(ROWS.map(function(r){return r.c}).filter(function(v,i,a){
    return a.indexOf(v)===i}).sort()),cat,function(v){cat=v});
  bar('kinds',['all','correct','incorrect','false_refusal','retrieval_miss',
    'abstained','hallucinated'],kind,function(v){kind=v});
  var rows=ROWS.filter(function(r){
    return (cat==='all'||r.c===cat) && (kind==='all'||kindOf(r)===kind)});
  var nabs=rows.filter(function(r){return r.abs}).length;
  document.getElementById('count').innerHTML =
    'showing <b>'+rows.length+'</b> of '+ROWS.length+' questions'+
    (nabs?' &middot; '+nabs+' of them unanswerable by construction':'');
  var L=document.getElementById('list');
  if(!rows.length){L.innerHTML='<p class=none>nothing matches that filter</p>';return}
  L.innerHTML = rows.map(function(r,i){
    var ev = r.ev.length ? r.ev.map(function(e){
      return '<div class="ev'+(e.g?' gold':'')+'">['+e.n+'] '+esc((e.t||[]).join(' '))+
        ' <span class="'+(e.g?'g':'m')+'">'+(e.g?'GOLD ':'')+esc(e.s||'')+'</span>'+
        ' <span class=m>'+esc(e.d)+(e.m.length?' &middot; '+esc(e.m.join(', ')):'')+
        '</span></div>'}).join('')
      : '<div class=ev><span class=m>no evidence was retrieved</span></div>';
    var rep = Object.keys(r.r).map(function(k){return k+'='+r.r[k]}).join('  ');
    return '<div class="row '+kindOf(r)+'" data-i="'+i+'">'+
      '<span class=q>'+esc(r.q)+'</span>'+
      '<span class=meta><span class=tag>'+r.c+'</span>'+
      '<span class=tag>'+kindOf(r)+'</span>asked '+esc(r.at)+'</span>'+
      '<div class=detail>'+
        (r.abs?'':'<p class=kv><i>reference</i>'+esc(r.gold)+'</p>')+
        '<p class=kv><i>produced</i>'+(r.ans==null?'<span class=m>refused</span>':esc(r.ans))+'</p>'+
        (r.why?'<p class=kv><i>reason</i>'+esc(r.why)+'</p>':'')+
        (r.jw?'<p class=kv><i>judge</i>'+esc(r.jw)+(r.jm?' <span class=m>('+esc(r.jm)+')</span>':'')+'</p>':'')+
        '<p class=kv><i>evidence</i></p>'+ev+
        '<p class=rep>'+esc(rep)+'</p>'+
      '</div></div>'}).join('');
  L.onclick=function(e){
    var row=e.target.closest('.row'); if(!row)return;
    row.querySelector('.detail').classList.toggle('open')}
}
draw();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=os.path.join(STATE, "answers-500.jsonl"))
    ap.add_argument("--judged", default=os.path.join(STATE, "judged-500.jsonl"))
    a = ap.parse_args()
    for p in (a.answers, a.judged):
        if not os.path.exists(p):
            raise SystemExit("missing file: " + p)

    rows = build(load(a.answers), load(a.judged))
    blob = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    # A "</script>" anywhere in the data would close the tag early. The
    # transcripts are user text and can contain anything.
    blob = blob.replace("</", "<\\/")
    print(PAGE.replace("__DATA__", blob))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
