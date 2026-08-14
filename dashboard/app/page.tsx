"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useMemo, useRef, useState } from "react";

type Series = { label: string; color: string; values: number[]; dashed?: boolean };

function Chart({ series, labels, ariaLabel }: { series: Series[]; labels?: string[]; ariaLabel: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const box = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, box.width * ratio);
      canvas.height = Math.max(1, box.height * ratio);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(ratio, ratio);
      const w = box.width, h = box.height, left = 42, right = 18, top = 22, bottom = 34;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(234,239,224,.14)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = top + ((h - top - bottom) * i) / 4;
        ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
      }
      if (labels) {
        ctx.fillStyle = "rgba(234,239,224,.46)";
        ctx.font = "10px Arial";
        labels.forEach((label, i) => {
          const x = left + ((w - left - right) * i) / Math.max(1, labels.length - 1);
          ctx.fillText(label, x - 7, h - 10);
        });
      }
      series.forEach((line) => {
        const finiteValues = series.flatMap((s) => s.values).filter(Number.isFinite);
        const max = Math.max(1, ...finiteValues);
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.label === "Observed" ? 3 : 2;
        ctx.setLineDash(line.dashed ? [6, 7] : []);
        ctx.beginPath();
        let drawing = false;
        line.values.forEach((value, i) => {
          if (!Number.isFinite(value)) { drawing = false; return; }
          const x = left + ((w - left - right) * i) / Math.max(1, line.values.length - 1);
          const y = h - bottom - (value / max) * (h - top - bottom);
          if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [labels, series]);
  return <canvas className="chartCanvas" ref={canvasRef} role="img" aria-label={ariaLabel} />;
}

const basePV = [0,0,0,0,0,0,.3,1.8,4.1,6.8,8.4,9.2,9.6,9.1,7.8,6.1,3.9,1.7,.4,0,0,0,0,0];
const weather = [0,0,0,0,0,0,.2,1.2,3.2,5.8,7.8,8.8,9.5,9.4,8.2,6.4,4.4,2.1,.5,0,0,0,0,0];
const forecastClear = [0,0,0,0,0,0,.2,1.6,3.9,6.5,8.1,9,9.3,8.8,7.6,5.9,3.7,1.6,.3,0,0,0,0,0];
const forecastCloudy = [0,0,0,0,0,0,.2,1.3,3.4,5.1,4.3,6.8,7.5,5.8,6.4,4.7,3.1,1.3,.2,0,0,0,0,0];
const observedCloudy = [0,0,0,0,0,0,.2,1.1,3.6,5.5,3.5,6.2,7.9,5.1,6.9,4.5,2.8,1.2,.2,0,0,0,0,0];

const resultFigures = [
  { src:"/results/pv_v4_fig_results.png", label:"Headline results", note:"Combined model comparison and representative forecast evidence." },
  { src:"/results/pv_v4_fig_model_comparison.png", label:"Model comparison", note:"Measured performance across the evaluated model family." },
  { src:"/results/pv_v4_fig_representative_week.png", label:"Representative week", note:"Saved out-of-fold forecasts against observed PV." },
  { src:"/results/pv_v4_fig_app_alignment.png", label:"Alignment", note:"Empirical sensitivity to −1 h, 0 h and +1 h alignment settings." },
  { src:"/results/pv_v4_fig_app_clearness_sensitivity.png", label:"Physics cap", note:"Measured sensitivity to clearness-index clipping policy." },
  { src:"/results/pv_v4_fig_app_accuracy_cost.png", label:"Accuracy vs cost", note:"Forecast skill against measured computational cost." },
  { src:"/results/pv_v4_fig_app_missingness.png", label:"Missingness", note:"Audited missing observations across the study period." },
];

const evidenceCards = [
  { no:"01", title:"Missing is not zero", copy:"A separate availability channel tells the CNN what was observed. Missing targets receive zero loss weight and never enter evaluation." },
  { no:"02", title:"Time has to agree", copy:"The −1 hour shift is supported by empirical diagnostics and compared against 0 and +1 hour settings." },
  { no:"03", title:"Weather has an information date", copy:"Historical archive or reanalysis weather creates a retrospective upper bound, not expected live-deployment accuracy." },
  { no:"04", title:"Physics needs guardrails", copy:"Clearness clipping is centralised and tested at 1.2, 1.5, 2.0 and no upper cap." },
  { no:"05", title:"Complexity must earn its place", copy:"GBM and stackers are compared on error, training time and inference cost—not accuracy alone." },
  { no:"06", title:"Evaluation changes the answer", copy:"Random day-fold and rolling-origin evidence are presented together rather than collapsed into one score." },
  { no:"07", title:"One site is one case study", copy:"FEVER, one annual cycle and assumed geometry define the boundary of the claim." },
];

export default function Home() {
  const [openingAnswer, setOpeningAnswer] = useState<"model"|"pipeline"|null>(null);
  const [missingRate, setMissingRate] = useState(9);
  const [maskOn, setMaskOn] = useState(true);
  const [shift, setShift] = useState(-1);
  const [weatherScenario, setWeatherScenario] = useState<"clear"|"cloudy">("cloudy");
  const [protocol, setProtocol] = useState<"random"|"rolling">("random");
  const [priority, setPriority] = useState(52);
  const [evidenceIndex, setEvidenceIndex] = useState(0);
  const [storyStep, setStoryStep] = useState(0);
  const [resultFigure, setResultFigure] = useState(0);

  const missingSeries = useMemo(() => {
    return basePV.map((v,i) => ((i * 17 + 5) % 100 < missingRate ? (maskOn ? Number.NaN : 0) : v));
  }, [missingRate, maskOn]);
  const missingFlags = useMemo(() => basePV.map((_,i) => (i * 17 + 5) % 100 < missingRate), [missingRate]);
  const [cap, setCap] = useState<"1.2"|"1.5"|"2.0"|"none">("1.5");
  const rawKt = [0.12,.25,.42,.61,.79,.96,1.08,1.22,1.38,1.58,1.86,2.2];
  const capValue = cap === "none" ? Number.POSITIVE_INFINITY : Number(cap);
  const clippedKt = rawKt.map(v => Math.min(v, capValue));
  const shiftWeather = useMemo(() => weather.map((_,i) => weather[Math.max(0,Math.min(weather.length-1,i-shift))]), [shift]);
  const alignmentScore = shift === -1 ? 0.601 : shift === 0 ? 0.420 : 0.082;
  const observed = weatherScenario === "clear" ? basePV : observedCloudy;
  const predicted = weatherScenario === "clear" ? forecastClear : forecastCloudy;
  const recommendation = priority < 55 ? "GBM" : "ENSEMBLE";
  const chapterNames = ["The question","The setting","The hidden errors","The turning point","The evidence room","The decision","The invitation"];

  useEffect(() => {
    const ids = ["question","setting","labs","turning","evidence-room","decision","invitation"];
    const update = () => {
      let current = 0;
      ids.forEach((id,i) => { const box = document.getElementById(id)?.getBoundingClientRect(); if (box && box.top < innerHeight*.58) current = i; });
      setStoryStep(current);
    };
    update(); addEventListener("scroll",update,{passive:true}); return () => removeEventListener("scroll",update);
  }, []);

  const jump = (id:string) => document.getElementById(id)?.scrollIntoView({behavior:"smooth"});

  return <main>
    <aside className="storyRail" aria-label="Story progress"><div className="railBrand"><i/> PV</div><div className="railTrack"><span style={{height:`${(storyStep/6)*100}%`}}/></div><div className="railChapter"><b>0{storyStep+1}</b><span>{chapterNames[storyStep]}</span></div><button onClick={() => document.documentElement.requestFullscreen?.()} aria-label="Enter fullscreen">↗</button></aside>

    <section className="cinema opening" id="question">
      <div className="ambientSun"/><div className="gridHorizon"/>
      <div className="openingContent"><div className="officialRibbon"><a className="ccaisMark" href="https://ccais.ac.uk/" target="_blank" rel="noreferrer" aria-label="Visit the UK AI Conference website"><img src="/partners/ccais-logo.svg" alt="CCAIS"/></a><i/><a className="feverMark" href="https://www.fever-ev.ac.uk/" target="_blank" rel="noreferrer" aria-label="Visit the FEVER project website"><img src="/partners/fever-logo.png" alt="FEVER"/></a></div><div className="conferencePlaque"><strong>THE FOURTH UK AI CONFERENCE 2026</strong><span>29–30 SEPTEMBER · NOTTINGHAM, UK · HILTON NOTTINGHAM</span></div><p className="overline">Before we talk about algorithms...</p><h1>What if the biggest error in an AI forecast happens <em>before</em> the model sees a single example?</h1><p className="prompt">Where would you invest first?</p><div className="openingChoices"><button className={openingAnswer === "model" ? "selected" : ""} onClick={() => setOpeningAnswer("model")}><span>A</span>A more powerful model</button><button className={openingAnswer === "pipeline" ? "selected" : ""} onClick={() => setOpeningAnswer("pipeline")}><span>B</span>A more trustworthy pipeline</button></div>{openingAnswer && <div className="reveal"><span>{openingAnswer === "model" ? "A reasonable instinct." : "That is our hypothesis."}</span><strong>But the data had three hidden traps.</strong><button onClick={() => jump("setting")}>Enter the story ↓</button></div>}</div>
    </section>

    <section className="cinema setting" id="setting">
      <header className="chapterHeader"><span>01 / THE SETTING</span><h2>A new solar site.<br/>One year to learn.</h2></header>
      <div className="humanFrame"><div className="operatorCard"><span className="liveDot"/> DAY-AHEAD DECISION</div><p>An EV-charging operator needs tomorrow&apos;s solar profile before tomorrow arrives—not merely a good average score.</p><div className="decisionClock"><span>Today<br/><b>17:00</b></span><i/><span>Tomorrow<br/><b>00–24h</b></span></div></div><div className="officialContext"><article><img src="/partners/fever-logo.png" alt="FEVER project"/><div><span>THE APPLICATION</span><p>FEVER is developing grid-independent, renewably powered EV charging. Better day-ahead PV information supports the decisions behind that ambition.</p><a href="https://www.fever-ev.ac.uk/" target="_blank" rel="noreferrer">Explore the FEVER project ↗</a></div></article><article className="conferenceContext"><img src="/partners/ccais-logo.svg" alt="CCAIS"/><div><span>THE CONVERSATION</span><p>Presented as trustworthy, physics-aware machine learning for real-world energy decision support at the Fourth UK AI Conference 2026.</p><a href="https://ccais.ac.uk/" target="_blank" rel="noreferrer">Visit the conference website ↗</a></div></article></div>
      <div className="fiveWs"><article><span>WHO</span><strong>FEVER site operators</strong></article><article><span>WHAT</span><strong>24 hourly PV forecasts</strong></article><article><span>WHERE</span><strong>Southampton, UK</strong></article><article><span>WHEN</span><strong>One day ahead</strong></article><article className="why"><span>WHY</span><strong>Better-informed energy decisions</strong></article></div>
      <div className="antagonist"><span>THE ANTAGONIST</span><h3>Not a person. A chain of invisible assumptions.</h3><p>Missing measurements. Misaligned clocks. Weather that would not yet exist in deployment. Each can make a sophisticated model look better—or worse—than it really is.</p></div>
    </section>

    <section className="cinema labs" id="labs">
      <header className="chapterHeader"><span>02 / RISING ACTION</span><h2>Find the hidden errors.</h2><p>Change the controls. Watch how a seemingly small pipeline choice changes what the model is allowed to learn.</p></header>
      <article className="labCard missingLab"><div className="labCopy"><p className="labNo">LAB 01 · AUDIT: 9.25% MISSING</p><h3>When is zero not zero?</h3><p>A missing reading can look identical to night-time generation unless availability travels with the value.</p><label>Missing observations <output>{missingRate}%</output><input type="range" min="0" max="25" value={missingRate} onChange={e => setMissingRate(+e.target.value)}/></label><button className={maskOn ? "toggle on" : "toggle"} onClick={() => setMaskOn(!maskOn)}><i/>{maskOn ? "Availability mask ON" : "Availability mask OFF"}</button></div><div className="labVisual"><div className="chartTitle"><span>Illustrative daily PV profile</span><div><i className="legend observed"/>Expected profile <i className="legend missing"/>{maskOn ? "Masked placeholder" : "False zero"}</div></div><Chart ariaLabel="Illustrative PV curve showing the effect of missing observations" labels={["00","06","12","18","23"]} series={[{label:"Observed",color:"#f6c344",values:basePV},{label:"Missing",color:maskOn?"rgba(156,224,189,.35)":"#f27965",values:missingSeries,dashed:true}]}/><div className={`availabilityStrip ${maskOn ? "maskMode" : "zeroMode"}`} aria-label="Hourly data availability">{missingFlags.map((missing,i)=><i key={i} className={missing ? "missing" : ""} title={`${String(i).padStart(2,"0")}:00 — ${missing ? (maskOn ? "unavailable" : "written as zero") : "observed"}`}/>)}</div><div className="availabilityCaption"><span>00:00</span><strong>{maskOn ? "Gaps remain visibly unavailable" : "Missing hours collapse into false zeros"}</strong><span>23:00</span></div><div className={maskOn ? "labVerdict safe" : "labVerdict danger"}>{maskOn ? "The line now breaks where readings are absent. The availability channel tells the model: unknown—not zero." : "The line falls to zero at missing hours, teaching the model a physically false observation."}</div></div></article>

      <article className="labCard alignLab"><div className="labCopy"><p className="labNo">LAB 02</p><h3>Two clocks. One physical event.</h3><p>Move the PV timestamp relative to weather and see the relationship strengthen or break.</p><div className="shiftControl">{[-1,0,1].map(v => <button key={v} className={shift===v?"active":""} onClick={() => setShift(v)}>{v>0?"+":""}{v} h</button>)}</div><div className="scoreDial"><strong>{alignmentScore.toFixed(3)}</strong><span>empirical physics-proxy R²</span></div></div><div className="labVisual"><div className="chartTitle"><span>PV output vs irradiance shape</span><div><i className="legend observed"/>PV <i className="legend weather"/>Weather</div></div><Chart ariaLabel="Illustrative alignment between PV and weather curves" labels={["00","06","12","18","23"]} series={[{label:"Observed",color:"#f6c344",values:basePV},{label:"Weather",color:"#8fc9ef",values:shiftWeather,dashed:true}]}/><div className="labVerdict neutral">The −1 h setting is an empirical alignment choice—not verified inverter metadata.</div></div></article>
    </section>

    <section className="cinema turning" id="turning">
      <header className="chapterHeader"><span>03 / THE TURNING POINT</span><h2>The model did not change.<br/>The question did.</h2></header>
      <div className="protocolSwitch"><button className={protocol==="random"?"active":""} onClick={() => setProtocol("random")}><span>01</span>Random day-fold</button><button className={protocol==="rolling"?"active":""} onClick={() => setProtocol("rolling")}><span>02</span>Rolling-origin</button></div>
      <div className="protocolStage"><div className="protocolNarrative"><span>{protocol === "random" ? "THE COMPARISON QUESTION" : "THE DEPLOYMENT QUESTION"}</span><h3>{protocol === "random" ? "Can the model generalise across held-out days within this year?" : "Can the model forecast the future using only what was available before it?"}</h3><p>{protocol === "random" ? "Seasons are mixed across folds. Useful for broad model comparison, but less faithful to deployment." : "Time order is preserved. Distribution shift becomes visible, making this the harder and more operationally relevant test."}</p>{protocol === "random" ? <div className="reportedResult"><strong>0.861</strong><span>NNLS stack · all-hours R²<br/>verified private run</span></div> : <div className="reportedResult"><strong>0.726</strong><span>ridge stack · all-hours R²<br/>verified rolling-origin run</span></div>}</div><div className={`timeline ${protocol}`}><div className="season spring">SPR</div><div className="season summer">SUM</div><div className="season autumn">AUT</div><div className="season winter">WIN</div>{protocol === "random" ? <div className="foldDots">{Array.from({length:20},(_,i)=><i key={i} className={i%4===0?"test":"train"}/>)}</div> : <div className="originBlocks"><i/><i/><i/><i/></div>}</div></div>
      <div className="weatherLab"><div><p className="labNo">EXPLORE A DAY</p><h3>Weather changes the shape of the challenge.</h3><div className="scenarioButtons"><button className={weatherScenario==="clear"?"active":""} onClick={()=>setWeatherScenario("clear")}>Clear day</button><button className={weatherScenario==="cloudy"?"active":""} onClick={()=>setWeatherScenario("cloudy")}>Variable cloud</button></div><p className="dataNote">Profile shapes remain explanatory; measured out-of-fold evidence appears in the Evidence Room.</p></div><div><Chart ariaLabel="Conceptual observed and forecast photovoltaic profiles" labels={["00","06","12","18","23"]} series={[{label:"Observed",color:"#f6c344",values:observed},{label:"Forecast",color:"#9ce0bd",values:predicted,dashed:true}]}/></div></div>
    </section>

    <section className="cinema evidenceRoom" id="evidence-room">
      <header className="chapterHeader"><span>04 / THE EVIDENCE ROOM</span><h2>What exactly was tested?</h2><p>A trustworthy result is more than one score. This map exposes the models, protocols, physics checks and reproducibility machinery behind the claim.</p></header>
      <div className="verifiedBanner"><span>PRIVATE RUN · 06 AUG 2026</span><strong>12 / 12 stages succeeded</strong><p>391-day span · 8,519 raw PV rows · 9.25% missing hours · generated outputs checksum-inventoried</p></div>
      <div className="resultGallery"><div className="figureTabs">{resultFigures.map((figure,i)=><button key={figure.src} className={resultFigure===i?"active":""} onClick={()=>setResultFigure(i)}><span>0{i+1}</span>{figure.label}</button>)}</div><figure><img src={resultFigures[resultFigure].src} alt={resultFigures[resultFigure].label}/><figcaption><strong>{resultFigures[resultFigure].label}</strong><span>{resultFigures[resultFigure].note}</span></figcaption></figure></div>      <div className="conferenceFit"><article><span>CONFERENCE LENS 01</span><strong>Machine learning</strong><p>Five complementary learners and four ensemble rules test whether model diversity adds useful forecast skill.</p></article><article><span>CONFERENCE LENS 02</span><strong>Trustworthy AI</strong><p>Availability masks, leakage-safe folds and explicit uncertainty make hidden assumptions inspectable.</p></article><article><span>CONFERENCE LENS 03</span><strong>Real-world implementation</strong><p>Rolling-origin evaluation, runtime cost and weather provenance connect retrospective evidence to deployment.</p></article></div>
      <div className="experimentGrid"><article><strong>5</strong><span>BASE LEARNERS</span><p>Ridge · GBM · clearness GBM · per-hour GBM · 2D CNN</p></article><article><strong>4</strong><span>FUSION RULES</span><p>Mean · median · non-negative least squares · stacked GBM</p></article><article><strong>2</strong><span>PROTOCOLS</span><p>Random day-fold and rolling-origin answer different questions.</p></article><article><strong>3</strong><span>TIME SHIFTS</span><p>−1 h · 0 h · +1 h are tested rather than assumed.</p></article><article><strong>4</strong><span>PHYSICS CAPS</span><p>1.2 · 1.5 · 2.0 · uncapped sensitivity settings.</p></article><article><strong>1</strong><span>RUN MANIFEST</span><p>Configuration, environment, inputs, checksums, timings and outputs.</p></article></div>
      <div className="clearnessLab"><div><p className="labNo">PHYSICS EXPLAINER</p><h3>Why clip the clearness ratio?</h3><p>Near dawn, dusk or low irradiance, dividing PV by a small physical baseline can create extreme targets. Explore how a cap constrains that tail without pretending this conceptual curve is an experimental result.</p><div className="capButtons">{(["1.2","1.5","2.0","none"] as const).map(v=><button key={v} className={cap===v?"active":""} onClick={()=>setCap(v)}>{v === "none" ? "No cap" : `Cap ${v}`}</button>)}</div><small>Use this control to understand the rule; see the verified Physics cap figure above for measured results.</small></div><div><div className="chartTitle"><span>Raw versus constrained clearness ratio</span><div><i className="legend missing"/>Raw <i className="legend weather"/>After rule</div></div><Chart ariaLabel="Conceptual illustration of clearness ratio clipping" labels={["low","","","","","high"]} series={[{label:"Raw",color:"#f17d68",values:rawKt,dashed:true},{label:"Constrained",color:"#83cbed",values:clippedKt}]}/></div></div>
      <div className="architecture"><div><span>MODEL INPUT</span><strong>24 × 9 daily tensor</strong><p>Normalized PV · past/target indicator · availability mask · six weather channels</p></div><i>→</i><div><span>LEARNING</span><strong>Five model perspectives</strong><p>Linear, nonlinear, physics-targeted, hour-specialised and spatiotemporal</p></div><i>→</i><div><span>DECISION</span><strong>Four fusion strategies</strong><p>Compare predictive skill with training, inference and maintenance cost</p></div></div>
    </section>
    <section className="cinema decision" id="decision">
      <header className="chapterHeader"><span>05 / RESOLUTION</span><h2>Accuracy is not the only decision.</h2><p>Move the priority slider. The technically “best” choice depends on what the deployment values.</p></header>
      <div className="decisionTool"><div className="priorityLabels"><span>Simplicity & speed</span><span>Maximum forecast skill</span></div><input type="range" min="0" max="100" value={priority} onChange={e=>setPriority(+e.target.value)}/><div className="recommendation"><span>FOR THIS PRIORITY</span><strong>{recommendation}</strong><p>{recommendation === "GBM" ? "A lean deployment choice when maintainability and speed dominate." : "Worth the extra training and maintenance only when incremental error reduction has sufficient value."}</p></div><div className="costMap"><div className="mapAxis y">forecast skill ↑</div><div className="mapAxis x">system cost →</div><button className={recommendation==="GBM"?"chosen gbm":"gbm"}>GBM<small>lean</small></button><button className={recommendation==="ENSEMBLE"?"chosen ensemble":"ensemble"}>STACK<small>skill</small></button></div></div>
      <div className="evidenceExplorer"><div className="evidenceTabs">{evidenceCards.map((x,i)=><button key={x.no} className={evidenceIndex===i?"active":""} onClick={()=>setEvidenceIndex(i)}><span>{x.no}</span>{x.title}</button>)}</div><article><span>{evidenceCards[evidenceIndex].no} / EVIDENCE</span><h3>{evidenceCards[evidenceIndex].title}</h3><p>{evidenceCards[evidenceIndex].copy}</p><div className="evidenceStamp">TESTABLE · TRACEABLE · REPRODUCIBLE</div></article></div>
    </section>

    <section className="cinema invitation" id="invitation">
      <div className="finalQuestion"><span>06 / INVITATION · WHAT SHOULD WE TRUST?</span><h2>Not a leaderboard.<br/>A chain of evidence.</h2></div>
      <div className="dmmi"><article><span>DATA</span><strong>One FEVER site, imperfect observations</strong></article><article><span>MEANING</span><strong>Pipeline choices can distort apparent skill</strong></article><article><span>DECISION</span><strong>Evaluate leakage, time, physics and cost together</strong></article><article><span>IMPACT</span><strong>Forecasts we can explain—and limitations we can defend</strong></article></div>
      <div className="knownGrid"><article><span>WHAT WE KNOW</span><p>The private 12-stage run succeeded: random-fold R² 0.861 and rolling-origin R² 0.726 for the best all-hours ensembles.</p></article><article><span>WHAT WE DO NOT KNOW</span><p>External-site transfer and live forecast-vintage performance remain untested; this run uses one site and retrospective historical weather.</p></article><article><span>WHAT WE DO NEXT</span><p>Acquire comparable external-site measurements and archived day-ahead forecasts with issue timestamps, then rerun the same protocol.</p></article></div>
      <div className="callToAction"><p>Trustworthy environmental AI is built through the whole pipeline—</p><strong>not just the final model.</strong><button onClick={()=>jump("question")}>Replay the story ↑</button></div>
      <footer><span>THE FOURTH UK AI CONFERENCE 2026</span><span>No confidential measurements embedded</span><a href="https://michaeltheanalyst.github.io/" target="_blank" rel="noreferrer">Built by Masood Nazari ↗</a></footer>
    </section>
  </main>;
}








