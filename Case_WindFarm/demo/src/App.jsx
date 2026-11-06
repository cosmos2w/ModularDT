import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, ArrowUpRight, Box, ChevronDown, CircleHelp, Crosshair, Expand, Layers3, LoaderCircle, Minus, MousePointer2, Move, Network, Pause, Play, Plus, RotateCcw, Settings2, Wind, X } from 'lucide-react';
import FlowScene from './FlowScene.jsx';
import ScientificPanels from './ScientificPanels.jsx';
import { FIELD_COLORS, getFieldRange, sampleVelocity, sliceSampleCoordinates } from './flowMath.js';

const INITIAL_VIEW = { mode: 'field', organizationAddress: null, field: 'speed', streamlines: true, slice: true, sliceAxis: 'z', slicePosition: 0.14, organization: false, exaggeration: 2.8, playing: true, tool: 'orbit' };
const RESOLUTIONS = { draft: [28, 20, 12], standard: [40, 28, 16], fine: [56, 32, 16] };
const copy = value => structuredClone(value);
const signature = scene => JSON.stringify(scene && [scene.id, scene.turbines, scene.domain, scene.wind]);
const format = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '—';
const formatError = value => Number.isFinite(Number(value)) ? Number(value).toPrecision(3) : '—';

async function getJSON(url, options) {
  const response = await fetch(url, options);
  const isJSON = response.headers.get('content-type')?.includes('application/json');
  const body = isJSON ? await response.json() : { detail: `Inference service returned ${response.status}. Retry the reconstruction.` };
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : body.message || `Request failed (${response.status})`);
  return body;
}

function IconButton({ title, active, onClick, children, className = '', ...props }) {
  return <button className={`icon-button ${active ? 'active' : ''} ${className}`} title={title} aria-label={title} aria-pressed={active} onClick={onClick} {...props}>{children}</button>;
}

function Range({ label, value, min, max, step = 1, onChange, unit = '', ...props }) {
  return <label className="range-control"><span>{label}<output>{Number(value).toFixed(step < 1 ? 1 : 0)}{unit && ` ${unit}`}</output></span><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} {...props}/></label>;
}

function ErrorReadout({ result, dirty, view }) {
  const errors = result?.errors;
  const displayedErrors = useMemo(() => {
    const grid = result?.grid;
    if (!grid?.reference_velocity) return null;
    const referenceGrid = { ...grid, velocity: grid.reference_velocity };
    const sum = [0, 0, 0, 0];
    let count = 0;
    const accumulate = (prediction, reference) => {
      if (!prediction || !reference) return;
      for (let i = 0; i < 3; i += 1) sum[i] += (prediction[i] - reference[i]) ** 2;
      sum[3] += (Math.hypot(...prediction) - Math.hypot(...reference)) ** 2;
      count += 1;
    };
    if (view.slice) {
      for (const point of sliceSampleCoordinates(grid, view.sliceAxis, view.slicePosition).coordinates) accumulate(sampleVelocity(grid, point), sampleVelocity(referenceGrid, point));
    } else {
      for (let i = 0; i < grid.velocity.length; i += 3) accumulate(grid.velocity.slice(i, i + 3), grid.reference_velocity.slice(i, i + 3));
    }
    return count ? sum.map(value => Math.sqrt(value / count)) : null;
  }, [result, view.slice, view.sliceAxis, view.slicePosition]);
  if (!result || dirty) return <span className="error-readout">Reference error unavailable</span>;
  if (displayedErrors) return <span className="error-readout">{view.slice ? 'Slice' : 'Volume'} RMSE |U| {formatError(displayedErrors[3])} / Ux {formatError(displayedErrors[0])} / Uy {formatError(displayedErrors[1])} / Uz {formatError(displayedErrors[2])} m/s</span>;
  if (!errors || errors.available === false) return <span className="error-readout">Reference error unavailable · edited scene</span>;
  if (errors.components) return <span className="error-readout">Volume RMSE {Object.entries(errors.components).map(([name, values]) => `${name} ${formatError(values.rmse_m_s)}`).join(' / ')} m/s</span>;
  const rmse = errors.rmse_m_s || errors.rmse || errors.fieldwise_rmse_m_s;
  const rel = errors.relative_l2 || errors.fieldwise_relative_l2;
  const values = rmse || rel;
  if (!values) return <span className="error-readout">Reference error unavailable</span>;
  const entries = Array.isArray(values) ? ['Ux', 'Uy', 'Uz'].map((name, i) => [name, values[i]]) : Object.entries(values);
  return <span className="error-readout">{rmse ? 'RMSE' : 'Relative L2'} {entries.map(([name, value]) => `${name} ${format(value)}`).join(' / ')}{rmse ? ' m/s' : ''}</span>;
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [scene, setScene] = useState(null);
  const [original, setOriginal] = useState(null);
  const [result, setResult] = useState(null);
  const [resultKey, setResultKey] = useState('');
  const [view, setView] = useState(INITIAL_VIEW);
  const [selected, setSelected] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [quality, setQuality] = useState('standard');
  const [busy, setBusy] = useState(false);
  const [loadingCase, setLoadingCase] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [cameraPreset, setCameraPreset] = useState('perspective');
  const [cameraRevision, setCameraRevision] = useState(0);
  const requestId = useRef(0);
  const caseRequestId = useRef(0);
  const reconstructionTimer = useRef(null);
  const sceneKey = signature(scene);
  const dirty = `${sceneKey}:${quality}` !== resultKey;
  const edited = !!scene && sceneKey !== signature(original);
  const visibleResult = dirty || dragging ? null : result;
  const chosen = scene?.turbines.find(turbine => turbine.id === selected);

  const updateView = (key, value) => setView(previous => ({ ...previous, [key]: value }));
  const changeMode = mode => setView(previous => {
    const ids = result?.organization?.selection?.selected_anchor_ids || [];
    return { ...previous, mode, organization: mode === 'organization', organizationAddress: previous.organizationAddress ?? ids[Math.floor(ids.length / 2)] ?? null };
  });
  const updateScene = useCallback(change => {
    requestId.current += 1;
    setError('');
    setScene(previous => typeof change === 'function' ? change(previous) : change);
  }, []);

  const loadCase = useCallback(async caseId => {
    const id = ++caseRequestId.current;
    requestId.current += 1;
    setLoadingCase(true); setError(''); setResult(null); setSelected(null);
    try {
      const next = await getJSON(`/api/cases/${encodeURIComponent(caseId)}`);
      if (id !== caseRequestId.current) return;
      setOriginal(copy(next)); setScene(next);
    } catch (failure) { if (id === caseRequestId.current) setError(failure.message); }
    finally { if (id === caseRequestId.current) setLoadingCase(false); }
  }, []);

  useEffect(() => {
    let live = true;
    getJSON('/api/meta').then(data => {
      if (!live) return;
      setMeta(data);
      loadCase(data.defaults?.case_id ?? data.cases?.[0]?.id);
    }).catch(failure => { if (live) setError(failure.message); });
    return () => { live = false; };
  }, [loadCase]);

  const reconstruct = useCallback(async () => {
    if (!scene) return;
    clearTimeout(reconstructionTimer.current);
    const id = ++requestId.current;
    const key = `${signature(scene)}:${quality}`;
    setBusy(true); setError('');
    try {
      const next = await getJSON('/api/reconstruct', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ case_id: scene.id, turbines: scene.turbines, domain: scene.domain, wind: scene.wind, resolution: RESOLUTIONS[quality] }) });
      if (id !== requestId.current) return;
      setResult(next); setResultKey(key);
    } catch (failure) { if (id === requestId.current) setError(failure.message); }
    finally { if (id === requestId.current) setBusy(false); }
  }, [scene, quality]);

  useEffect(() => {
    if (!scene || loadingCase) return;
    reconstructionTimer.current = setTimeout(reconstruct, 650);
    return () => clearTimeout(reconstructionTimer.current);
  }, [reconstruct, loadingCase]);

  const moveTurbine = useCallback((id, x, y) => {
    updateScene(previous => ({ ...previous, turbines: previous.turbines.map(turbine => turbine.id === id ? { ...turbine, x, y } : turbine) }));
  }, [updateScene]);

  const changeCount = delta => updateScene(previous => {
    if (delta < 0) return { ...previous, turbines: previous.turbines.slice(0, -1) };
    const center = previous.domain.min.map((value, axis) => (value + previous.domain.max[axis]) / 2);
    const ids = new Set(previous.turbines.map(turbine => turbine.id));
    let id = 0; while (ids.has(id) || ids.has(String(id))) id += 1;
    return { ...previous, turbines: [...previous.turbines, { id, x: center[0] + (previous.turbines.length % 3 - 1) * 1.2, y: center[1], z: previous.turbines[0]?.z ?? 0.875 }] };
  });

  const changeDomain = (axis, length) => {
    if (!Number.isFinite(length) || length <= 0) return;
    updateScene(previous => {
      const center = (previous.domain.min[axis] + previous.domain.max[axis]) / 2;
      const required = axis === 2 ? 1.5 : 2 * Math.max(...previous.turbines.map(turbine => Math.abs(turbine[['x', 'y'][axis]] - center))) + 0.1;
      length = Math.max(required, Math.min(axis === 2 ? 6.25 : 100, length));
      const min = [...previous.domain.min], max = [...previous.domain.max];
      min[axis] = axis === 2 ? 0 : center - length / 2;
      max[axis] = axis === 2 ? length : center + length / 2;
      return { ...previous, domain: { min, max } };
    });
  };

  const reset = () => {
    if (original) updateScene(copy(original));
    setSelected(null); setView(INITIAL_VIEW); setCameraPreset('perspective'); setCameraRevision(value => value + 1);
  };

  const fieldRange = useMemo(() => {
    const range = getFieldRange(visibleResult?.grid, view.field);
    return range ? [range.min, range.max] : [0, 12];
  }, [visibleResult, view.field]);

  const directions = meta?.controls?.wind_direction_deg?.options || [270, 285, 300];
  const status = error ? 'Attention needed' : loadingCase ? 'Loading scene' : busy || (dirty && scene) ? 'Reconstructing' : result ? 'Model ready' : 'Loading model';
  const epoch = meta?.checkpoint?.epoch ?? '—';
  const timing = result?.provenance?.inference_ms ?? (result?.provenance?.inference_seconds === undefined ? undefined : result.provenance.inference_seconds * 1000);

  return <div className={`app-shell ${view.mode === 'organization' ? 'organization-mode' : 'field-mode'}`}>
    <header className="topbar">
      <a className="brand" href="/" aria-label="HONF Wind Lab"><svg viewBox="0 0 36 36" aria-hidden="true"><path d="M18 15 18 2M18 15 5 26M18 15 31 26M18 18v15"/><circle cx="18" cy="15" r="2.5"/></svg><strong>HONF <span>/ Wind Lab</span></strong></a>
      <div className="topnav"><span className="current">Explore</span></div>
      <div className={`model-status ${error ? 'failed' : ''}`} role="status"><i className={busy || dirty ? 'pulsing' : ''}/><span>{status}</span></div>
      <button className="reset-button" onClick={reset} disabled={!scene}><RotateCcw size={15}/>Reset</button>
      <IconButton title="Model information" onClick={() => setInfo(true)}><CircleHelp size={19}/></IconButton>
    </header>

    <button className="mobile-panel-button" onClick={() => setPanelOpen(value => !value)} aria-expanded={panelOpen}><Settings2 size={17}/>Scene</button>
    <aside className={`scene-panel ${panelOpen ? 'mobile-open' : ''}`} aria-label="Scene controls">
      <div className="panel-title"><h2>Scene</h2><span className="live-mark">Live</span><IconButton title="Close scene controls" className="mobile-close" onClick={() => setPanelOpen(false)}><X size={17}/></IconButton></div>
      <label className="form-label" htmlFor="case-select">Dataset</label>
      <div className="select-wrap"><select id="case-select" value={scene?.id ?? ''} onChange={event => loadCase(event.target.value)} disabled={!meta || loadingCase}>{!meta && <option>Loading…</option>}{meta?.cases?.map(item => <option key={item.id} value={item.id}>Layout {String(item.layout_index ?? item.id).padStart(3, '0')} · {item.wind_direction_deg ?? item.direction_deg}° · {item.partition}</option>)}</select><ChevronDown size={14}/></div>
      <div className="control-row count-row"><span>Turbines</span><div className="stepper"><button aria-label="Remove turbine" disabled={!scene || scene.turbines.length <= 1} onClick={() => changeCount(-1)}><Minus size={14}/></button><output>{scene?.turbines.length ?? '—'}</output><button aria-label="Add turbine" disabled={!scene || scene.turbines.length >= (meta?.controls?.turbine_count?.max ?? 30)} onClick={() => changeCount(1)}><Plus size={14}/></button></div></div>
      <div className="control-row"><span>Inflow</span><span className="numeric">{scene?.wind?.speed_m_s ?? 9}<small> m/s</small></span></div>
      <div className="inflow-line" aria-hidden="true"/>
      <div className="direction-block"><div className="control-row"><span>Direction</span><output className="numeric">{scene?.wind?.direction_deg ?? 270}°</output></div><div className="segmented direction-options">{directions.map(direction => <button key={direction} className={scene?.wind?.direction_deg === direction ? 'selected' : ''} disabled={!scene} onClick={() => updateScene(previous => ({ ...previous, wind: { ...previous.wind, direction_deg: direction } }))}>{direction}°</button>)}</div></div>
      <div className="section-divider"/>
      <div className="control-row section-label"><span>Domain</span><small>D</small></div>
      <div className="dimensions">{['X', 'Y', 'Z'].map((name, axis) => <label key={name}><span>{name}</span><input aria-label={`Domain ${name}`} type="number" step="0.5" min="1" max={axis === 2 ? 6.25 : 100} value={scene ? Number((scene.domain.max[axis] - scene.domain.min[axis]).toFixed(2)) : ''} onChange={event => changeDomain(axis, Number(event.target.value))} disabled={!scene}/></label>)}</div>
      {chosen && <div className="selected-turbine"><div className="control-row"><span>Turbine {chosen.id}</span><button className="text-button" onClick={() => setSelected(null)} aria-label="Deselect turbine"><X size={13}/></button></div><div className="coordinate-inputs">{['x', 'y'].map(axis => <label key={axis}>{axis.toUpperCase()}<input aria-label={`Turbine ${axis.toUpperCase()}`} type="number" value={Number(chosen[axis].toFixed(2))} step="0.1" onChange={event => moveTurbine(chosen.id, axis === 'x' ? Number(event.target.value) : chosen.x, axis === 'y' ? Number(event.target.value) : chosen.y)}/></label>)}</div></div>}
      <button className="reconstruct-button" disabled={!scene || loadingCase || busy} onClick={reconstruct}>{busy ? <LoaderCircle className="spin" size={17}/> : <Play size={16} fill="currentColor"/>}{busy ? 'Reconstructing' : 'Reconstruct'}{!busy && <ArrowUpRight size={17}/>}</button>
      <button className="advanced-button" onClick={() => setAdvanced(value => !value)} aria-expanded={advanced}><Settings2 size={14}/>Display settings<ChevronDown size={13} className={advanced ? 'flipped' : ''}/></button>
      {advanced && <div className="advanced-controls"><Range label="Vertical scale" value={view.exaggeration} min={0.5} max={5} step={0.1} unit="×" onChange={value => updateView('exaggeration', value)}/><label className="control-row">Resolution<select aria-label="Resolution" value={quality} onChange={event => setQuality(event.target.value)}><option value="draft">Quick</option><option value="standard">Standard</option><option value="fine">Fine</option></select></label></div>}
    </aside>

    <main className="workspace" aria-label="Three-dimensional flow workspace">
      <div className="workspace-heading"><h1>Wind, revealed.</h1><p>{view.mode === 'organization' ? 'Learned M/E organization' : 'Three-dimensional flow reconstruction'}</p></div>
      <div className="mode-switch" role="tablist" aria-label="Visualization mode"><button role="tab" aria-selected={view.mode === 'field'} onClick={() => changeMode('field')}><Wind size={15}/>Flow field</button><button role="tab" aria-selected={view.mode === 'organization'} onClick={() => changeMode('organization')}><Network size={15}/>Learned hypergraph</button></div>
      {view.mode === 'organization' && <div className="organization-inspector"><label htmlFor="hyperedge-select">E-address hyperedge</label><select id="hyperedge-select" value={view.organizationAddress ?? 'all'} onChange={event => updateView('organizationAddress', event.target.value === 'all' ? null : Number(event.target.value))}><option value="all">All representative rows</option>{result?.organization?.selection?.selected_anchor_ids?.map(id => <option key={id} value={id}>E{id} ← M/E members</option>)}</select><span><i className="m-swatch"/>M turbines<i className="e-swatch"/>E environment</span></div>}
      <div className="scene-viewport"><FlowScene scene={scene} result={visibleResult} view={view} onTurbineMove={moveTurbine} onSelect={setSelected} onDraggingChange={setDragging} selected={selected} cameraPreset={cameraPreset} cameraRevision={cameraRevision}/></div>
      <ScientificPanels scene={scene} result={visibleResult} field={view.field} range={fieldRange} onPlaneSelect={({ axis, position }) => setView(previous => ({ ...previous, mode: 'field', organization: false, slice: true, sliceAxis: axis, slicePosition: position }))}/>
      {(!scene || (dirty && !error)) && <div className="canvas-loading" aria-live="polite"><LoaderCircle className="spin" size={19}/><span>{scene ? 'Reconstructing flow' : 'Loading checkpoint'}</span></div>}
      {error && <div className="error-toast" role="alert"><span>{error}</span><button onClick={reconstruct}>Retry</button></div>}
      <div className="canvas-tools" role="toolbar" aria-label="Camera and editing tools">
        <IconButton title="Orbit view" active={view.tool === 'orbit'} onClick={() => updateView('tool', 'orbit')}><MousePointer2 size={20}/></IconButton>
        <IconButton title="Move turbine" active={view.tool === 'move'} onClick={() => updateView('tool', 'move')}><Move size={20}/></IconButton>
        <div className="tool-separator"/>
        <IconButton title="Perspective view" active={cameraPreset === 'perspective'} onClick={() => { setCameraPreset('perspective'); setCameraRevision(value => value + 1); }}><Box size={20}/></IconButton>
        <IconButton title="Top view" active={cameraPreset === 'top'} onClick={() => { setCameraPreset('top'); setCameraRevision(value => value + 1); }}><Layers3 size={20}/></IconButton>
        <IconButton title="Fit scene" onClick={() => { setCameraPreset('fit'); setCameraRevision(value => value + 1); }}><Crosshair size={20}/></IconButton>
        <div className="tool-separator"/>
        <IconButton title="Fullscreen" onClick={() => document.fullscreenElement ? document.exitFullscreen?.() : document.documentElement.requestFullscreen?.()}><Expand size={20}/></IconButton>
      </div>

      {view.mode === 'field' && view.slice && <div className="slice-inspector"><span>Section</span><div className="segmented">{[['z', 'XY'], ['y', 'XZ'], ['x', 'YZ']].map(([axis, name]) => <button key={axis} className={view.sliceAxis === axis ? 'selected' : ''} onClick={() => updateView('sliceAxis', axis)}>{name}</button>)}</div><input aria-label="Section position" type="range" min="0" max="1" step="0.01" value={view.slicePosition} onChange={event => updateView('slicePosition', Number(event.target.value))}/><output>{visibleResult?.grid ? format(sliceSampleCoordinates(visibleResult.grid, view.sliceAxis, view.slicePosition)?.fixedCoordinate) : '—'} D</output></div>}

      <div className="field-legend"><div><select aria-label="Physical field" value={view.field} onChange={event => updateView('field', event.target.value)}><option value="speed">|U|</option><option value="ux">Ux</option><option value="uy">Uy</option><option value="uz">Uz</option></select><span>m/s</span></div><div className="colorbar" style={{ background: `linear-gradient(90deg,${FIELD_COLORS.join(',')})` }}/><div className="legend-ticks"><span>{format(fieldRange[0])}</span><span>{format(fieldRange[1])}</span></div></div>

      {view.mode === 'field' ? <div className="view-dock" role="toolbar" aria-label="Flow display">
        <button className="play-button" aria-label={view.playing ? 'Pause animation' : 'Play animation'} onClick={() => updateView('playing', !view.playing)}>{view.playing ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</button>
        <div className="dock-separator"/>
        <button className={`dock-toggle ${view.streamlines ? 'on' : ''}`} aria-pressed={view.streamlines} onClick={() => updateView('streamlines', !view.streamlines)}><Wind size={20}/><span>Streamlines</span><i/></button>
        <div className="dock-separator"/>
        <button className={`dock-toggle ${view.slice ? 'on' : ''}`} aria-pressed={view.slice} onClick={() => updateView('slice', !view.slice)}><Layers3 size={19}/><span>Slice</span><i/></button>
        <div className="dock-separator"/>
        <button className="dock-toggle" onClick={() => changeMode('organization')}><Network size={19}/><span>See structure</span></button>
      </div> : <div className="view-dock organization-dock"><button className="play-button" aria-label={view.playing ? 'Pause animation' : 'Play animation'} onClick={() => updateView('playing', !view.playing)}>{view.playing ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</button><div className="dock-separator"/><span>{scene?.turbines.length ?? '—'} M <small>turbines</small></span><div className="dock-separator"/><span>{result?.organization?.anchors?.length ?? '—'} E <small>environment</small></span><Network size={20}/></div>}
      <div className="field-evidence">{view.mode === 'field' ? <ErrorReadout result={visibleResult} dirty={dirty} view={view}/> : <span className="error-readout">Full directory · {view.organizationAddress === null ? visibleResult?.organization?.selection.shown_rows ?? '—' : 1}/{visibleResult?.organization?.selection.total_rows ?? '—'} E-address hyperedges · all positive members</span>}<span className="mask-note">Z ×{view.exaggeration.toFixed(1)}{view.mode === 'organization' ? ' · schematic links' : visibleResult?.errors ? ' · native fluid reference' : ''}</span></div>
    </main>

    <footer className="statusbar"><span><Activity size={13}/>{timing === undefined ? 'Inference —' : `Inference ${Math.round(timing)} ms`}</span><span>Epoch {epoch}</span><span>{meta?.checkpoint?.training_cases ?? 420} training cases</span><span className={edited ? 'edited-label' : ''}>{edited ? 'Edited scene · model exploration' : 'Dataset scene'}</span><span className="status-hint">Drag to orbit · Scroll to zoom</span></footer>

    {info && <div className="modal-backdrop" onClick={() => setInfo(false)}><section className="info-modal" role="dialog" aria-modal="true" aria-labelledby="model-title" onClick={event => event.stopPropagation()}><div className="panel-title"><h2 id="model-title">Model & data</h2><IconButton title="Close model information" onClick={() => setInfo(false)}><X size={18}/></IconButton></div><dl><dt>Checkpoint</dt><dd>{meta?.checkpoint?.run}</dd><dt>Training</dt><dd>Full population · {meta?.checkpoint?.training_cases ?? 420} cases · epoch {epoch}</dd><dt>Prediction</dt><dd>Joint Ux, Uy, Uz · m/s</dd><dt>Coordinates</dt><dd>Rotor diameter D = 80 m</dd><dt>Inflow</dt><dd>9 m/s profile · 270°, 285°, 300°</dd><dt>Organization</dt><dd>Actual prepared M/E memberships</dd></dl><p>Edited scenes have no CFD reference; inflow speed is fixed by the saved model. Direction selects the trained direction category in the current layout frame.</p><p>Animated particles trace the reconstructed steady velocity field.</p></section></div>}
  </div>;
}
