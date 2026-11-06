import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, ArrowUpRight, Box, ChevronDown, CircleHelp, Crosshair, Expand, Layers3, LoaderCircle, Minus, MousePointer2, Move, Network, Pause, Play, Plus, RotateCcw, Settings2, Wind, X } from 'lucide-react';
import FlowScene from './FlowScene.jsx';
import ScientificWorkspace from './ScientificWorkspace.jsx';
import ModuleDesignPanel from './ModuleDesignPanel.jsx';
import { clampTurbinePosition } from './designMath.js';
import { FIELD_SPECS, fieldValue, getFieldRange, sampleVelocity, sliceSampleCoordinates } from './flowMath.js';

const INITIAL_VIEW = { mode: 'field', designSection: 'layout', organizationView: 'spatial', organizationLinks: 'focused', organizationAddress: null, field: 'speed', contrast: 'wake', streamlines: true, slice: true, sliceAxis: 'z', slicePosition: 0.14, organization: false, exaggeration: 2.8, playing: true, tool: 'orbit' };
const RESOLUTIONS = { draft: [28, 20, 12], standard: [40, 28, 16], fine: [56, 32, 16] };
const copy = value => structuredClone(value);
const signature = scene => JSON.stringify(scene && [scene.id, scene.turbines, scene.domain, scene.wind]);
const format = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : '—';
const formatError = value => Number.isFinite(Number(value)) ? Number(value).toPrecision(3) : '—';
const FIELD_NAMES = { speed: 'Speed', ux: 'Downstream', uy: 'Crosswind', uz: 'Vertical', deficit: 'Deficit' };

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
  const displayedError = useMemo(() => {
    const grid = result?.grid;
    if (!grid?.reference_velocity) return null;
    let sum = 0, count = 0;
    const accumulate = (prediction, reference, point) => {
      if (!prediction || !reference) return;
      const difference = fieldValue(prediction, view.field, point, grid) - fieldValue(reference, view.field, point, grid);
      if (!Number.isFinite(difference)) return;
      sum += difference ** 2;
      count += 1;
    };
    if (view.slice) {
      for (const point of sliceSampleCoordinates(grid, view.sliceAxis, view.slicePosition).coordinates) accumulate(sampleVelocity(grid, point), sampleVelocity(grid, point, grid.reference_velocity), point);
    } else {
      for (let i = 0; i < grid.velocity.length; i += 3) {
        const index = i / 3, [, ny, nz] = grid.shape;
        const point = [grid.axes.x[Math.floor(index / (ny * nz))], grid.axes.y[Math.floor(index / nz) % ny], grid.axes.z[index % nz]];
        accumulate(grid.velocity.slice(i, i + 3), grid.reference_velocity.slice(i, i + 3), point);
      }
    }
    return count ? Math.sqrt(sum / count) : null;
  }, [result, view.field, view.slice, view.sliceAxis, view.slicePosition]);
  if (!result || dirty || displayedError === null) return <span className="error-readout">Reference error unavailable{result && !dirty ? ' · edited scene' : ''}</span>;
  return <span className="error-readout">{view.slice ? 'Slice' : 'Volume'} {FIELD_SPECS[view.field].label} RMSE {formatError(displayedError)} m/s</span>;
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
  const [organizationSummary, setOrganizationSummary] = useState(null);
  const [mapsLayout, setMapsLayout] = useState(() => ({ width: window.innerWidth >= 1750 ? 328 : 286, fullscreen: false }));
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
  const changeMode = mode => setView(previous => ({ ...previous, mode, organization: mode === 'organization', tool: 'orbit', organizationView: mode === 'organization' && previous.mode !== mode ? 'spatial' : previous.organizationView, organizationAddress: mode === 'organization' && previous.mode !== mode ? null : previous.organizationAddress }));
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
    if (![x, y].every(Number.isFinite)) return;
    updateScene(previous => {
      const [nextX, nextY] = clampTurbinePosition(previous.domain, x, y);
      return { ...previous, turbines: previous.turbines.map(turbine => turbine.id === id ? { ...turbine, x: nextX, y: nextY } : turbine) };
    });
  }, [updateScene]);

  const addTurbine = useCallback((x, y) => {
    if (![x, y].every(Number.isFinite)) return;
    updateScene(previous => {
      if (!previous || previous.turbines.length >= 30) return previous;
      const ids = new Set(previous.turbines.map(turbine => String(turbine.id)));
      let id = 0; while (ids.has(String(id))) id += 1;
      const [nextX, nextY] = clampTurbinePosition(previous.domain, x, y);
      return { ...previous, turbines: [...previous.turbines, { id, x: nextX, y: nextY, z: 0.875 }] };
    });
  }, [updateScene]);

  const deleteTurbine = useCallback(id => {
    updateScene(previous => !previous || previous.turbines.length <= 1 ? previous : { ...previous, turbines: previous.turbines.filter(turbine => turbine.id !== id) });
    setSelected(previous => previous === id ? null : previous);
  }, [updateScene]);

  const domainLimits = useMemo(() => [0, 1].map(axis => {
    if (!scene) return [4, 100];
    const center = (scene.domain.min[axis] + scene.domain.max[axis]) / 2;
    const required = 2 * Math.max(0, ...scene.turbines.map(turbine => Math.abs(turbine[axis === 0 ? 'x' : 'y'] - center))) + 1;
    return [Math.min(100, Math.max(4, required)), 100];
  }), [scene]);

  const changeCount = delta => {
    if (!scene) return;
    if (delta < 0) { deleteTurbine(scene.turbines.at(-1)?.id); return; }
    const center = scene.domain.min.map((value, axis) => (value + scene.domain.max[axis]) / 2);
    addTurbine(center[0] + (scene.turbines.length % 3 - 1) * 1.2, center[1]);
  };

  const changeDomain = (axis, length) => {
    if (!Number.isFinite(length) || length <= 0) return;
    updateScene(previous => {
      const center = (previous.domain.min[axis] + previous.domain.max[axis]) / 2;
      const required = axis === 2 ? 1.5 : Math.min(100, Math.max(4, 2 * Math.max(0, ...previous.turbines.map(turbine => Math.abs(turbine[['x', 'y'][axis]] - center))) + 1));
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
    return getFieldRange(visibleResult?.grid, view.field, { contrast: view.contrast, turbines: scene?.turbines }) || { min: 0, max: 12, clipped: false, scope: 'pending' };
  }, [visibleResult, view.field, view.contrast, scene?.turbines]);
  const fieldSpec = FIELD_SPECS[view.field];
  const vennMode = view.organizationView === 'venn';
  const availableRows = vennMode ? visibleResult?.organization?.directory_rows || [] : visibleResult?.organization?.anchors?.filter(anchor => anchor.members) || [];

  const directions = meta?.controls?.wind_direction_deg?.options || [270, 285, 300];
  const status = error ? 'Attention needed' : loadingCase ? 'Loading scene' : busy || (dirty && scene) ? 'Reconstructing' : result ? 'Model ready' : 'Loading model';
  const epoch = meta?.checkpoint?.epoch ?? '—';
  const timing = result?.provenance?.inference_ms ?? (result?.provenance?.inference_seconds === undefined ? undefined : result.provenance.inference_seconds * 1000);

  return <div className={`app-shell ${view.mode === 'organization' ? 'organization-mode' : view.mode === 'design' ? 'design-mode' : 'field-mode'} ${mapsLayout.width > 420 ? 'maps-expanded' : ''}`}>
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

    <main className="workspace" aria-label="Three-dimensional flow workspace" style={{ '--science-width': `${mapsLayout.width}px`, '--science-half': `${mapsLayout.width / 2}px` }}>
      <div className="workspace-heading"><h1>Wind, revealed.</h1><p>{view.mode === 'organization' ? 'Learned M/E organization' : view.mode === 'design' ? 'Turbine layout & domain design' : 'Three-dimensional flow reconstruction'}</p></div>
      <div className="mode-switch" role="tablist" aria-label="Visualization mode"><button role="tab" aria-selected={view.mode === 'field'} onClick={() => changeMode('field')}><Wind size={15}/>Flow field</button><button role="tab" aria-selected={view.mode === 'design'} onClick={() => changeMode('design')}><Settings2 size={15}/>Module design</button><button role="tab" aria-selected={view.mode === 'organization'} onClick={() => changeMode('organization')}><Network size={15}/>Learned hypergraph</button></div>
      {view.mode === 'design' && <ModuleDesignPanel scene={scene} view={view} selected={selected} domainLimits={domainLimits} onSectionChange={section => setView(previous => ({ ...previous, designSection: section, tool: 'orbit' }))} onToolChange={tool => updateView('tool', tool)} onDomainChange={changeDomain} onDeleteSelected={() => deleteTurbine(selected)}/>}
      {view.mode === 'field' && <section className="field-panel" aria-label="Physical quantity panel"><div className="field-panel-title"><h2>Physical field</h2><span>m/s</span></div><div className="field-options">{Object.entries(FIELD_SPECS).map(([key, spec]) => <button key={key} aria-label={`${FIELD_NAMES[key]} (${spec.label})`} aria-pressed={view.field === key} className={view.field === key ? 'selected' : ''} onClick={() => updateView('field', key)} title={spec.description}><span>{FIELD_NAMES[key]}</span><strong>{spec.label}</strong><i style={{ background: `linear-gradient(90deg,${spec.colors.join(',')})` }}/></button>)}</div><div className="field-scale-heading">Color scale</div><div className="segmented contrast-options"><button className={view.contrast === 'wake' ? 'selected' : ''} onClick={() => updateView('contrast', 'wake')}>Wake focus</button><button className={view.contrast === 'global' ? 'selected' : ''} onClick={() => updateView('contrast', 'global')}>Full range</button></div><span className="scale-status">{fieldRange.scope === 'pending' ? 'Awaiting field' : fieldRange.percentiles ? 'Rotor-band scale · 2–98%' : 'Full field scale'}</span></section>}
      {view.mode === 'organization' && <div className="organization-inspector"><div className="segmented organization-view-options"><button className={vennMode ? 'selected' : ''} onClick={() => setView(previous => ({ ...previous, organizationView: 'venn', organizationAddress: null, tool: 'orbit' }))}>Venn sets</button><button className={!vennMode ? 'selected' : ''} onClick={() => setView(previous => ({ ...previous, organizationView: 'spatial', organizationAddress: null }))}>Spatial links</button></div><label htmlFor="hyperedge-select">{vennMode ? 'Native hyperedges' : 'E-address hyperedge'}</label><select id="hyperedge-select" value={view.organizationAddress ?? 'auto'} onChange={event => updateView('organizationAddress', event.target.value === 'auto' ? null : Number(event.target.value))}><option value="auto">{vennMode ? 'Automatic selection' : 'All representative rows'}</option>{availableRows.map(row => <option key={row.id} value={row.id}>E{row.id} ← M/E members</option>)}</select>{vennMode && <div className="venn-row-badges">{organizationSummary?.rowIds?.map(id => <span key={id}>h(E{id})</span>)}</div>}{vennMode && organizationSummary?.selection && <span className="venn-coverage" title="Directory profiles within L2 distance 0.5 of a displayed row; at most three example sets.">{Math.round(organizationSummary.selection.coverage * 100)}% profile coverage{organizationSummary.selection.budgetLimited ? ' · 3-set cap' : ''}</span>}{!vennMode && <><div className="segmented organization-link-options"><button className={view.organizationLinks === 'focused' ? 'selected' : ''} onClick={() => updateView('organizationLinks', 'focused')}>Focused links</button><button className={view.organizationLinks === 'all' ? 'selected' : ''} onClick={() => updateView('organizationLinks', 'all')}>All links</button></div><span className="link-coverage">{organizationSummary?.shownLinks ?? '—'} / {organizationSummary?.totalLinks ?? '—'} members · {organizationSummary?.retainedMass == null ? '—' : (100 * organizationSummary.retainedMass).toFixed(1)}% weight</span></>}<span><i className="m-swatch"/>M turbines<i className="e-swatch"/>E environment</span></div>}
      <div className="scene-viewport"><FlowScene scene={scene} result={visibleResult} fieldRange={fieldRange} view={view} domainLimits={domainLimits} onDomainChange={changeDomain} onTurbineAdd={addTurbine} onTurbineDelete={deleteTurbine} onTurbineMove={moveTurbine} onSelect={setSelected} onDraggingChange={setDragging} onOrganizationSummary={setOrganizationSummary} selected={selected} cameraPreset={cameraPreset} cameraRevision={cameraRevision}/></div>
      <ScientificWorkspace scene={scene} result={visibleResult} field={view.field} range={fieldRange} onLayoutChange={setMapsLayout} onPlaneSelect={({ axis, position }) => setView(previous => ({ ...previous, mode: 'field', organization: false, tool: 'orbit', slice: true, sliceAxis: axis, slicePosition: position }))}/>
      {view.mode !== 'design' && (!scene || (dirty && !error)) && <div className="canvas-loading" aria-live="polite"><LoaderCircle className="spin" size={19}/><span>{scene ? 'Reconstructing flow' : 'Loading checkpoint'}</span></div>}
      {error && <div className="error-toast" role="alert"><span>{error}</span><button onClick={reconstruct}>Retry</button></div>}
      <div className="canvas-tools" role="toolbar" aria-label="Camera and editing tools">
        <IconButton title="Orbit view" active={view.tool === 'orbit'} onClick={() => updateView('tool', 'orbit')}><MousePointer2 size={20}/></IconButton>
        <IconButton title="Move turbine" disabled={view.mode === 'organization' && vennMode} active={view.tool === 'move'} onClick={() => updateView('tool', 'move')}><Move size={20}/></IconButton>
        <div className="tool-separator"/>
        <IconButton title="Perspective view" active={cameraPreset === 'perspective'} onClick={() => { setCameraPreset('perspective'); setCameraRevision(value => value + 1); }}><Box size={20}/></IconButton>
        <IconButton title="Top view" disabled={view.mode === 'organization' && vennMode} active={cameraPreset === 'top' && !(view.mode === 'organization' && vennMode)} onClick={() => { setCameraPreset('top'); setCameraRevision(value => value + 1); }}><Layers3 size={20}/></IconButton>
        <IconButton title="Fit scene" onClick={() => { setCameraPreset('fit'); setCameraRevision(value => value + 1); }}><Crosshair size={20}/></IconButton>
        <div className="tool-separator"/>
        <IconButton title="Fullscreen" onClick={() => document.fullscreenElement ? document.exitFullscreen?.() : document.documentElement.requestFullscreen?.()}><Expand size={20}/></IconButton>
      </div>

      {view.mode === 'field' && view.slice && <div className="slice-inspector"><span>Section</span><div className="segmented">{[['z', 'XY'], ['y', 'XZ'], ['x', 'YZ']].map(([axis, name]) => <button key={axis} className={view.sliceAxis === axis ? 'selected' : ''} onClick={() => updateView('sliceAxis', axis)}>{name}</button>)}</div><input aria-label="Section position" type="range" min="0" max="1" step="0.01" value={view.slicePosition} onChange={event => updateView('slicePosition', Number(event.target.value))}/><output>{visibleResult?.grid ? format(sliceSampleCoordinates(visibleResult.grid, view.sliceAxis, view.slicePosition)?.fixedCoordinate) : '—'} D</output></div>}

      {view.mode === 'field' && <div className="field-legend"><div><span>{fieldSpec.label}</span><span>m/s</span></div><div className="colorbar" style={{ background: `linear-gradient(90deg,${fieldSpec.colors.join(',')})` }}/><div className="legend-ticks"><span>{format(fieldRange.min)}</span><span>{format(fieldRange.max)}</span></div></div>}

      {view.mode === 'field' ? <div className="view-dock" role="toolbar" aria-label="Flow display">
        <button className="play-button" aria-label={view.playing ? 'Pause animation' : 'Play animation'} onClick={() => updateView('playing', !view.playing)}>{view.playing ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</button>
        <div className="dock-separator"/>
        <button className={`dock-toggle ${view.streamlines ? 'on' : ''}`} aria-pressed={view.streamlines} onClick={() => updateView('streamlines', !view.streamlines)}><Wind size={20}/><span>Streamlines</span><i/></button>
        <div className="dock-separator"/>
        <button className={`dock-toggle ${view.slice ? 'on' : ''}`} aria-pressed={view.slice} onClick={() => updateView('slice', !view.slice)}><Layers3 size={19}/><span>Slice</span><i/></button>
        <div className="dock-separator"/>
        <button className="dock-toggle" onClick={() => changeMode('organization')}><Network size={19}/><span>See structure</span></button>
      </div> : view.mode === 'design' ? <div className="view-dock design-dock"><Settings2 size={18}/><span>Module design</span><div className="dock-separator"/><button className="dock-toggle" onClick={() => changeMode('field')}><Wind size={18}/>View flow</button></div> : <div className="view-dock organization-dock"><button className="play-button" aria-label={view.playing ? 'Pause animation' : 'Play animation'} onClick={() => updateView('playing', !view.playing)}>{view.playing ? <Pause size={17} fill="currentColor"/> : <Play size={17} fill="currentColor"/>}</button><div className="dock-separator"/><span>{scene?.turbines.length ?? '—'} M <small>turbines</small></span><div className="dock-separator"/><span>{result?.organization?.anchors?.length ?? '—'} E <small>environment</small></span><Network size={20}/></div>}
      <div className="field-evidence">{view.mode === 'field' ? <ErrorReadout result={visibleResult} dirty={dirty} view={view}/> : view.mode === 'design' ? <span className="error-readout">{scene?.turbines.length ?? '—'} turbines · {scene ? `${format(scene.domain.max[0] - scene.domain.min[0])} × ${format(scene.domain.max[1] - scene.domain.min[1])} D` : 'Loading layout'}</span> : <span className="error-readout">Native directory · {organizationSummary?.shownRows ?? '—'}/{visibleResult?.organization?.selection.total_rows ?? '—'} native hyperedges · {vennMode || view.organizationLinks === 'all' ? 'all positive members' : 'display-filtered links'}</span>}<span className="mask-note">{view.mode === 'organization' && vennMode ? 'Exact set intersections · abstract coordinates' : `Z ×${view.exaggeration.toFixed(1)}${view.mode === 'organization' ? ' · schematic links' : view.mode === 'design' ? ' · D = 80 m' : visibleResult?.errors ? ' · native fluid reference' : ''}`}</span></div>

    </main>

    <footer className="statusbar"><span><Activity size={13}/>{timing === undefined ? 'Inference —' : `Inference ${Math.round(timing)} ms`}</span><span>Epoch {epoch}</span><span>{meta?.checkpoint?.training_cases ?? 420} training cases</span><span className={edited ? 'edited-label' : ''}>{edited ? 'Edited scene · model exploration' : 'Dataset scene'}</span><span className="status-hint">Drag to orbit · Scroll to zoom</span></footer>

    {info && <div className="modal-backdrop" onClick={() => setInfo(false)}><section className="info-modal" role="dialog" aria-modal="true" aria-labelledby="model-title" onClick={event => event.stopPropagation()}><div className="panel-title"><h2 id="model-title">Model & data</h2><IconButton title="Close model information" onClick={() => setInfo(false)}><X size={18}/></IconButton></div><dl><dt>Checkpoint</dt><dd>{meta?.checkpoint?.run}</dd><dt>Training</dt><dd>Full population · {meta?.checkpoint?.training_cases ?? 420} cases · epoch {epoch}</dd><dt>Prediction</dt><dd>Joint Ux, Uy, Uz · m/s</dd><dt>Coordinates</dt><dd>Rotor diameter D = 80 m</dd><dt>Inflow</dt><dd>9 m/s profile · 270°, 285°, 300°</dd><dt>Organization</dt><dd>Actual prepared M/E directory memberships</dd><dt>Venn display</dt><dd>Exact intersections · abstract coordinates · at most 3 representative sets</dd></dl><p>Edited scenes have no CFD reference; inflow speed is fixed by the saved model. Direction selects the trained direction category in the current layout frame.</p><p>Downstream deficit uses the checkpoint training profile at each height. Wake focus clips colors to rotor-band percentiles; values and RMSE are unchanged.</p><p>Focused spatial links preserve at least 95% of each row’s access weight plus all positive M and own-E memberships. All links restores every positive relation. Venn arrows show prepared M-to-E directory access, with schematic curves. Animated particles trace the reconstructed steady velocity field.</p></section></div>}
  </div>;
}
