import { Grid3X3, MousePointer2, Move, Plus, Scaling, Trash2 } from 'lucide-react';

export default function ModuleDesignPanel({ scene, view, selected, domainLimits, onSectionChange, onToolChange, onDomainChange, onDeleteSelected }) {
  const layout = view.designSection !== 'domain';
  const tools = [['add', 'Add', Plus], ['move', 'Move', Move], ['delete', 'Delete', Trash2]];
  return <section className="design-panel" aria-label="Module design controls">
    <div className="field-panel-title"><h2>Module design</h2><Grid3X3 size={15}/></div>
    <div className="segmented design-sections">
      <button className={layout ? 'selected' : ''} onClick={() => onSectionChange('layout')}>Turbine layout</button>
      <button className={!layout ? 'selected' : ''} onClick={() => onSectionChange('domain')}>Domain sizing</button>
    </div>
    {layout ? <>
      <div className="design-tools">{tools.map(([tool, label, Icon]) => <button key={tool} aria-label={`${label} turbines`} aria-pressed={view.tool === tool} className={view.tool === tool ? 'selected' : ''} disabled={!scene || (tool === 'add' && scene.turbines.length >= 30) || (tool === 'delete' && scene.turbines.length <= 1)} onClick={() => onToolChange(tool)}><Icon size={18}/><span>{label}</span></button>)}</div>
      <div className="design-context"><span>{scene?.turbines.length ?? '—'} turbines</span><span>{selected == null ? 'No selection' : `M${selected}`}</span></div>
      <span className="design-hint">{view.tool === 'add' ? 'Click the grid to place a turbine' : view.tool === 'delete' ? 'Click a turbine to remove it' : view.tool === 'move' ? 'Drag a turbine to move it' : 'Drag to orbit · Click to select'}</span>
      {selected != null && <button className="delete-selection" disabled={!scene || scene.turbines.length <= 1} onClick={onDeleteSelected}><Trash2 size={13}/>Delete selected</button>}
    </> : <>
      <div className="domain-design-title"><Scaling size={16}/><span>Horizontal extent</span><span>D</span></div>
      {[0, 1].map(axis => <label className="range-control design-domain-range" key={axis}><span>{axis === 0 ? 'Length X' : 'Width Y'}<output>{scene ? (scene.domain.max[axis] - scene.domain.min[axis]).toFixed(1) : '—'} D</output></span><input type="range" aria-label={axis === 0 ? 'Domain length X' : 'Domain width Y'} min={domainLimits[axis][0]} max={domainLimits[axis][1]} step="0.1" value={scene ? scene.domain.max[axis] - scene.domain.min[axis] : domainLimits[axis][0]} disabled={!scene} onChange={event => onDomainChange(axis, Number(event.target.value))}/></label>)}
      <span className="design-hint">Drag X / Y edge handles or use the sliders</span>
    </>}
    <button className={`design-orbit ${view.tool === 'orbit' ? 'selected' : ''}`} onClick={() => onToolChange('orbit')}><MousePointer2 size={14}/>Orbit view</button>
  </section>;
}
