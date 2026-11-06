import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, Expand, Grid2X2 } from 'lucide-react';
import ScientificPanels from './ScientificPanels.jsx';

const defaultWidth = () => window.innerWidth >= 1750 ? 328 : 286;
const initialViews = () => ({ XY: { zoom: 1, x: 0.5, y: 0.5 }, XZ: { zoom: 1, x: 0.5, y: 0.5 }, YZ: { zoom: 1, x: 0.5, y: 0.5 } });

export default function ScientificWorkspace({ scene, result, field, range, onPlaneSelect, onLayoutChange }) {
  const [width, setWidth] = useState(defaultWidth);
  const [fullscreen, setFullscreen] = useState(false);
  const [focused, setFocused] = useState(null);
  const [views, setViews] = useState(initialViews);
  const [resetRevision, setResetRevision] = useState(0);
  const wrapperRef = useRef(null);
  const defaultRef = useRef(defaultWidth());
  const dragRef = useRef(null);

  useEffect(() => { onLayoutChange?.({ width, fullscreen }); }, [width, fullscreen, onLayoutChange]);
  useEffect(() => {
    const resize = () => {
      const nextDefault = defaultWidth();
      setWidth(previous => previous === defaultRef.current ? nextDefault : Math.min(previous, window.innerWidth / 2));
      defaultRef.current = nextDefault;
    };
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  const back = useCallback(() => {
    setFullscreen(false);
    setFocused(null);
    setWidth(defaultWidth());
    setViews(initialViews());
    setResetRevision(previous => previous + 1);
  }, []);
  useEffect(() => {
    if (!fullscreen) return undefined;
    const keydown = event => { if (event.key === 'Escape') back(); };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [fullscreen, back]);

  const updateView = useCallback((name, next) => setViews(previous => ({ ...previous, [name]: next })), []);
  const focusMap = useCallback(name => { setFocused(name); setFullscreen(true); }, []);
  const selectPlane = useCallback(plane => {
    onPlaneSelect?.(plane);
    if (fullscreen) back();
  }, [onPlaneSelect, fullscreen, back]);
  const resizeMove = event => {
    if (!dragRef.current || event.pointerId !== dragRef.current.pointerId) return;
    event.preventDefault();
    const desired = dragRef.current.right - event.clientX;
    const maximum = window.innerWidth / 2;
    if (desired > maximum + 32) {
      dragRef.current = null;
      setWidth(maximum);
      setFocused(null);
      setFullscreen(true);
      return;
    }
    setWidth(Math.max(240, Math.min(maximum, desired)));
  };
  const endResize = event => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const keyboardResize = event => {
    if (event.key === 'Enter') { setFocused(null); setFullscreen(true); return; }
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    setWidth(previous => Math.max(240, Math.min(window.innerWidth / 2, previous + (event.key === 'ArrowLeft' ? 24 : -24))));
  };

  return <section ref={wrapperRef} className={`scientific-workspace ${fullscreen ? 'is-fullscreen' : ''}`} style={fullscreen ? undefined : { width }} aria-label="Scientific map workspace">
    {!fullscreen && <div className="science-resize-handle" role="separator" tabIndex={0} aria-label="Resize scientific maps" aria-orientation="vertical" aria-valuemin={240} aria-valuemax={Math.round(window.innerWidth / 2)} aria-valuenow={Math.round(width)} onKeyDown={keyboardResize}
      onPointerDown={event => { if (event.button !== 0) return; event.preventDefault(); window.getSelection()?.removeAllRanges(); dragRef.current = { pointerId: event.pointerId, right: wrapperRef.current.getBoundingClientRect().right }; event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={resizeMove} onPointerUp={endResize} onPointerCancel={endResize}><span/></div>}
    <header className="science-workspace-header">
      <div className="science-workspace-heading">{fullscreen && <button type="button" className="science-back" onClick={back} aria-label="Back from scientific maps"><ArrowLeft size={15}/> Back</button>}<div><strong>Flow sections</strong><span>{field === 'speed' ? 'Speed' : field === 'ux' ? 'Downstream velocity' : field === 'uy' ? 'Crosswind velocity' : field === 'uz' ? 'Vertical velocity' : 'Velocity deficit'}</span></div></div>
      {fullscreen ? focused && <button type="button" className="science-workspace-action" onClick={() => setFocused(null)} aria-label="View all maps"><Grid2X2 size={14}/> All maps</button> : <button type="button" className="science-workspace-action" onClick={() => { setFocused(null); setFullscreen(true); }} aria-label="Expand maps"><Expand size={14}/></button>}
    </header>
    <ScientificPanels key={resetRevision} scene={scene} result={result} field={field} range={range} onPlaneSelect={selectPlane} fullscreen={fullscreen} focused={focused} onFocus={focusMap} views={views} onViewChange={updateView}/>
  </section>;
}
