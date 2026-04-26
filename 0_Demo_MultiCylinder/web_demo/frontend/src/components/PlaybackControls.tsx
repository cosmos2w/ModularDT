import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

interface Props {
  isPlaying: boolean;
  frame: number;
  frameCount: number;
  speed: number;
  onPlayingChange: (playing: boolean) => void;
  onFrameChange: (frame: number) => void;
  onSpeedChange: (speed: number) => void;
}

export default function PlaybackControls({
  isPlaying,
  frame,
  frameCount,
  speed,
  onPlayingChange,
  onFrameChange,
  onSpeedChange,
}: Props) {
  const maxFrame = Math.max(frameCount - 1, 0);
  return (
    <div className="playback-controls">
      <button className="icon-button" type="button" title="Previous phase" onClick={() => onFrameChange(frame === 0 ? maxFrame : frame - 1)}>
        <SkipBack size={17} />
      </button>
      <button className="icon-button primary-icon" type="button" title={isPlaying ? "Pause" : "Play"} onClick={() => onPlayingChange(!isPlaying)}>
        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
      </button>
      <button className="icon-button" type="button" title="Next phase" onClick={() => onFrameChange(frame >= maxFrame ? 0 : frame + 1)}>
        <SkipForward size={17} />
      </button>
      <input
        className="phase-slider"
        type="range"
        min="0"
        max={maxFrame}
        value={Math.min(frame, maxFrame)}
        onChange={(event) => onFrameChange(Number(event.target.value))}
      />
      <label className="speed-control">
        <span>{speed.toFixed(1)}x</span>
        <input
          type="range"
          min="0.25"
          max="3"
          step="0.25"
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        />
      </label>
    </div>
  );
}
