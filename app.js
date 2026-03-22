/**
 * SONIQ — Reactive Audio Visualizer
 * app.js
 *
 * Architecture:
 *  - AudioEngine  : Web Audio API setup, AnalyserNode, source management
 *  - Visualizer   : Canvas rendering, Lerp, peak caps, particles, bloom
 *  - UIController : DOM events, file upload, demo tracks, band meters, RMS
 */

'use strict';

/* ─────────────────────────────────────────────
   CONSTANTS
───────────────────────────────────────────── */
const FFT_SIZE     = 2048;
const SMOOTHING    = 0.82;      // AnalyserNode smoothingTimeConstant (0–1)
const LERP_SPEED   = 0.14;      // Per-frame lerp factor for bar heights
const PEAK_HOLD_MS = 1100;      // How long peak cap stays at the top (ms)
const PEAK_GRAVITY = 0.55;      // px/frame² – gravity for falling cap
const BAR_COUNT    = 90;        // Bars each side (mirrored → 90 × 2)
const MIN_DB       = -80;       // dBFS floor
const MAX_DB       = 0;         // dBFS ceiling

/* ─────────────────────────────────────────────
   COLOUR THEMES
───────────────────────────────────────────── */
const THEMES = {
  neon:   { hueStart:330, hueEnd:180, sat:90, wave:['#ff2d6b','#b44dff','#7c3bff','#00e5ff'], bloom1:[255,45,107],  bloom2:[124,59,255],  core:'#b44dff' },
  sunset: { hueStart:15,  hueEnd:52,  sat:90, wave:['#ff3300','#ff6600','#ff9900','#ffcc00'], bloom1:[255,100,0],   bloom2:[200,50,0],    core:'#ff7700' },
  matrix: { hueStart:105, hueEnd:145, sat:85, wave:['#00ff41','#00dd30','#00bb28','#39ff14'], bloom1:[0,210,60],    bloom2:[0,140,30],    core:'#00ff41' },
  ocean:  { hueStart:175, hueEnd:248, sat:90, wave:['#00e5ff','#0288d1','#1565c0','#7c4dff'], bloom1:[0,180,255],   bloom2:[0,80,200],    core:'#2979ff' },
  mono:   { hueStart:0,   hueEnd:0,   sat:0,  wave:['#555','#888','#bbb','#eee'],             bloom1:[160,160,160], bloom2:[100,100,100], core:'#ccc'    },
};

/* ─────────────────────────────────────────────
   UTILITY
───────────────────────────────────────────── */
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const dbToLinear = db => Math.pow(10, db / 20);

/** Linearly map a value from one range to another */
const remap = (v, a1, a2, b1, b2) => b1 + ((v - a1) / (a2 - a1)) * (b2 - b1);

/** Compute RMS from a Float32Array of time-domain samples */
function computeRMS(buf) {
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  return Math.sqrt(sum / buf.length);
}

/* Map FFT bin index → frequency (Hz) */
const binToHz = (bin, sampleRate, fftSize) => (bin * sampleRate) / fftSize;

/* ─────────────────────────────────────────────
   AUDIO ENGINE
───────────────────────────────────────────── */
class AudioEngine {
  constructor() {
    this.ctx        = null;
    this.analyser   = null;
    this.source     = null;
    this.gainNode   = null;
    this.filters    = [];     // 8-band EQ filter chain
    this.buffer     = null;
    this.micStream  = null;
    this.micNode    = null;

    this._freqData  = null;   // Uint8Array — frequency domain
    this._timeData  = null;   // Float32Array — time domain (for RMS)

    this.isPlaying  = false;
    this.startedAt  = 0;
    this.pausedAt   = 0;

    // Demo oscillator synthesizer state
    this._demoNode  = null;
    this._demoMode  = null;
    this._demoOscs  = [];
  }

  _ensureContext() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = FFT_SIZE;
      this.analyser.smoothingTimeConstant = SMOOTHING;
      this.analyser.minDecibels = MIN_DB;
      this.analyser.maxDecibels = MAX_DB;

      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = 1.0;

      // Create 8-band EQ filters
      const frequencies = [60, 170, 310, 600, 1000, 3000, 6000, 12000];
      this.filters = frequencies.map(freq => {
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'peaking';
        filter.frequency.value = freq;
        filter.Q.value = 1.4; // standard bandwidth
        filter.gain.value = 0;
        return filter;
      });

      // Chain filters: f0 -> f1 -> ... -> f7 -> gainNode
      for (let i = 0; i < this.filters.length - 1; i++) {
        this.filters[i].connect(this.filters[i + 1]);
      }
      this.filters[this.filters.length - 1].connect(this.gainNode);

      this.gainNode.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);

      const bins = this.analyser.frequencyBinCount;
      this._freqData = new Uint8Array(bins);
      this._timeData = new Float32Array(this.analyser.fftSize);
    }
  }

  async loadFile(file) {
    this._ensureContext();
    this.stop();
    const arrayBuf = await file.arrayBuffer();
    this.buffer = await this.ctx.decodeAudioData(arrayBuf);
    this._demoMode = null;
  }

  play(offset = null) {
    if (!this.ctx) return;
    if (this.ctx.state === 'suspended') this.ctx.resume();
    if (this.isPlaying && offset === null) return;

    if (this._demoMode) {
      this._startDemo(this._demoMode);
      return;
    }
    if (!this.buffer) return;

    this._stopSource();
    this.source = this.ctx.createBufferSource();
    this.source.buffer = this.buffer;
    
    // Connect to START of EQ chain
    this.source.connect(this.filters[0]);
    
    this.source.loop = false;
    
    const startOffset = (offset !== null) ? offset : this.pausedAt;
    this.source.start(0, startOffset);
    this.startedAt = this.ctx.currentTime - startOffset;
    this.isPlaying = true;

    this.source.onended = () => {
      if (this.isPlaying) {
        this.isPlaying = false;
        this.pausedAt = 0;
        ui.onEnded();
      }
    };
  }

  pause() {
    if (!this.isPlaying) return;
    if (this._demoMode) {
      this._stopDemoOscs();
      this.isPlaying = false;
      return;
    }
    this.pausedAt = this.ctx.currentTime - this.startedAt;
    this._stopSource();
    this.isPlaying = false;
  }

  stop() {
    this._stopSource();
    this._stopDemoOscs();
    this.stopMic();
    this.isPlaying = false;
    this.pausedAt  = 0;
    this.startedAt = 0;
  }

  _stopSource() {
    if (this.source) {
      this.source.onended = null; // Prevent race condition when stopping to seek
      try { this.source.stop(); } catch(_) {}
      this.source.disconnect();
      this.source = null;
    }
  }

  /* ── Demo synth modes ── */
  startDemoMode(mode) {
    this._ensureContext();
    this.stop();
    this._demoMode = mode;
    this._startDemo(mode);
  }

  _startDemo(mode) {
    this._stopDemoOscs();
    if (this.ctx.state === 'suspended') this.ctx.resume();

    const now = this.ctx.currentTime;
    this.isPlaying = true;

    const make = (type, freq, gain) => {
      const osc = this.ctx.createOscillator();
      const g   = this.ctx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      g.gain.value = gain;
      osc.connect(g);
      g.connect(this.filters[0]);
      osc.start(now);
      this._demoOscs.push({ osc, gain: g });
    };

    if (mode === 'bass') {
      // Deep sub-bass thumper with overtone
      make('sine',     55,  0.55);
      make('triangle', 110, 0.25);
      make('sine',     82,  0.20);
      // LFO-modulate gain for that kick feel
      this._modulateDemo(0, 1.6, 0.8);

    } else if (mode === 'mids') {
      make('sawtooth', 220, 0.15);
      make('square',   330, 0.10);
      make('sine',     440, 0.20);
      make('sine',     528, 0.12);
      this._modulateDemo(1, 3.2, 0.5);

    } else if (mode === 'treble') {
      make('sine',     2200, 0.18);
      make('sine',     4400, 0.12);
      make('triangle', 8800, 0.08);
      make('square',   3300, 0.06);
      this._modulateDemo(2, 7.0, 0.4);
    }
  }

  _modulateDemo(oscIdx, rate, depth) {
    if (!this._demoOscs[oscIdx]) return;
    const lfo  = this.ctx.createOscillator();
    const lfoG = this.ctx.createGain();
    lfo.frequency.value = rate;
    lfoG.gain.value = depth;
    lfo.connect(lfoG);
    lfoG.connect(this._demoOscs[oscIdx].gain.gain);
    lfo.start();
    this._demoOscs.push({ osc: lfo, gain: lfoG }); // track for cleanup
  }

  _stopDemoOscs() {
    for (const { osc } of this._demoOscs) {
      try { osc.stop(); } catch(_) {}
      try { osc.disconnect(); } catch(_) {}
    }
    this._demoOscs = [];
  }

  /* ── Data accessors ── */
  getFreqData() {
    if (!this.analyser) return null;
    this.analyser.getByteFrequencyData(this._freqData);
    return this._freqData;
  }

  getRMS() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this._timeData);
    return computeRMS(this._timeData);
  }

  get duration()    { return this.buffer ? this.buffer.duration : 0; }
  get currentTime() {
     if (!this.isPlaying || !this.ctx) return this.pausedAt;
     return Math.min(this.duration, this.ctx.currentTime - this.startedAt);
  }

  setVolume(val) {
    if (this.gainNode) {
      this.gainNode.gain.setTargetAtTime(val, this.ctx.currentTime, 0.02);
    }
  }

  setEqGain(freq, db) {
    if (!this.ctx) return;
    const filter = this.filters.find(f => f.frequency.value === freq);
    if (filter) {
      filter.gain.setTargetAtTime(db, this.ctx.currentTime, 0.05);
    }
  }

  get sampleRate()     { return this.ctx ? this.ctx.sampleRate : 44100; }
  get binCount()       { return this.analyser ? this.analyser.frequencyBinCount : 0; }
  get timeDomainData() { return this._timeData; }  // already current after getRMS()

  /* ── Microphone ── */
  async startMic() {
    this._ensureContext();
    this.stop();

    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.micNode = this.ctx.createMediaStreamSource(this.micStream);
      this.micNode.connect(this.filters[0]);
      this.isPlaying = true;
      return true;
    } catch (err) {
      console.error('Microphone error:', err);
      return false;
    }
  }

  stopMic() {
    if (this.micNode) {
      this.micNode.disconnect();
      this.micNode = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
  }
}

/* ─────────────────────────────────────────────
   PARTICLE SYSTEM (Treble-driven sparks)
───────────────────────────────────────────── */
class Particle {
  constructor(x, y, color) {
    this.x  = x;
    this.y  = y;
    this.vx = (Math.random() - 0.5) * 4;
    this.vy = -(Math.random() * 3 + 1);
    this.life   = 1.0;
    this.decay  = Math.random() * 0.03 + 0.02;
    this.radius = Math.random() * 2.5 + 0.5;
    this.color  = color;
  }
  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.vy += 0.08; // gravity
    this.vx *= 0.97;
    this.life -= this.decay;
  }
  draw(ctx) {
    ctx.save();
    ctx.globalAlpha = Math.max(0, this.life);
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.shadowBlur  = 6;
    ctx.shadowColor = this.color;
    ctx.fill();
    ctx.restore();
  }
  get dead() { return this.life <= 0; }
}

/* ─────────────────────────────────────────────
   DROP DETECTOR
   Watches for a volume spike after a quiet period
   and fires a screen flash + particle explosion.
───────────────────────────────────────────── */
class DropDetector {
  constructor() {
    // Tuning
    this.QUIET_THRESHOLD  = 0.045;  // RMS below this → considered quiet
    this.SPIKE_RATIO      = 3.2;    // fast/slow ratio needed to call a drop
    this.QUIET_MIN_MS     = 300;    // must be quiet for at least this long
    this.COOLDOWN_MS      = 800;    // min gap between consecutive drops

    this._slowRMS         = 0;      // very slow baseline tracker
    this._quietSince      = null;   // timestamp when quiet period started
    this._lastDrop        = 0;      // timestamp of last detected drop
    this._colorIdx        = 0;      // cycles through accent colors

    // Colors cycle: bass pink → mids purple → treble cyan
    this._colors = [
      { bg: 'rgba(255,45,107,0.22)',  glow: '#ff2d6b' },
      { bg: 'rgba(180,77,255,0.20)', glow: '#b44dff' },
      { bg: 'rgba(0,229,255,0.18)',  glow: '#00e5ff' },
    ];
  }

  /**
   * Call every animation frame with the current fast (smoothed) RMS.
   * Returns a drop event object {triggered, color} or null.
   */
  update(fastRMS) {
    // Extremely slow lerp for baseline — reacts only to sustained loudness
    this._slowRMS = lerp(this._slowRMS, fastRMS, 0.018);

    const now = performance.now();
    const isQuiet = fastRMS < this.QUIET_THRESHOLD;

    if (isQuiet) {
      if (this._quietSince === null) this._quietSince = now;
    } else {
      // Reset quiet timer on any loud moment
      if (!isQuiet) this._quietSince = null;
    }

    // Detect spike: loud enough, ratio high enough, was quiet long enough
    const quietDuration = this._quietSince !== null ? now - this._quietSince : 0;
    const ratio         = this._slowRMS > 0.001 ? fastRMS / this._slowRMS : 0;
    const cooledDown    = now - this._lastDrop > this.COOLDOWN_MS;

    // The spike check: quiet period ended in the PREVIOUS frame, now loud
    // We detect the transition: quietSince is null (just went loud) AND ratio is big
    if (
      this._quietSince === null &&                    // currently loud (just turned loud)
      ratio > this.SPIKE_RATIO &&
      cooledDown
    ) {
      this._lastDrop = now;
      // Determine if the preceding quiet was long enough
      // We track via a separate flag since quietSince resets when loud
      if (this._hadQuiet) {
        this._hadQuiet = false;
        const color = this._colors[this._colorIdx % this._colors.length];
        this._colorIdx++;
        return { triggered: true, color };
      }
    }

    // Track that we were quiet (for the next spike check)
    if (isQuiet && now - (this._quietSince || now) > this.QUIET_MIN_MS) {
      this._hadQuiet = true;
    }
    if (!isQuiet && fastRMS > this.QUIET_THRESHOLD * 2) {
      // loud — don't clear hadQuiet yet so the spike check can use it
    }

    return null;
  }
}


class Visualizer {
  constructor(canvas, engine) {
    this.canvas  = canvas;
    this.ctx2d   = canvas.getContext('2d');
    this.engine  = engine;

    this.barCount  = BAR_COUNT;
    this.smoothed  = new Float32Array(this.barCount); // Lerp-smoothed heights
    this.peaks     = new Float32Array(this.barCount); // Peak cap heights
    this.peakVels  = new Float32Array(this.barCount); // Falling velocities
    this.peakHold  = new Array(this.barCount).fill(0); // Timestamps

    this.particles = [];
    this._rafId    = null;
    this._running  = false;
    this._rms      = 0;   // smoothed RMS
    this._bassAvg  = 0;   // smoothed bass
    this._trebleAvg = 0;  // smoothed treble

    this._mode          = 'bars';
    this._theme         = THEMES.neon;
    this._dropDetector = new DropDetector();

    this._initResize();
  }

  _initResize() {
    // Use ResizeObserver so we always get real layout dimensions
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(this.canvas.parentElement);
    this.resize();
  }

  resize() {
    const wrapper = this.canvas.parentElement;
    const w = wrapper.clientWidth  || wrapper.offsetWidth  || 600;
    const h = wrapper.clientHeight || wrapper.offsetHeight || 300;
    this.canvas.width  = w * devicePixelRatio;
    this.canvas.height = h * devicePixelRatio;
    this.canvas.style.width  = w + 'px';
    this.canvas.style.height = h + 'px';
    // Resetting canvas dimensions clears the transform, so re-apply scale
    this.ctx2d.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }

  get W() { return this.canvas.width  / devicePixelRatio; }
  get H() { return this.canvas.height / devicePixelRatio; }

  start() {
    if (this._running) return;
    this._running = true;
    this._loop();
  }

  stop() {
    this._running = false;
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._rafId = null;
    this._clear();
  }

  setTheme(name) {
    this._theme = THEMES[name] || THEMES.neon;
    document.documentElement.setAttribute('data-theme', name);
  }

  setMode(mode) {
    this._mode = mode;
    // Reset state so old mode's data doesn't bleed into the new rendering
    this.smoothed.fill(0);
    this.peaks.fill(0);
    this.peakVels.fill(0);
    this.peakHold.fill(0);
    this.particles = [];
  }

  _loop() {
    if (!this._running) return;
    this._rafId = requestAnimationFrame(() => this._loop());
    this._frame();
  }

  _frame() {
    const W = this.W, H = this.H;
    // Bail out if canvas has no real dimensions yet
    if (!(W > 0) || !(H > 0)) return;

    const ctx = this.ctx2d;
    const freq = this.engine.getFreqData();    // Uint8[0..255] × binCount
    const rms  = this.engine.getRMS();

    // Smooth RMS
    this._rms = lerp(this._rms, rms, 0.12);

    /* Build bar heights from frequency bins
       We map bins logarithmically for a perceptual spread */
    const bins      = freq ? freq.length : 0;
    const minHz     = 20, maxHz = 18000;
    const sr        = this.engine.sampleRate;
    const fftSize   = this.engine.analyser ? this.engine.analyser.fftSize : FFT_SIZE;

    const minBin = Math.max(1, Math.floor(minHz * fftSize / sr));   // ≥1 to avoid log(0)
    const maxBin = Math.min(Math.floor(maxHz * fftSize / sr), Math.max(bins - 1, 1));

    // Accumulate band averages for UI meters
    let bassSum = 0, bassN = 0, midsSum = 0, midsN = 0, trebleSum = 0, trebleN = 0;

    const targets = new Float32Array(this.barCount);
    for (let i = 0; i < this.barCount; i++) {
      // Logarithmic bin mapping — safe even when bins=0
      const t   = this.barCount > 1 ? i / (this.barCount - 1) : 0;
      const rawBin = minBin * Math.pow(maxBin / minBin, t);
      const bin = isFinite(rawBin) ? Math.floor(rawBin) : minBin;
      const val = (freq && bins > 0) ? freq[clamp(bin, 0, bins - 1)] / 255 : 0;
      targets[i] = val;

      const hz = binToHz(bin, sr, fftSize);
      if (hz < 250)        { bassSum   += val; bassN++;   }
      else if (hz < 4000)  { midsSum   += val; midsN++;   }
      else                 { trebleSum += val; trebleN++; }
    }

    const bassAvg   = bassN   ? bassSum   / bassN   : 0;
    const midsAvg   = midsN   ? midsSum   / midsN   : 0;
    const trebleAvg = trebleN ? trebleSum / trebleN : 0;

    this._bassAvg   = lerp(this._bassAvg,   bassAvg,   0.12);
    this._trebleAvg = lerp(this._trebleAvg, trebleAvg, 0.12);

    // Expose to UI
    ui.updateBands(this._bassAvg, lerp(0, midsAvg, 0.12 + midsAvg * 0.88), this._trebleAvg);
    ui.updateRMS(this._rms);

    // ── Drop Detection ──
    const drop = this._dropDetector.update(this._rms);
    if (drop) ui.onDrop(drop, W, H, this);

    /* ── Clear with bass-driven bloom glow ── */
    ctx.clearRect(0, 0, W, H);

    const th = this._theme;
    const bloomR = this._bassAvg * 160;
    if (bloomR > 2) {
      const [r1,g1,b1] = th.bloom1;
      const [r2,g2,b2] = th.bloom2;
      const grd = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.75);
      grd.addColorStop(0,   `rgba(${r1},${g1},${b1},${this._bassAvg * 0.18})`);
      grd.addColorStop(0.5, `rgba(${r2},${g2},${b2},${this._bassAvg * 0.07})`);
      grd.addColorStop(1,   'rgba(0,0,0,0)');
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, W, H);
    }

    /* ── Lerp smoothing ── */
    for (let i = 0; i < this.barCount; i++) {
      this.smoothed[i] = lerp(this.smoothed[i], targets[i], LERP_SPEED);
    }

    /* ── Mode dispatch ── */
    const totalBars = this.barCount * 2;
    const gap       = 2;
    const barW      = Math.max(1, (W - gap * (totalBars - 1)) / totalBars);
    const baseY     = H * 0.88;
    const maxBarH   = H * 0.80;
    const glowFactor = clamp(this._rms * 10, 0, 1);

    switch (this._mode) {
      case 'bars':     this._drawBars(W, H, ctx, barW, baseY, maxBarH, glowFactor); break;
      case 'wave':     this._drawWaveform(W, H, ctx, glowFactor); break;
      case 'circular': this._drawCircular(W, H, ctx, glowFactor); break;
      case 'lissajous':this._drawLissajous(W, H, ctx, glowFactor); break;
      case 'perspective': this._drawPerspective(W, H, ctx, glowFactor); break;
      case 'oscilloscope': this._drawOscilloscope(W, H, ctx, glowFactor); break;
    }

    /* ── Update & draw particles (all modes — drop explosions work everywhere) ── */
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.update();
      if (p.dead) { this.particles.splice(i, 1); }
      else { p.draw(ctx); }
    }
  }

  /* ─────────────────────────── MODE: BARS ─────────────────────────── */
  _drawBars(W, H, ctx, barW, baseY, maxBarH, glowFactor) {
    const th  = this._theme;
    const gap = 2;
    for (let i = 0; i < this.barCount; i++) {
      const val  = this.smoothed[i];
      const barH = val * maxBarH;
      const leftIdx  = this.barCount - 1 - i;
      const rightIdx = this.barCount + i;
      const xLeft  = leftIdx  * (barW + gap);
      const xRight = rightIdx * (barW + gap);
      const y      = baseY - barH;
      const hue       = remap(i, 0, this.barCount - 1, th.hueStart, th.hueEnd);
      const sat       = th.sat;
      const glowColor = `hsl(${hue},${Math.max(sat,20)}%,70%)`;

      if (barH > 0.5 && isFinite(y) && isFinite(xLeft) && isFinite(xRight)) {
        ctx.save();
        ctx.shadowBlur  = 8 + glowFactor * 24 + val * 16;
        ctx.shadowColor = glowColor;
        const grad = ctx.createLinearGradient(0, y, 0, baseY);
        grad.addColorStop(0,   `hsl(${hue},${sat}%,${clamp(70 + val * 20, 0, 100)}%)`);
        grad.addColorStop(0.4, `hsl(${hue},${sat}%,50%)`);
        grad.addColorStop(1,   `hsl(${hue},${Math.floor(sat*0.8)}%,20%)`);
        const radius = Math.min(barW / 2, 4);
        this._roundRect(ctx, xLeft,  y, barW, barH, radius);
        ctx.fillStyle = grad;
        ctx.fill();
        this._roundRect(ctx, xRight, y, barW, barH, radius);
        ctx.fill();
        ctx.restore();
      }

      const now = performance.now();
      if (barH > this.peaks[i]) {
        this.peaks[i] = barH; this.peakHold[i] = now; this.peakVels[i] = 0;
      } else if (now - this.peakHold[i] > PEAK_HOLD_MS) {
        this.peakVels[i] = Math.min(this.peakVels[i] + PEAK_GRAVITY, 18);
        this.peaks[i]    = Math.max(0, this.peaks[i] - this.peakVels[i]);
      }
      const capY = baseY - this.peaks[i] - 3;
      if (this.peaks[i] > 2 && isFinite(capY)) {
        ctx.save();
        ctx.shadowBlur = 10 + glowFactor * 12;
        ctx.shadowColor = glowColor;
        ctx.fillStyle  = `hsl(${hue},${sat}%,82%)`;
        ctx.fillRect(xLeft,  capY, barW, 2);
        ctx.fillRect(xRight, capY, barW, 2);
        ctx.restore();
      }

      if (this._trebleAvg > 0.35 && val > 0.72 && Math.random() < 0.12) {
        const glowColor2 = `hsl(${remap(i, 0, this.barCount-1, th.hueStart, th.hueEnd)},${Math.max(th.sat,20)}%,70%)`;
        this.particles.push(new Particle(xRight + barW / 2, y, glowColor2));
        this.particles.push(new Particle(xLeft  + barW / 2, y, glowColor2));
      }
    }

    // Baseline
    ctx.save();
    ctx.globalAlpha = 0.12 + glowFactor * 0.08;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(0, baseY + 1); ctx.lineTo(W, baseY + 1);
    ctx.stroke();
    ctx.restore();
  }

  /* ────────────────────────── MODE: WAVEFORM ──────────────────────── */
  _drawWaveform(W, H, ctx, glowFactor) {
    const td = this.engine.timeDomainData;
    if (!td || td.length === 0) return;

    const th     = this._theme;
    const [c0,c1,c2,c3] = th.wave;

    const baseY  = H * 0.5;
    const scaleY = H * 0.40;
    const step   = W / (td.length - 1);

    // Horizontal center line
    ctx.save();
    ctx.globalAlpha = 0.08;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(0, baseY); ctx.lineTo(W, baseY);
    ctx.stroke();
    ctx.restore();

    // Glow pass (wide, blurred)
    const hGrad = ctx.createLinearGradient(0, 0, W, 0);
    hGrad.addColorStop(0,    c0);
    hGrad.addColorStop(0.33, c1);
    hGrad.addColorStop(0.66, c2);
    hGrad.addColorStop(1,    c3);

    const drawLine = (alpha, blur, lineW) => {
      ctx.save();
      ctx.globalAlpha  = alpha;
      ctx.shadowBlur   = blur;
      ctx.shadowColor  = `rgba(180,77,255,0.9)`;
      ctx.lineWidth    = lineW;
      ctx.lineCap      = 'round';
      ctx.lineJoin     = 'round';
      ctx.strokeStyle  = hGrad;
      ctx.beginPath();
      for (let i = 0; i < td.length; i++) {
        const x = i * step;
        const y = baseY + td[i] * scaleY;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    drawLine(0.25 + glowFactor * 0.2, 18 + glowFactor * 20, 6);
    drawLine(1.0,                      8 + glowFactor * 12,  2);

    // Soft mirror reflection below
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.shadowBlur  = 6;
    ctx.shadowColor = '#7c3bff';
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = 'round';
    ctx.strokeStyle = hGrad;
    ctx.beginPath();
    const reflectY = H * 0.88;
    for (let i = 0; i < td.length; i++) {
      const x = i * step;
      const y = reflectY - td[i] * H * 0.06;  // subtle shallow reflection
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ─────────────────────────── MODE: CIRCULAR ─────────────────────── */
  _drawCircular(W, H, ctx, glowFactor) {
    const th      = this._theme;
    const cx      = W / 2;
    const cy      = H * 0.48;
    const minDim  = Math.min(W, H);
    const innerR  = minDim * 0.16;
    const maxLen  = minDim * 0.34;
    const total   = this.barCount * 2;

    for (let i = 0; i < this.barCount; i++) {
      const val    = this.smoothed[i];
      const barLen = val * maxLen;
      if (barLen < 0.5) continue;

      const hue       = remap(i, 0, this.barCount - 1, th.hueStart, th.hueEnd);
      const sat       = th.sat;
      const glowColor = `hsl(${hue},${Math.max(sat,20)}%,70%)`;
      const lineW     = Math.max(1.5, 2.5 - (i / this.barCount) * 1.2);

      for (let side = 0; side < 2; side++) {
        const idx   = side === 0 ? i : (this.barCount * 2 - 1 - i);
        const angle = (idx / total) * Math.PI * 2 - Math.PI / 2;
        const x1 = cx + Math.cos(angle) * innerR;
        const y1 = cy + Math.sin(angle) * innerR;
        const x2 = cx + Math.cos(angle) * (innerR + barLen);
        const y2 = cy + Math.sin(angle) * (innerR + barLen);

        ctx.save();
        ctx.shadowBlur  = 6 + glowFactor * 16 + val * 10;
        ctx.shadowColor = glowColor;
        ctx.strokeStyle = `hsl(${hue},${sat}%,62%)`;
        ctx.lineWidth   = lineW;
        ctx.lineCap     = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
      }
    }

    // Inner glowing core circle
    ctx.save();
    const [r1,g1,b1] = th.bloom2;
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, innerR);
    coreGrad.addColorStop(0,   `rgba(${r1},${g1},${b1},${0.40 + glowFactor * 0.45})`);
    coreGrad.addColorStop(0.6, `rgba(${r1},${g1},${b1},${0.12 + glowFactor * 0.15})`);
    coreGrad.addColorStop(1,   `rgba(${r1},${g1},${b1},0.02)`);
    ctx.beginPath();
    ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.fill();
    ctx.shadowBlur  = 12 + glowFactor * 18;
    ctx.shadowColor = th.core;
    ctx.strokeStyle = `rgba(${r1},${g1},${b1},${0.45 + glowFactor * 0.4})`;
    ctx.lineWidth   = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  /* ────────────────────────── MODE: LISSAJOUS ─────────────────────── */
  _drawLissajous(W, H, ctx, glowFactor) {
    const td = this.engine.timeDomainData;
    if (!td || td.length === 0) return;

    const th     = this._theme;
    const cx     = W / 2;
    const cy     = H / 2;
    const scale  = Math.min(W, H) * 0.42;
    const len    = td.length;
    const offset = Math.floor(len / 4);

    // Subtle crosshairs
    ctx.save();
    ctx.globalAlpha = 0.07;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1;
    ctx.setLineDash([4, 8]);
    ctx.beginPath();
    ctx.moveTo(cx, H * 0.08); ctx.lineTo(cx, H * 0.92);
    ctx.moveTo(W * 0.08, cy); ctx.lineTo(W * 0.92, cy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Plot phase-shifted XY pairs as glowing dots
    ctx.save();
    for (let i = 0; i < len - offset; i++) {
      const x = cx + td[i]          * scale;
      const y = cy + td[i + offset] * scale;
      if (!isFinite(x) || !isFinite(y)) continue;

      const t     = i / len;
      const hue   = th.hueStart + t * (th.hueEnd - th.hueStart);
      const sat   = th.sat;
      const amp   = Math.abs(td[i]) + Math.abs(td[i + offset]);
      const alpha = clamp(0.3 + glowFactor * 0.5 + amp * 0.4, 0, 1);
      const size  = 1.2 + amp * 1.4;

      ctx.shadowBlur  = 4 + glowFactor * 8;
      ctx.shadowColor = `hsl(${hue},${Math.max(sat,20)}%,70%)`;
      ctx.fillStyle   = `hsla(${hue},${sat}%,72%,${alpha})`;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    }
    ctx.restore();

    // Unit circle guide
    ctx.save();
    ctx.globalAlpha = 0.06 + glowFactor * 0.05;
    ctx.strokeStyle = th.core;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  /* ──────────────────────── MODE: PERSPECTIVE ────────────────────── */
  _drawPerspective(W, H, ctx, glowFactor) {
    const th = this._theme;
    const cx = W / 2;
    const cy = H * 0.45; // vanishing point
    const perspective = 0.85;

    // Receding Floor Grid
    ctx.save();
    ctx.strokeStyle = `rgba(${th.bloom2.join(',')}, ${0.1 + glowFactor * 0.1})`;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const z = i / 10;
      const y = cy + (H - cy) * Math.pow(z, 2);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
    for (let i = -5; i <= 5; i++) {
        ctx.beginPath();
        ctx.moveTo(cx + i * W * 0.2, H);
        ctx.lineTo(cx, cy);
        ctx.stroke();
    }
    ctx.restore();

    // Bars with perspective
    const numBars = this.barCount;
    for (let i = 0; i < numBars; i++) {
      const val = this.smoothed[i];
      const hue = remap(i, 0, numBars - 1, th.hueStart, th.hueEnd);
      const sat = th.sat;
      
      // Mirror halves
      for (let side = -1; side <= 1; side += 2) {
        if (side === 0) continue;
        
        const xOffset = (i / numBars) * W * 0.45 * side;
        const xBase = cx + xOffset;
        const z = 1.0; // depth (fixed for this "row", but could be varied)
        
        // Simulating depth by scaling based on xOffset
        const scale = 1.0 - Math.abs(xOffset / cx) * 0.5;
        const barW = (W / numBars) * 1.2 * scale;
        const barH = val * H * 0.6 * scale;
      
        const x = cx + xOffset;
        const y = H * 0.9 - (1.0 - scale) * H * 0.2;
        
        if (barH > 0.5 && isFinite(x) && isFinite(y)) {
          const glowColor = `hsl(${hue},${Math.max(sat,20)}%,70%)`;
          ctx.save();
          ctx.shadowBlur = (4 + glowFactor * 10) * scale;
          ctx.shadowColor = glowColor;

          const grad = ctx.createLinearGradient(x, y - barH, x, y);
          grad.addColorStop(0, `hsl(${hue},${sat}%,${70 + val * 20}%)`);
          grad.addColorStop(1, `hsl(${hue},${sat}%,20%)`);

          ctx.fillStyle = grad;
          // Draw a slanted "3D" bar (trapezoid)
          const topW = barW * 0.8;
          ctx.beginPath();
          ctx.moveTo(x - barW/2, y);
          ctx.lineTo(x + barW/2, y);
          ctx.lineTo(cx + (xOffset + barW/2) * perspective, y - barH);
          ctx.lineTo(cx + (xOffset - barW/2) * perspective, y - barH);
          ctx.closePath();
          ctx.fill();
          
          ctx.restore();
        }
      }
    }
  }

  /* ──────────────────────── MODE: OSCILLOSCOPE ───────────────────── */
  _drawOscilloscope(W, H, ctx, glowFactor) {
    const td = this.engine.timeDomainData;
    if (!td || td.length === 0) return;

    const cx = W / 2;
    const cy = H / 2;

    // Trigger logic (find zero-crossing with positive slope)
    let triggerIdx = 0;
    const triggerThreshold = 0.01;
    for (let i = 1; i < td.length / 2; i++) {
        if (td[i] > 0 && td[i-1] <= 0 && td[i] > triggerThreshold) {
            triggerIdx = i;
            break;
        }
    }

    // Grid background
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.08)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x <= W; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y <= H; y += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    // Main Axis
    ctx.strokeStyle = 'rgba(0, 255, 65, 0.2)';
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    ctx.restore();

    // Waveform drawing
    const phosphorGreen = '#39ff14';
    const numPoints = Math.min(td.length - triggerIdx, 1024);
    const step = W / (numPoints - 1);
    const scaleY = H * 0.45;

    const drawWave = (alpha, blur, width) => {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.shadowBlur = blur;
        ctx.shadowColor = phosphorGreen;
        ctx.strokeStyle = phosphorGreen;
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
            const x = i * step;
            const y = cy + td[triggerIdx + i] * scaleY;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
    };

    // Layered glow for phosphor look
    drawWave(0.3, 20 + glowFactor * 15, 6);
    drawWave(1.0, 4 + glowFactor * 6, 2);

    // Scanline effect hint
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.05)';
    for(let i=0; i<H; i+=4) {
        ctx.fillRect(0, i, W, 1);
    }
    ctx.restore();
  }

  _clear() {
    const ctx = this.ctx2d;
    ctx.clearRect(0, 0, this.W, this.H);
  }

  _roundRect(ctx, x, y, w, h, r) {
    if (h < r * 2) r = h / 2;
    if (h <= 0) return;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }
}

/* ─────────────────────────────────────────────
   UI CONTROLLER
───────────────────────────────────────────── */
class UIController {
  constructor() {
    this.engine     = new AudioEngine();
    this.canvas     = document.getElementById('vizCanvas');
    this.visualizer = new Visualizer(this.canvas, this.engine);

    this._btnPlay   = document.getElementById('btnPlay');
    this._btnStop   = document.getElementById('btnStop');
    this._volumeSlider = document.getElementById('volumeSlider');
    this._fileInput = document.getElementById('fileInput');
    this._uploadZone = document.getElementById('uploadZone');
    this._idleMsg   = document.getElementById('idleMsg');
    this._playIcon  = document.getElementById('playIcon');

    this._fillBass   = document.getElementById('fillBass');
    this._fillMids   = document.getElementById('fillMids');
    this._fillTreble = document.getElementById('fillTreble');
    this._rmsFill    = document.getElementById('rmsFill');
    this._rmsValue   = document.getElementById('rmsValue');

    this._wrapper    = document.getElementById('visualizer-wrapper');
    this._demoBtns   = document.querySelectorAll('.btn-demo');

    // Background Video
    this._bgVideo         = document.getElementById('bgVideo');
    this._bgVideoInput    = document.getElementById('bgVideoInput');
    this._bgVideoControls = document.getElementById('bgVideoControls');
    this._bgVideoOpacity  = document.getElementById('bgVideoOpacity');
    this._bgVideoRemove   = document.getElementById('bgVideoRemove');
    this._bgVideoURL      = null;

    // Equalizer
    this._eqSliders  = document.querySelectorAll('.eq-slider');
    this._btnResetEQ = document.getElementById('btnResetEQ');

    // Mic
    this._btnMic = document.getElementById('btnMic');

    // Help
    this._helpSection  = document.querySelector('.help-section');
    this._btnHelp      = document.getElementById('btnHelp');
    this._btnCloseHelp = document.getElementById('btnCloseHelp');

    // Seek Bar
    this._seekBar     = document.getElementById('seekBar');
    this._seekFill    = document.getElementById('seekFill');
    this._timeCurrent = document.getElementById('timeCurrent');
    this._timeTotal   = document.getElementById('timeTotal');
    this._isScrubbing = false;

    this._ready      = false; // true once a source is loaded
    this._isPlaying  = false;
    this._activeDemo = null;

    this._bindEvents();
    this._bindKeyboard();
    this._startIdleLoop();
  }

  _bindEvents() {
    /* File upload */
    this._fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) this._loadFile(file);
    });

    /* Drag-and-drop */
    this._uploadZone.addEventListener('dragover', e => {
      e.preventDefault();
      this._uploadZone.style.borderColor = 'var(--treble)';
    });
    this._uploadZone.addEventListener('dragleave', () => {
      this._uploadZone.style.borderColor = '';
    });
    this._uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      this._uploadZone.style.borderColor = '';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('audio/')) this._loadFile(file);
    });
    this._uploadZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') this._fileInput.click();
    });

    /* Play / Pause */
    this._btnPlay.addEventListener('click', () => {
      if (!this._ready) return;
      if (this._isPlaying) this._pause();
      else this.play();
    });

    /* Stop */
    this._btnStop.addEventListener('click', () => {
      this._stop();
    });

    /* Volume */
    this._volumeSlider.addEventListener('input', e => {
      this.engine.setVolume(parseFloat(e.target.value));
    });

    /* Equalizer */
    this._eqSliders.forEach(slider => {
      slider.addEventListener('input', e => {
        const freq = parseInt(slider.dataset.freq);
        const gain = parseFloat(e.target.value);
        this.engine.setEqGain(freq, gain);
      });
    });

    this._btnResetEQ.addEventListener('click', () => {
      this._eqSliders.forEach(slider => {
        slider.value = 0;
        const freq = parseInt(slider.dataset.freq);
        this.engine.setEqGain(freq, 0);
      });
    });

    /* Seek Bar */
    this._seekBar.addEventListener('input', e => {
      this._isScrubbing = true;
      const val = parseFloat(e.target.value);
      const time = (val / 100) * this.engine.duration;
      this._timeCurrent.textContent = this._formatTime(time);
      this._seekFill.style.width = val + '%';
    });

    this._seekBar.addEventListener('change', e => {
      this._isScrubbing = false;
      const val = parseFloat(e.target.value);
      const time = (val / 100) * this.engine.duration;
      this.play(time); // Use the refactored UIController.play()
    });

    /* Demo tracks */
    this._demoBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.freq;
        this._clearDemoBtns();
        this._onMicStop(); // Disable mic if active
        btn.classList.add('active');
        this._activeDemo = mode;
        this._ready = true;
        this.engine.startDemoMode(mode);
        this._onPlay();
      });
    });

    /* Mic */
    this._btnMic.addEventListener('click', () => {
      this._toggleMic();
    });

    /* Help */
    this._btnHelp.addEventListener('click', (e) => {
      e.stopPropagation();
      this._helpSection.classList.add('active');
    });
    this._btnCloseHelp.addEventListener('click', () => {
      this._helpSection.classList.remove('active');
    });
    document.addEventListener('click', (e) => {
      if (!this._helpSection.contains(e.target)) {
        this._helpSection.classList.remove('active');
      }
    });

    /* Resize */
    window.addEventListener('resize', () => {
      this.visualizer.resize();
    });

    /* Background Video */
    this._bgVideoInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) this._loadBgVideo(file);
    });

    this._bgVideoOpacity.addEventListener('input', e => {
      this._bgVideo.style.opacity = e.target.value / 100;
    });

    this._bgVideoRemove.addEventListener('click', () => {
      this._removeBgVideo();
    });

    /* Theme picker */
    document.querySelectorAll('.theme-dot').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.theme-dot').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.visualizer.setTheme(btn.dataset.theme);
      });
    });

    /* Mode switcher */
    document.querySelectorAll('.btn-mode').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-mode').forEach(b => {
          b.classList.remove('active');
          b.setAttribute('aria-selected', 'false');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        this.visualizer.setMode(btn.dataset.mode);
      });
    });
  }

  /* ── Microphone ── */
  async startMic() {
    this._ensureContext();
    this.stop(); // Stop everything before starting Mic

    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.micNode = this.ctx.createMediaStreamSource(this.micStream);
      
      // Connect to start of EQ chain
      this.micNode.connect(this.filters[0]);
      
      this.isPlaying = true; // For Visualizer loop
      return true;
    } catch (err) {
      console.error('Microphone error:', err);
      return false;
    }
  }

  stopMic() {
    if (this.micNode) {
      this.micNode.disconnect();
      this.micNode = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
  }

  async _loadFile(file) {
    try {
      this._updateUploadLabel('Loading…');
      await this.engine.loadFile(file);
      this._clearDemoBtns();
      this._activeDemo = null;
      this._ready = true;
      this._btnPlay.disabled  = false;
      this._btnStop.disabled  = false;
      const name = file.name.replace(/\.[^/.]+$/, '').slice(0, 28);
      this._updateUploadLabel('♬ ' + name);
      // Auto-play
      this.play();
    } catch (err) {
      console.error('Audio decode error:', err);
      this._updateUploadLabel('⚠ Error — try another file');
    }
  }

  /* ── Background Video ── */
  _loadBgVideo(file) {
    if (this._bgVideoURL) URL.revokeObjectURL(this._bgVideoURL);
    this._bgVideoURL = URL.createObjectURL(file);
    this._bgVideo.src = this._bgVideoURL;
    this._bgVideo.hidden = false;
    this._bgVideoControls.hidden = false;
    this._bgVideo.play().catch(e => console.warn('Video auto-play failed:', e));
  }

  _removeBgVideo() {
    if (this._bgVideoURL) URL.revokeObjectURL(this._bgVideoURL);
    this._bgVideoURL = null;
    this._bgVideo.src = '';
    this._bgVideo.hidden = true;
    this._bgVideoControls.hidden = true;
    this._bgVideoInput.value = '';
  }

  _updateUploadLabel(text) {
    this._uploadZone.querySelector('.upload-text').textContent = text;
  }

  play(offset = null) {
    this.engine.play(offset);
    this._onPlay();
  }

  _pause() {
    this.engine.pause();
    this._onPause();
  }

  _stop() {
    this.engine.stop();
    this._onMicStop();
    this._clearDemoBtns();
    this._activeDemo = null;
    this._isPlaying  = false;
    this._ready      = false;
    this._btnPlay.disabled = true;
    this._btnStop.disabled = true;
    this._playIcon.textContent = '▶';
    this.visualizer.stop();
    this._idleMsg.style.opacity = '1';
    this._idleMsg.style.pointerEvents = 'auto';
    this._wrapper.style.boxShadow = '';
    this._updateUploadLabel('Upload MP3');
    this._fileInput.value = '';
    this.updateBands(0, 0, 0);
    this.updateRMS(0); // Changed from this._updateRMS() to updateRMS(0) to match existing public method
    this._updateSeekBar();
  }

  _updateSeekBar() {
    if (!this.engine.isPlaying || this._isScrubbing) return;
    const cur = this.engine.currentTime;
    const dur = this.engine.duration;
    if (dur > 0) {
      const per = (cur / dur) * 100;
      this._seekBar.value = per;
      this._seekFill.style.width = per + '%';
      this._timeCurrent.textContent = this._formatTime(cur);
      this._timeTotal.textContent = this._formatTime(dur);
    }
  }

  _formatTime(s) {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  _onPlay() {
    this._isPlaying = true;
    this._btnPlay.disabled = false;
    this._btnStop.disabled = false;
    this._playIcon.textContent = '⏸';
    this._idleMsg.style.opacity = '0';
    this._idleMsg.style.pointerEvents = 'none';
    this.visualizer.start();
  }

  _onPause() {
    this._isPlaying = false;
    this._playIcon.textContent = '▶';
    this.visualizer.stop();
  }

  onEnded() {
    this._isPlaying = false;
    this._playIcon.textContent = '▶';
    setTimeout(() => {
      if (!this._isPlaying) this.visualizer.stop();
    }, 500);
  }

  /* Band meter + RMS updates (called from Visualizer._frame) */
  updateBands(bass, mids, treble) {
    this._fillBass.style.width   = (bass   * 100).toFixed(1) + '%';
    this._fillMids.style.width   = (mids   * 100).toFixed(1) + '%';
    this._fillTreble.style.width = (treble * 100).toFixed(1) + '%';

    // Wrapper bloom glow from bass
    const glow = clamp(bass * 2, 0, 1);
    if (glow > 0.05) {
      this._wrapper.style.boxShadow =
        `0 0 ${20 + glow * 60}px rgba(255,45,107,${glow * 0.35}),
         0 0 ${40 + glow * 80}px rgba(124,59,255,${glow * 0.18}),
         0 24px 64px rgba(0,0,0,0.6)`;
    } else {
      this._wrapper.style.boxShadow = '0 24px 64px rgba(0,0,0,0.6)';
    }
  }

  updateRMS(rms) {
    const db  = rms > 0 ? 20 * Math.log10(rms) : -80;
    const pct = clamp(remap(db, -60, 0, 0, 100), 0, 100);
    this._rmsFill.style.width = pct.toFixed(1) + '%';
    this._rmsValue.textContent = isFinite(db) ? db.toFixed(1) + ' dB' : '−∞ dB';
  }

  /* ── Drop event (called by Visualizer._frame) ── */
  onDrop(drop, W, H, viz) {
    // 1) Screen flash
    const el = document.getElementById('dropFlash');
    el.style.setProperty('--drop-color', drop.color.bg);
    // Force reflow to restart animation if it was already running
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');

    // 2) Radial particle explosion from canvas center
    const cx = W / 2;
    const cy = H * 0.88;           // burst from the baseline — feels like the drop "hits" the ground
    const count = 80;
    for (let i = 0; i < count; i++) {
      const angle  = (i / count) * Math.PI * 2;
      const speed  = 3 + Math.random() * 7;
      const p      = new Particle(cx, cy, drop.color.glow);
      p.vx = Math.cos(angle) * speed;
      p.vy = Math.sin(angle) * speed - 4;  // bias upward
      p.life   = 1.0;
      p.decay  = 0.012 + Math.random() * 0.018;
      p.radius = 1.5 + Math.random() * 3.5;
      viz.particles.push(p);
    }

    // 3) Extra upward fountain burst for drama
    for (let i = 0; i < 30; i++) {
      const p = new Particle(
        cx + (Math.random() - 0.5) * 60,
        cy,
        drop.color.glow
      );
      p.vx = (Math.random() - 0.5) * 5;
      p.vy = -(4 + Math.random() * 9);
      p.life   = 1.0;
      p.decay  = 0.015 + Math.random() * 0.02;
      p.radius = 1 + Math.random() * 2.5;
      viz.particles.push(p);
    }
  }

  _clearDemoBtns() {
    this._demoBtns.forEach(b => b.classList.remove('active'));
  }

  /* Idle animation — subtle waving bars even before audio */
  _startIdleLoop() {
    const canvas = this.canvas;
    const ctx    = canvas.getContext('2d');
    let t = 0;
    const idle = () => {
      if (this._isPlaying) { requestAnimationFrame(idle); return; }
      const W = this.visualizer.W, H = this.visualizer.H;
      ctx.clearRect(0, 0, W, H);
      const bars   = 60;
      const barW   = (W - bars * 2) / bars;
      const baseY  = H * 0.88;
      const maxBH  = H * 0.06;
      t += 0.025;
      for (let i = 0; i < bars; i++) {
        const val  = Math.sin(t + i * 0.18) * 0.5 + 0.5;
        const bh   = val * maxBH;
        const hue  = remap(i, 0, bars - 1, 330, 180);
        const x    = i * (barW + 2);
        ctx.save();
        ctx.globalAlpha = 0.28;
        ctx.fillStyle   = `hsl(${hue},80%,60%)`;
        ctx.fillRect(x, baseY - bh, barW, bh);
        ctx.restore();
      }
      requestAnimationFrame(idle);
    };
    requestAnimationFrame(idle);
  }

  /* ── Microphone methods ── */
  async _toggleMic() {
    const isActive = this._btnMic.classList.contains('active');
    
    if (isActive) {
      this._onMicStop();
      this.engine.stopMic();
    } else {
      const success = await this.engine.startMic();
      if (success) {
        this._onMicStart();
      } else {
        alert('Could not access microphone. Please check permissions.');
      }
    }
  }

  _onMicStart() {
    this._clearDemoBtns();
    this._isPlaying = true;
    this._btnMic.classList.add('active');
    
    // Disable file-playback controls
    this._btnPlay.disabled = true;
    this._playIcon.textContent = '▶';
    this._updateUploadLabel('● LIVE MICROPHONE');
    this._uploadZone.style.pointerEvents = 'none';
    this._uploadZone.style.opacity = '0.5';
    
    // Hide seek bar (not used for mic)
    document.querySelector('.seek-container').style.opacity = '0.3';
    document.querySelector('.seek-container').style.pointerEvents = 'none';
    
    this.visualizer.start();
  }

  _onMicStop() {
    this._isPlaying = false;
    this._btnMic.classList.remove('active');
    
    // Enable file-playback controls
    this._btnPlay.disabled = !this._ready;
    this._uploadZone.style.pointerEvents = 'auto';
    this._uploadZone.style.opacity = '1';
    if (!this._ready) this._updateUploadLabel('Upload MP3');
    else this._updateUploadLabel('♬ ' + this._currentFileName);
    
    // Restore seek bar
    document.querySelector('.seek-container').style.opacity = '1';
    document.querySelector('.seek-container').style.pointerEvents = 'auto';
  }

  /* ── Keyboard Shortcuts ── */
  _bindKeyboard() {
    window.addEventListener('keydown', (e) => {
      // Ignore if typing in an input (though we don't have many)
      if (e.target.tagName === 'INPUT') return;

      const key = e.key.toLowerCase();

      // Space -> Toggle Play/Pause
      if (e.code === 'Space') {
        e.preventDefault();
        if (this._ready) this.play();
      }

      // S -> Stop
      if (key === 's') {
        this._stop();
      }

      // M -> Toggle Mic
      if (key === 'm') {
        this._toggleMic();
      }

      // T -> Cycle Theme
      if (key === 't') {
        this._cycleTheme();
      }

      // 1-7 -> Modes
      if (key >= '1' && key <= '7') {
        const modes = ['bars', 'wave', 'radial', 'lissajous', '3d', 'oscilloscope', 'retro'];
        const mode = modes[parseInt(key) - 1];
        if (mode) this._setMode(mode);
      }

      // Arrows -> Volume
      if (e.code === 'ArrowUp') {
        e.preventDefault();
        this._adjustVolume(0.1);
      }
      if (e.code === 'ArrowDown') {
        e.preventDefault();
        this._adjustVolume(-0.1);
      }
    });
  }

  _cycleTheme() {
    const themeNames = Object.keys(THEMES);
    const currentIdx = themeNames.findIndex(name => THEMES[name] === this.visualizer._theme);
    const nextIdx = (currentIdx + 1) % themeNames.length;
    this._setTheme(themeNames[nextIdx]);
  }

  _setMode(mode) {
    this.visualizer._mode = mode;
    this._modeBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === mode);
    });
  }

  _setTheme(themeName) {
    const theme = THEMES[themeName];
    if (theme) {
      this.visualizer._theme = theme;
      this._themeDots.forEach(dot => {
        dot.classList.toggle('active', dot.dataset.theme === themeName);
      });
    }
  }

  _adjustVolume(delta) {
    let vol = parseFloat(this._volumeSlider.value);
    vol = Math.max(0, Math.min(1.5, vol + delta));
    this._volumeSlider.value = vol;
    this.engine.setVolume(vol);
  }
}

/* ─────────────────────────────────────────────
   BOOT
───────────────────────────────────────────── */
let ui;
document.addEventListener('DOMContentLoaded', () => {
  ui = new UIController();
});
