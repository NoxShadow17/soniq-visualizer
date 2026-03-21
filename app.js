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
    this.buffer     = null;

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

  play() {
    if (!this.ctx) return;
    if (this.ctx.state === 'suspended') this.ctx.resume();
    if (this.isPlaying) return;

    if (this._demoMode) {
      this._startDemo(this._demoMode);
      return;
    }
    if (!this.buffer) return;

    this._stopSource();
    this.source = this.ctx.createBufferSource();
    this.source.buffer = this.buffer;
    this.source.connect(this.gainNode);
    this.source.loop = false;
    const offset = this.pausedAt;
    this.source.start(0, offset);
    this.startedAt = this.ctx.currentTime - offset;
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
    this.isPlaying = false;
    this.pausedAt  = 0;
    this.startedAt = 0;
  }

  _stopSource() {
    if (this.source) {
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
      g.connect(this.gainNode);
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

  get sampleRate()     { return this.ctx ? this.ctx.sampleRate : 44100; }
  get binCount()       { return this.analyser ? this.analyser.frequencyBinCount : 0; }
  get timeDomainData() { return this._timeData; }  // already current after getRMS()
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

    this._mode          = 'bars'; // 'bars' | 'wave' | 'circular' | 'lissajous'
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

    // Vignette + bloom background
    const bloomR = this._bassAvg * 160;
    if (bloomR > 2) {
      const grd = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.75);
      grd.addColorStop(0, `rgba(255,45,107,${this._bassAvg * 0.18})`);
      grd.addColorStop(0.5, `rgba(124,59,255,${this._bassAvg * 0.07})`);
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
    const gap = 2;
    for (let i = 0; i < this.barCount; i++) {
      const val  = this.smoothed[i];
      const barH = val * maxBarH;
      const leftIdx  = this.barCount - 1 - i;
      const rightIdx = this.barCount + i;
      const xLeft  = leftIdx  * (barW + gap);
      const xRight = rightIdx * (barW + gap);
      const y      = baseY - barH;
      const hue       = remap(i, 0, this.barCount - 1, 330, 180);
      const glowColor = `hsl(${hue},100%,70%)`;

      if (barH > 0.5 && isFinite(y) && isFinite(xLeft) && isFinite(xRight)) {
        ctx.save();
        ctx.shadowBlur  = 8 + glowFactor * 24 + val * 16;
        ctx.shadowColor = glowColor;
        const grad = ctx.createLinearGradient(0, y, 0, baseY);
        grad.addColorStop(0,   `hsl(${hue},100%,${clamp(70 + val * 20, 0, 100)}%)`);
        grad.addColorStop(0.4, `hsl(${hue},90%,50%)`);
        grad.addColorStop(1,   `hsl(${hue},70%,20%)`);
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
        ctx.fillStyle  = `hsl(${hue},100%,82%)`;
        ctx.fillRect(xLeft,  capY, barW, 2);
        ctx.fillRect(xRight, capY, barW, 2);
        ctx.restore();
      }

      if (this._trebleAvg > 0.35 && val > 0.72 && Math.random() < 0.12) {
        this.particles.push(new Particle(xRight + barW / 2, y, glowColor));
        this.particles.push(new Particle(xLeft  + barW / 2, y, glowColor));
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
    hGrad.addColorStop(0,    '#ff2d6b');
    hGrad.addColorStop(0.33, '#b44dff');
    hGrad.addColorStop(0.66, '#7c3bff');
    hGrad.addColorStop(1,    '#00e5ff');

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
    const cx      = W / 2;
    const cy      = H * 0.48;
    const minDim  = Math.min(W, H);
    const innerR  = minDim * 0.16;
    const maxLen  = minDim * 0.34;
    const total   = this.barCount * 2; // full 360° using both mirror halves

    for (let i = 0; i < this.barCount; i++) {
      const val    = this.smoothed[i];
      const barLen = val * maxLen;
      if (barLen < 0.5) continue;

      const hue       = remap(i, 0, this.barCount - 1, 330, 180);
      const glowColor = `hsl(${hue},100%,70%)`;
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
        ctx.strokeStyle = `hsl(${hue},90%,62%)`;
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
    const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, innerR);
    coreGrad.addColorStop(0,   `rgba(180,77,255,${0.35 + glowFactor * 0.45})`);
    coreGrad.addColorStop(0.6, `rgba(124,59,255,${0.12 + glowFactor * 0.15})`);
    coreGrad.addColorStop(1,   'rgba(124,59,255,0.02)');
    ctx.beginPath();
    ctx.arc(cx, cy, innerR, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.fill();
    // Ring stroke
    ctx.shadowBlur  = 12 + glowFactor * 18;
    ctx.shadowColor = '#b44dff';
    ctx.strokeStyle = `rgba(180,77,255,${0.4 + glowFactor * 0.4})`;
    ctx.lineWidth   = 1.5;
    ctx.stroke();
    ctx.restore();
  }

  /* ────────────────────────── MODE: LISSAJOUS ─────────────────────── */
  _drawLissajous(W, H, ctx, glowFactor) {
    const td = this.engine.timeDomainData;
    if (!td || td.length === 0) return;

    const cx     = W / 2;
    const cy     = H / 2;
    const scale  = Math.min(W, H) * 0.42;
    const len    = td.length;
    const offset = Math.floor(len / 4); // quarter-buffer phase shift

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
      const hue   = remap(t, 0, 1, 200, 360);
      const amp   = Math.abs(td[i]) + Math.abs(td[i + offset]);
      const alpha = clamp(0.3 + glowFactor * 0.5 + amp * 0.4, 0, 1);
      const size  = 1.2 + amp * 1.4;

      ctx.shadowBlur  = 4 + glowFactor * 8;
      ctx.shadowColor = `hsl(${hue},100%,70%)`;
      ctx.fillStyle   = `hsla(${hue},90%,72%,${alpha})`;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    }
    ctx.restore();

    // Unit circle guide (subtle)
    ctx.save();
    ctx.globalAlpha = 0.06 + glowFactor * 0.05;
    ctx.strokeStyle = '#b44dff';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.stroke();
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

    this._ready      = false; // true once a source is loaded
    this._isPlaying  = false;
    this._activeDemo = null;

    this._bindEvents();
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
      else this._play();
    });

    /* Stop */
    this._btnStop.addEventListener('click', () => {
      this._stop();
    });

    /* Demo tracks */
    this._demoBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.freq;
        this._clearDemoBtns();
        btn.classList.add('active');
        this._activeDemo = mode;
        this._ready = true;
        this.engine.startDemoMode(mode);
        this._onPlay();
      });
    });

    /* Resize */
    window.addEventListener('resize', () => {
      this.visualizer.resize();
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
      this._play();
    } catch (err) {
      console.error('Audio decode error:', err);
      this._updateUploadLabel('⚠ Error — try another file');
    }
  }

  _updateUploadLabel(text) {
    this._uploadZone.querySelector('.upload-text').textContent = text;
  }

  _play() {
    this.engine.play();
    this._onPlay();
  }

  _pause() {
    this.engine.pause();
    this._onPause();
  }

  _stop() {
    this.engine.stop();
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
    this.updateRMS(0);
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
}

/* ─────────────────────────────────────────────
   BOOT
───────────────────────────────────────────── */
let ui;
document.addEventListener('DOMContentLoaded', () => {
  ui = new UIController();
});
