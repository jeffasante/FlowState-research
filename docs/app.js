/**
 * FlowState - Neural Audio Transition Engine
 * app.js - The Brain + Muscle
 */

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    MODEL_HEAVY: './model/flowstate_brain_v2.onnx',
    MODEL_LIGHT: './model/flowstate_brain_v2_quantized.onnx',
    SAMPLE_RATE: 22050, 
    MEL_BANDS: 128,
    SPEC_WIDTH: 128,
    TARGET_SAMPLES: 65536,
};

// ============================================
// STATE
// ============================================
const state = {
    audioContext: null,
    modelSession: null,
    modelType: 'none', 
    trackA: { buffer: null, filename: null, classification: null, confidence: 85 },
    trackB: { buffer: null, filename: null, classification: null, confidence: 85 },
    
    // Buffers
    bufferCut: null,
    bufferFade: null,
    bufferAI: null,
    outputBuffer: null, 
    
    isProcessing: false,
    isPlaying: false,
    currentSource: null,
    playheadAnimationId: null,
    playStartTime: 0,
    
    // Playback State
    activeBtn: null,
    activeCanvas: null,
    activeBuffer: null,
    baseImageData: null,
    
    // Controls
    transitionDuration: 4.0,
    filterIntensity: 80
};

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {};

function cacheElements() {
    // Status
    elements.modelStatusText = document.getElementById('modelStatusText');
    elements.modelStatusDot = document.getElementById('modelStatusDot');
    
    // Track A
    elements.loadTrackA = document.getElementById('loadTrackA');
    elements.inputA = document.getElementById('inputA');
    elements.waveformA = document.getElementById('waveformA');
    elements.filenameA = document.getElementById('filenameA');
    elements.classA = document.getElementById('classA');
    elements.trackAPanel = document.getElementById('trackA');
    elements.previewA = document.querySelector('#previewA .play-overlay');
    
    // Track B
    elements.loadTrackB = document.getElementById('loadTrackB');
    elements.inputB = document.getElementById('inputB');
    elements.waveformB = document.getElementById('waveformB');
    elements.filenameB = document.getElementById('filenameB');
    elements.classB = document.getElementById('classB');
    elements.trackBPanel = document.getElementById('trackB');
    elements.previewB = document.querySelector('#previewB .play-overlay');
    
    // Controls
    elements.processBtn = document.getElementById('processBtn');
    elements.playBtn = document.getElementById('playBtn');
    elements.downloadBtn = document.getElementById('downloadBtn');
    elements.exportLogsBtn = document.getElementById('exportLogsBtn');
    elements.durationDisplay = document.getElementById('durationDisplay');
    
    // Visualization
    elements.spectrogram = document.getElementById('spectrogram');
    elements.patternValue = document.getElementById('patternValue');
    elements.confidenceValue = document.getElementById('confidenceValue');
    
    // Stepper buttons
    elements.stepButtons = document.querySelectorAll('.step-btn');
    
    // Comparison Views
    elements.specCut = document.getElementById('specCut');
    elements.specFade = document.getElementById('specFade');
    elements.specAI = document.getElementById('specAI');
    elements.playCutBtn = document.getElementById('playCutBtn');
    elements.playFadeBtn = document.getElementById('playFadeBtn');
    elements.playAIBtn = document.getElementById('playAIBtn');

    // Comparison header info
    elements.compareModelInfo = document.getElementById('compareModelInfo');
    elements.compareModeInfo = document.getElementById('compareModeInfo');
}

// ============================================
// INITIALIZATION
// ============================================
async function init() {
    cacheElements();
    
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: CONFIG.SAMPLE_RATE
    });
    
    await loadModel();
    setupEventListeners();
    setupTabs();
    
    drawEmptyWaveform(elements.waveformA);
    drawEmptyWaveform(elements.waveformB);
    drawEmptyWaveform(elements.spectrogram, '#000000', '#FFFFFF'); // Init summary as black waveform

    // Redraw once after first layout pass to avoid fallback-size canvases.
    requestAnimationFrame(() => refreshCanvasesForView('view-mixer'));

    // Initialize compare header text.
    updateCompareHeader();

    // Keep canvases sized correctly when the view/layout changes.
    window.addEventListener('resize', () => {
        if (state.isPlaying) stopPlayback();
        const activeViewId = document.querySelector('.view-section.active')?.id;
        if (activeViewId) refreshCanvasesForView(activeViewId);
    });
    
    console.log('[FlowState] Initialized');
}

async function loadModel() {
    updateStatus('LOADING...', 'loading');
    
    // WebGL options - works with full model (no quantization ops)
    const webglOptions = {
        executionProviders: ['webgl'],
        graphOptimizationLevel: 'all',
    };
    
    // WASM options with low memory footprint
    const wasmOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
        enableCpuMemArena: false,
        enableMemPattern: false,
    };
    
    // Strategy 1: Try FULL model with WASM first (WebGL returns incorrect 0.0 predictions)
    try {
        const response = await fetch(CONFIG.MODEL_HEAVY, { method: 'HEAD' });
        if (response.ok) {
            console.log('[FlowState] Trying full model with WASM...');
            state.modelSession = await ort.InferenceSession.create(CONFIG.MODEL_HEAVY, wasmOptions);
            state.modelType = 'heavy';
            updateStatus('READY (CPU)', 'ready');
            updateCompareHeader();
            console.log('[FlowState] Full model loaded with WASM');
            await testModelSanity();
            return;
        }
    } catch (e) {
        console.warn('[FlowState] Full model + WASM failed:', e.message);
    }
    
    // Strategy 2: Try QUANTIZED model with WASM (DynamicQuantizeLinear not supported in WebGL)
    try {
        const response = await fetch(CONFIG.MODEL_LIGHT, { method: 'HEAD' });
        if (response.ok) {
            console.log('[FlowState] Trying quantized model with WASM...');
            state.modelSession = await ort.InferenceSession.create(CONFIG.MODEL_LIGHT, wasmOptions);
            state.modelType = 'light';
            updateStatus('READY (QUANT)', 'ready');
            updateCompareHeader();
            console.log('[FlowState] Quantized model loaded with WASM');
            return;
        }
    } catch (e) {
        console.warn('[FlowState] Quantized model + WASM failed:', e.message);
    }
    
    // Strategy 3: Try FULL model with WASM as last resort
    try {
        const response = await fetch(CONFIG.MODEL_HEAVY, { method: 'HEAD' });
        if (response.ok) {
            console.log('[FlowState] Trying full model with WASM...');
            state.modelSession = await ort.InferenceSession.create(CONFIG.MODEL_HEAVY, wasmOptions);
            state.modelType = 'heavy';
            updateStatus('READY (CPU)', 'ready');
            updateCompareHeader();
            console.log('[FlowState] Full model loaded with WASM');
            return;
        }
    } catch (e) {
        console.warn('[FlowState] Full model + WASM failed:', e.message);
    }
    
    // Fallback: Demo mode
    state.modelType = 'none';
    updateStatus('DEMO', 'warning');
    updateCompareHeader();
    console.log('[FlowState] Running in DEMO mode (no model loaded)');
}

// Sanity test for the loaded model
async function testModelSanity() {
    if (!state.modelSession) return;
    
    const inputName = state.modelSession.inputNames[0];
    const outputName = state.modelSession.outputNames[0];
    
    console.log('[FlowState] === MODEL SANITY TEST ===');
    
    // Test 1: All zeros
    const zeros = new Float32Array(128 * 128).fill(0);
    const tensorZeros = new ort.Tensor('float32', zeros, [1, 1, 128, 128]);
    const resultZeros = await state.modelSession.run({ [inputName]: tensorZeros });
    console.log(`[FlowState] Input ALL ZEROS -> Output: ${resultZeros[outputName].data[0].toFixed(6)}`);
    
    // Test 2: All 0.5
    const halves = new Float32Array(128 * 128).fill(0.5);
    const tensorHalves = new ort.Tensor('float32', halves, [1, 1, 128, 128]);
    const resultHalves = await state.modelSession.run({ [inputName]: tensorHalves });
    console.log(`[FlowState] Input ALL 0.5 -> Output: ${resultHalves[outputName].data[0].toFixed(6)}`);
    
    // Test 3: All ones
    const ones = new Float32Array(128 * 128).fill(1.0);
    const tensorOnes = new ort.Tensor('float32', ones, [1, 1, 128, 128]);
    const resultOnes = await state.modelSession.run({ [inputName]: tensorOnes });
    console.log(`[FlowState] Input ALL ONES -> Output: ${resultOnes[outputName].data[0].toFixed(6)}`);
    
    // Test 4: Random values
    const random = new Float32Array(128 * 128);
    for (let i = 0; i < random.length; i++) random[i] = Math.random();
    const tensorRandom = new ort.Tensor('float32', random, [1, 1, 128, 128]);
    const resultRandom = await state.modelSession.run({ [inputName]: tensorRandom });
    console.log(`[FlowState] Input RANDOM -> Output: ${resultRandom[outputName].data[0].toFixed(6)}`);
    
    console.log('[FlowState] === END SANITY TEST ===');
}

function getModelLabel() {
    if (state.modelType === 'light') return 'MODEL: QUANTIZED';
    if (state.modelType === 'heavy') return 'MODEL: FULL';
    return 'MODEL: DEMO';
}

function updateCompareHeader(modeOverride) {
    if (elements.compareModelInfo) elements.compareModelInfo.textContent = getModelLabel();
    if (elements.compareModeInfo) {
        if (modeOverride) elements.compareModeInfo.textContent = `MODE: ${modeOverride.replace('_', ' ')}`;
        else if (state.trackA.classification && state.trackB.classification) {
            const inferred = getMode(state.trackA.classification, state.trackB.classification);
            elements.compareModeInfo.textContent = `MODE: ${inferred.replace('_', ' ')}`;
        } else {
            elements.compareModeInfo.textContent = 'MODE: --';
        }
    }
}

function updateStatus(text, status) {
    if (elements.modelStatusText) elements.modelStatusText.textContent = text;
    if (elements.modelStatusDot) {
        elements.modelStatusDot.className = 'status-dot';
        if (status === 'ready') elements.modelStatusDot.classList.add('ready');
    }
}

// ============================================
// EVENT LISTENERS
// ============================================
function setupEventListeners() {
    elements.loadTrackA?.addEventListener('click', () => elements.inputA?.click());
    elements.loadTrackB?.addEventListener('click', () => elements.inputB?.click());
    
    elements.inputA?.addEventListener('change', (e) => handleFileSelect(e, 'A'));
    elements.inputB?.addEventListener('change', (e) => handleFileSelect(e, 'B'));
    
    setupDragDrop(elements.trackAPanel, 'A');
    setupDragDrop(elements.trackBPanel, 'B');
    
    elements.stepButtons.forEach(btn => btn.addEventListener('click', handleStepperClick));
    
    elements.processBtn?.addEventListener('click', processTransition);
    elements.downloadBtn?.addEventListener('click', exportWAV);
    if(elements.exportLogsBtn) {
        console.log('[FlowState] Attaching listener to exportLogsBtn');
        elements.exportLogsBtn.addEventListener('click', exportLogs);
    } else {
        console.error('[FlowState] exportLogsBtn NOT FOUND');
    }
    
    // Playback Listeners
    elements.playBtn?.addEventListener('click', () => 
        togglePlayback(state.outputBuffer, elements.playBtn, elements.spectrogram, 'waveform'));
        
    elements.previewA?.addEventListener('click', (e) => {
        e.stopPropagation();
        togglePlayback(state.trackA.buffer, elements.previewA, elements.waveformA, 'waveform');
    });
    
    elements.previewB?.addEventListener('click', (e) => {
        e.stopPropagation();
        togglePlayback(state.trackB.buffer, elements.previewB, elements.waveformB, 'waveform');
    });
    
    elements.playCutBtn?.addEventListener('click', () => 
        togglePlayback(state.bufferCut, elements.playCutBtn, elements.specCut, 'spectrogram'));
        
    elements.playFadeBtn?.addEventListener('click', () => 
        togglePlayback(state.bufferFade, elements.playFadeBtn, elements.specFade, 'spectrogram'));
        
    elements.playAIBtn?.addEventListener('click', () => 
        togglePlayback(state.bufferAI, elements.playAIBtn, elements.specAI, 'spectrogram'));
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab-item');
    const views = document.querySelectorAll('.view-section');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.target;
            if (!targetId) return; 

            // Prevent playhead/canvas snapshot desync across views.
            if (state.isPlaying) stopPlayback();

            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            views.forEach(view => {
                view.style.display = 'none';
                view.classList.remove('active');
                if (view.id === targetId) {
                    view.style.display = 'block';
                    view.classList.add('active');
                }
            });

            // Redraw now that the canvases have real layout sizes.
            refreshCanvasesForView(targetId);
        });
    });
}

function refreshCanvasesForView(viewId) {
    if (viewId === 'view-mixer') {
        if (state.trackA.buffer) drawWaveform(elements.waveformA, state.trackA.buffer);
        else drawEmptyWaveform(elements.waveformA);

        if (state.trackB.buffer) drawWaveform(elements.waveformB, state.trackB.buffer);
        else drawEmptyWaveform(elements.waveformB);

        if (state.outputBuffer) drawWaveform(elements.spectrogram, state.outputBuffer, '#000000', '#FFFFFF');
        else drawEmptyWaveform(elements.spectrogram, '#000000', '#FFFFFF');
    } else if (viewId === 'view-comparison') {
        if (state.bufferCut) drawSpectrogram(elements.specCut, state.bufferCut);
        else drawEmptySpectrogram(elements.specCut);

        if (state.bufferFade) drawSpectrogram(elements.specFade, state.bufferFade);
        else drawEmptySpectrogram(elements.specFade);

        if (state.bufferAI) drawSpectrogram(elements.specAI, state.bufferAI);
        else drawEmptySpectrogram(elements.specAI);
    }
}

function handleStepperClick(e) {
    const btn = e.target;
    const wrapper = btn.closest('.track-controls-col');
    const valueEl = wrapper?.querySelector('.step-val');
    const label = wrapper?.querySelector('.control-label')?.textContent;
    const isPlus = btn.textContent.trim() === '+';
    
    if (label === 'FILTER') {
        state.filterIntensity = Math.max(0, Math.min(100, state.filterIntensity + (isPlus ? 10 : -10)));
        valueEl.textContent = `${state.filterIntensity}%`;
    } else if (label === 'DURATION') {
        state.transitionDuration = Math.max(2, Math.min(8, state.transitionDuration + (isPlus ? 0.5 : -0.5)));
        valueEl.textContent = `${state.transitionDuration.toFixed(1)}s`;
    }
}

function setupDragDrop(element, track) {
    if (!element) return;
    element.addEventListener('dragover', (e) => { e.preventDefault(); element.classList.add('dragover'); });
    element.addEventListener('dragleave', () => { element.classList.remove('dragover'); });
    element.addEventListener('drop', async (e) => {
        e.preventDefault(); element.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) await loadAudioFile(file, track);
    });
}

async function handleFileSelect(event, track) {
    const file = event.target.files[0];
    if (file) await loadAudioFile(file, track);
}

async function loadAudioFile(file, track) {
    try {
        const trackData = track === 'A' ? state.trackA : state.trackB;
        const filenameEl = track === 'A' ? elements.filenameA : elements.filenameB;
        const waveformEl = track === 'A' ? elements.waveformA : elements.waveformB;
        const statusEl = track === 'A' ? elements.classA : elements.classB;
        const previewBtn = track === 'A' ? elements.previewA : elements.previewB;
        
        filenameEl.textContent = file.name.toUpperCase();
        trackData.filename = file.name;
        if (statusEl) statusEl.textContent = "ANALYZING...";
        
        const arrayBuffer = await file.arrayBuffer();
        trackData.buffer = await state.audioContext.decodeAudioData(arrayBuffer);
        
        drawWaveform(waveformEl, trackData.buffer);
        if (previewBtn) previewBtn.style.display = 'flex'; 
        
        await classifyTrack(track);
        updateButtonStates();
        
    } catch (error) {
        console.error(`[FlowState] Failed to load track ${track}:`, error);
        (track === 'A' ? elements.classA : elements.classB).textContent = "ERROR";
    }
}

// ============================================
// VISUALIZATION
// ============================================
function getCanvasCssSize(canvas, fallbackW, fallbackH) {
    // Prefer actual rendered size; fall back to parent box; then to provided fallback.
    const rect = canvas.getBoundingClientRect();
    let w = rect.width;
    let h = rect.height;
    if ((!w || !h) && canvas.parentElement) {
        const parentRect = canvas.parentElement.getBoundingClientRect();
        w = w || parentRect.width;
        h = h || parentRect.height;
    }
    w = Math.max(1, Math.round(w || fallbackW || 1));
    h = Math.max(1, Math.round(h || fallbackH || 1));
    return { w, h };
}

function setupCanvasForCssSize(canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    return ctx;
}

function drawEmptyWaveform(canvas, bgColor='#f0f0f0', fgColor='#000000') {
    if (!canvas) return;
    const { w, h } = getCanvasCssSize(canvas, canvas.offsetWidth || 80, canvas.offsetHeight || 80);
    const ctx = setupCanvasForCssSize(canvas, w, h);
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);
}

function drawWaveform(canvas, audioBuffer, bgColor = '#f0f0f0', fgColor = '#000000') {
    if (!canvas || !audioBuffer) return;
    const { w, h } = getCanvasCssSize(canvas, canvas.offsetWidth || 80, canvas.offsetHeight || 80);
    const ctx = setupCanvasForCssSize(canvas, w, h);
    
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);
    
    const data = audioBuffer.getChannelData(0);
    const step = Math.max(1, Math.floor(data.length / w));

    // Small preview canvases (80x80) easily saturate into a solid block on loud/limited audio.
    // Use a compressed peak envelope so you can actually see the waveform.
    const isTinyPreview = w <= 120 && h <= 120;
    const midY = h / 2;

    // Normalize using a coarse scan for performance.
    let maxAbs = 0;
    for (let i = 0; i < data.length; i += step) {
        const v = Math.abs(data[i]);
        if (v > maxAbs) maxAbs = v;
    }
    if (maxAbs < 1e-6) maxAbs = 1;

    ctx.strokeStyle = fgColor;
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Fix: Step by 2 pixels to create "gaps" so it doesn't look like a solid block
    for (let x = 0; x < w; x += 2) {
        const start = x * step;
        const end = Math.min(start + step * 2, data.length); // Include the skipped pixel in analysis

        if (isTinyPreview) {
            let peak = 0;
            for (let i = start; i < end; i++) {
                const v = Math.abs(data[i]) / maxAbs;
                if (v > peak) peak = v;
            }
            // Fix: Square the value to visually "un-compress" the waveform (makes it spikier)
            // Fix: Keep height at 0.45 (90% total) now that we have gaps
            const amp = (peak * peak) * (h * 0.45); 
            ctx.moveTo(x + 0.5, midY - amp);
            ctx.lineTo(x + 0.5, midY + amp);
        } else {
            let min = 1.0;
            let max = -1.0;
            for (let i = start; i < end; i++) {
                const v = data[i] / maxAbs;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            const y1 = midY - max * (h * 0.45);
            const y2 = midY - min * (h * 0.45);
            ctx.moveTo(x + 0.5, y1);
            ctx.lineTo(x + 0.5, y2);
        }
    }

    ctx.stroke();
}

function drawEmptySpectrogram(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const w = canvas.offsetWidth || 1024;
    const h = canvas.offsetHeight || 128;
    
    canvas.width = w * window.devicePixelRatio;
    canvas.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);
}

function drawSpectrogram(canvas, buffer) {
    if (!canvas || !buffer) return;
    const ctx = canvas.getContext('2d');
    
    const w = canvas.offsetWidth || 1024;
    const h = canvas.offsetHeight || 128;
    
    canvas.width = w * window.devicePixelRatio;
    canvas.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const data = buffer.getChannelData(0);
    const nFft = 2048;
    const hopSize = Math.floor(data.length / w);
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);
    
    // Create FFT instance
    const fft = new FFT(nFft);
    const fftInput = fft.createComplexArray();
    const fftOutput = fft.createComplexArray();
    
    // Pre-compute all spectrogram columns
    const specData = [];
    let globalMax = 1e-10;
    
    for (let x = 0; x < w; x++) {
        const frameStart = x * hopSize;
        
        // Extract frame with Hann window
        for (let i = 0; i < nFft; i++) {
            const sample = (frameStart + i < data.length) ? data[frameStart + i] : 0;
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)));
            fftInput[i * 2] = sample * window;
            fftInput[i * 2 + 1] = 0;
        }
        
        // FFT
        fft.transform(fftOutput, fftInput);
        
        // Compute power spectrum for display frequencies
        const column = new Float32Array(h);
        for (let y = 0; y < h; y++) {
            // Map y to frequency bin (inverted: top = high freq)
            const freqRatio = (h - 1 - y) / h;
            const k = Math.floor(freqRatio * nFft / 2);
            const re = fftOutput[k * 2];
            const im = fftOutput[k * 2 + 1];
            const power = re * re + im * im;
            column[y] = power;
            if (power > globalMax) globalMax = power;
        }
        specData.push(column);
    }
    
    // Draw with dB scale and magma-like colormap
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
            const power = specData[x][y];
            // Convert to dB, normalize to 0-1
            const db = 10 * Math.log10(Math.max(power, 1e-10) / globalMax);
            const normalized = Math.max(0, Math.min(1, (db + 80) / 80)); // -80dB to 0dB
            
            // Magma-like colormap
            const r = Math.floor(255 * Math.pow(normalized, 0.8));
            const g = Math.floor(100 * Math.pow(normalized, 1.5));
            const b = Math.floor(255 * (0.3 + 0.7 * Math.pow(1 - normalized, 0.5)));
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, y, 1, 1);
        }
    }
}

// ============================================
// AI CLASSIFICATION
// ============================================
async function classifyTrack(track) {
    const trackData = track === 'A' ? state.trackA : state.trackB;
    if (!trackData.buffer) return;
    
    if (state.modelType === 'none') {
        trackData.classification = Math.random() > 0.5 ? 'RHYTHMIC' : 'HARMONIC';
        trackData.confidence = 85;
        updateClassificationUI(track);
        return;
    }
  try {
        const segment = track === 'A' ? 'end' : 'start';
        const melSpec = generateMelSpectrogram(trackData.buffer, segment);
        const tensor = new ort.Tensor('float32', melSpec, [1, 1, CONFIG.MEL_BANDS, CONFIG.SPEC_WIDTH]);
        
        // Debug: Check tensor
        console.log(`[FlowState] Tensor shape: [${tensor.dims.join(', ')}], type: ${tensor.type}`);
        console.log(`[FlowState] Tensor data sample (first 10): [${Array.from(tensor.data.slice(0, 10)).map(x => x.toFixed(4)).join(', ')}]`);
        console.log(`[FlowState] Tensor data sample (last 10): [${Array.from(tensor.data.slice(-10)).map(x => x.toFixed(4)).join(', ')}]`);
        
        // --- FIX: Dynamic Input Name ---
        const feeds = {};
        const inputName = state.modelSession.inputNames[0]; // Get actual name from model
        feeds[inputName] = tensor;
        console.log(`[FlowState] Input name: ${inputName}`);
        
        const results = await state.modelSession.run(feeds);
        // -------------------------------

        // --- FIX: Dynamic Output Name ---
        const outputName = state.modelSession.outputNames[0];
        console.log(`[FlowState] Output name: ${outputName}`);
        console.log(`[FlowState] Full output: [${Array.from(results[outputName].data).map(x => x.toFixed(6)).join(', ')}]`);
        const output = results[outputName].data[0];
        // -------------------------------
        
        // Debug: Log prediction like Python does
        console.log(`[FlowState] Track ${track} (${segment}): pred=${output.toFixed(4)} → ${output < 0.5 ? 'RHYTHMIC' : 'HARMONIC'}`);
        
        trackData.classification = output < 0.5 ? 'RHYTHMIC' : 'HARMONIC';
        trackData.confidence = Math.abs(output - 0.5) * 2 * 100;
        updateClassificationUI(track);
    } catch (error) {
        console.error("AI Error:", error); // Log error to see it
        trackData.classification = 'UNKNOWN';
        updateClassificationUI(track);
    }
}

function generateMelSpectrogram(audioBuffer, segment = 'start') {
    const bufferSampleRate = audioBuffer.sampleRate;
    const data = audioBuffer.getChannelData(0);
    
    // Resample if needed
    let samples;
    if (bufferSampleRate !== CONFIG.SAMPLE_RATE) {
        const ratio = bufferSampleRate / CONFIG.SAMPLE_RATE;
        const newLength = Math.floor(data.length / ratio);
        const resampled = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
            resampled[i] = data[Math.floor(i * ratio)];
        }
        samples = resampled;
    } else {
        samples = new Float32Array(data);
    }
    
    // Extract target segment
    let chunk = new Float32Array(CONFIG.TARGET_SAMPLES);
    if (segment === 'end') {
        if (samples.length > CONFIG.TARGET_SAMPLES) {
            chunk.set(samples.subarray(samples.length - CONFIG.TARGET_SAMPLES));
        } else {
            chunk.set(samples, CONFIG.TARGET_SAMPLES - samples.length);
        }
    } else {
        if (samples.length > CONFIG.TARGET_SAMPLES) {
            chunk.set(samples.subarray(0, CONFIG.TARGET_SAMPLES));
        } else {
            chunk.set(samples);
        }
    }
    
    // FFT-based Mel Spectrogram (matching librosa)
    const nFft = 2048;
    const hopLength = 512;
    const nMels = CONFIG.MEL_BANDS; // 128
    const sr = CONFIG.SAMPLE_RATE;  // 22050
    
    // Create FFT instance
    const fft = new FFT(nFft);
    const fftInput = fft.createComplexArray();
    const fftOutput = fft.createComplexArray();
    
    // Pre-compute mel filterbank
    const melFilters = createMelFilterbank(nFft, nMels, sr);
    
    // Compute number of frames
    const nFrames = Math.floor((chunk.length - nFft) / hopLength) + 1;
    const targetFrames = CONFIG.SPEC_WIDTH; // 128
    
    // Compute mel spectrogram
    const melSpec = new Float32Array(nMels * targetFrames);
    
    for (let t = 0; t < Math.min(nFrames, targetFrames); t++) {
        const frameStart = t * hopLength;
        
        // Extract frame and apply Hann window, convert to complex format
        for (let i = 0; i < nFft; i++) {
            const sample = (frameStart + i < chunk.length) ? chunk[frameStart + i] : 0;
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)));
            fftInput[i * 2] = sample * window;     // Real part
            fftInput[i * 2 + 1] = 0;               // Imaginary part
        }
        
        // Compute FFT
        fft.transform(fftOutput, fftInput);
        
        // Compute power spectrum (only first half + DC)
        const powerSpec = new Float32Array(nFft / 2 + 1);
        for (let k = 0; k <= nFft / 2; k++) {
            const re = fftOutput[k * 2];
            const im = fftOutput[k * 2 + 1];
            powerSpec[k] = re * re + im * im;
        }
        
        // Apply mel filterbank
        for (let m = 0; m < nMels; m++) {
            let melEnergy = 0;
            for (let k = 0; k < powerSpec.length; k++) {
                melEnergy += powerSpec[k] * melFilters[m][k];
            }
            melSpec[m * targetFrames + t] = melEnergy;
        }
    }
    
    // Convert to dB (power_to_db with ref=np.max)
    let maxPower = 1e-10;
    for (let i = 0; i < melSpec.length; i++) {
        if (melSpec[i] > maxPower) maxPower = melSpec[i];
    }
    
    for (let i = 0; i < melSpec.length; i++) {
        let db = 10 * Math.log10(Math.max(melSpec[i], 1e-10) / maxPower);
        melSpec[i] = Math.max(db, -80.0);
    }
    
    // Min-Max normalize (EXACTLY as in training)
    let minVal = melSpec[0], maxVal = melSpec[0];
    for (let i = 1; i < melSpec.length; i++) {
        if (melSpec[i] < minVal) minVal = melSpec[i];
        if (melSpec[i] > maxVal) maxVal = melSpec[i];
    }
    
    const range = maxVal - minVal;
    if (range > 1e-6) {
        for (let i = 0; i < melSpec.length; i++) {
            melSpec[i] = (melSpec[i] - minVal) / range;
        }
    } else {
        melSpec.fill(0.5);
    }
    
    // Debug
    let sum = 0;
    for (let i = 0; i < melSpec.length; i++) sum += melSpec[i];
    console.log(`[FlowState] Mel spec (FFT): dB=[${minVal.toFixed(1)}, ${maxVal.toFixed(1)}], mean=${(sum/melSpec.length).toFixed(4)}`);
    
    return melSpec;
}

// Create mel filterbank (matching librosa.filters.mel)
function createMelFilterbank(nFft, nMels, sr) {
    const fMin = 0;
    const fMax = sr / 2;
    
    // Hz to Mel conversion (HTK formula)
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
    
    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);
    
    // Create mel points (nMels + 2 points for triangular filters)
    const nMelPoints = nMels + 2;
    const melPoints = new Float32Array(nMelPoints);
    for (let i = 0; i < nMelPoints; i++) {
        melPoints[i] = melMin + (melMax - melMin) * i / (nMelPoints - 1);
    }
    
    // Convert to Hz
    const hzPoints = melPoints.map(melToHz);
    
    // Convert to FFT bin indices
    const binPoints = hzPoints.map(hz => Math.floor((nFft + 1) * hz / sr));
    
    // Create triangular filterbank
    const filters = [];
    for (let m = 0; m < nMels; m++) {
        const filter = new Float32Array(nFft / 2 + 1);
        
        const startBin = binPoints[m];
        const centerBin = binPoints[m + 1];
        const endBin = binPoints[m + 2];
        
        // Rising slope
        for (let k = startBin; k < centerBin; k++) {
            if (k >= 0 && k < filter.length && centerBin > startBin) {
                filter[k] = (k - startBin) / (centerBin - startBin);
            }
        }
        
        // Falling slope
        for (let k = centerBin; k < endBin; k++) {
            if (k >= 0 && k < filter.length && endBin > centerBin) {
                filter[k] = (endBin - k) / (endBin - centerBin);
            }
        }
        
        filters.push(filter);
    }
    
    return filters;
}

// ============================================
// FEATURE ANALYSIS & UI UPDATES
// ============================================
function analyzeFeatures(buffer) {
    if (!buffer) return { zcr: 0, flux: 0, bpm: 0 };
    
    const data = buffer.getChannelData(0);
    
    // 1. Zero Crossing Rate (ZCR)
    let zeroCrossings = 0;
    for (let i = 1; i < data.length; i++) {
        if ((data[i] >= 0 && data[i-1] < 0) || (data[i] < 0 && data[i-1] >= 0)) {
            zeroCrossings++;
        }
    }
    const zcr = zeroCrossings / data.length;
    
    // 2. Simple Spectral Flux (Change in energy)
    // We'll use a simplified time-domain approach: delta energy between chunks
    const chunkSize = 1024;
    let totalFlux = 0;
    let prevEnergy = 0;
    let chunks = 0;
    
    for (let i = 0; i < data.length; i += chunkSize) {
        let energy = 0;
        for (let j = 0; j < chunkSize && i + j < data.length; j++) {
            energy += data[i+j] * data[i+j];
        }
        energy = Math.sqrt(energy); // RMS
        
        if (i > 0) {
            totalFlux += Math.abs(energy - prevEnergy);
            chunks++;
        }
        prevEnergy = energy;
    }
    const flux = chunks > 0 ? totalFlux / chunks : 0;
    
    // 3. BPM Placeholder (Pulse Clarity or Peak Detect is complex, return estimate)
    // For now, return a placeholder based on classification
    // In a real app, use a dedicated beat detection algo
    
    return { zcr, flux };
}

function updateAnalysisUI() {
    const featFluxA = document.getElementById('featFluxA');
    const featZcrA = document.getElementById('featZcrA');
    const featBpmA = document.getElementById('featBpmA');
    
    const featFluxB = document.getElementById('featFluxB');
    const featZcrB = document.getElementById('featZcrB');
    const featBpmB = document.getElementById('featBpmB');
    
    const analysisTransition = document.getElementById('analysisTransition');
    const analysisMode = document.getElementById('analysisMode');
    
    // Track A
    if (state.trackA.buffer) {
        const statsA = analyzeFeatures(state.trackA.buffer);
        featFluxA.textContent = statsA.flux.toFixed(3);
        featZcrA.textContent = statsA.zcr.toFixed(3);
        // Estimate BPM label based on ZCR (noisy = likely percussion)
        featBpmA.textContent = state.trackA.classification === 'RHYTHMIC' ? '~120-140' : '-- (Ambient)';
    } else {
        featFluxA.textContent = '--';
        featZcrA.textContent = '--';
        featBpmA.textContent = '--';
    }
    
    // Track B
    if (state.trackB.buffer) {
        const statsB = analyzeFeatures(state.trackB.buffer);
        featFluxB.textContent = statsB.flux.toFixed(3);
        featZcrB.textContent = statsB.zcr.toFixed(3);
        featBpmB.textContent = state.trackB.classification === 'RHYTHMIC' ? '~120-140' : '-- (Ambient)';
    } else {
        featFluxB.textContent = '--';
        featZcrB.textContent = '--';
        featBpmB.textContent = '--';
    }
    
    // Strategy
    const classA = state.trackA.classification || '--';
    const classB = state.trackB.classification || '--';
    const mode = document.getElementById('compareModeInfo').textContent.replace('MODE: ', '') || '--';
    
    analysisTransition.textContent = `${classA} -> ${classB}`;
    analysisMode.textContent = mode;
}

function updateExportUI() {
    const exportFormat = document.getElementById('exportFormat');
    const exportRate = document.getElementById('exportRate');
    const exportChannels = document.getElementById('exportChannels');
    
    // Check what we have
    if (state.outputBuffer) {
        exportFormat.textContent = 'WAV (PCM 32-BIT FLOAT)'; // WebAudio default
        exportRate.textContent = `${state.outputBuffer.sampleRate}Hz`;
        exportChannels.textContent = state.outputBuffer.numberOfChannels === 2 ? 'STEREO' : 'MONO';
    } else {
        exportFormat.textContent = 'WAV (PENDING)';
        exportRate.textContent = '22050Hz (Target)';
        exportChannels.textContent = 'STEREO';
    }
}

function updateClassificationUI(track) {
    const trackData = track === 'A' ? state.trackA : state.trackB;
    const el = track === 'A' ? elements.classA : elements.classB;
    if (el) {
        el.textContent = trackData.classification || '--';
        el.style.color = '#000';
        el.style.fontWeight = '700';
    }
    updateInsights();
}

function updateInsights() {
    if (state.trackA.classification && state.trackB.classification) {
        let mode = getMode(state.trackA.classification, state.trackB.classification);
        if (elements.patternValue) elements.patternValue.textContent = `MODE: ${mode.replace('_', ' ')}`;
        const avg = (state.trackA.confidence + state.trackB.confidence) / 2;
        if (elements.confidenceValue) elements.confidenceValue.textContent = `${avg.toFixed(0)}%`;
    }
}

function getMode(typeA, typeB) {
    if (typeA === 'RHYTHMIC' && typeB === 'RHYTHMIC') return 'BEAT_MATCH';
    if (typeA === 'HARMONIC' && typeB === 'HARMONIC') return 'LIQUID_BLEND';
    if (typeA === 'RHYTHMIC' && typeB === 'HARMONIC') return 'ECHO_FREEZE';
    return 'DROP';
}

// ============================================
// PROCESSING ENGINE & GENERATORS
// ============================================
async function processTransition() {
    if (!state.trackA.buffer || !state.trackB.buffer) return;
    
    state.isProcessing = true;
    elements.processBtn.innerHTML = 'PROCESSING...';
    elements.processBtn.disabled = true;
    
    try {
        const mode = getMode(state.trackA.classification, state.trackB.classification);
        const trimOffset = findFirstBeat(state.trackB.buffer);

        updateCompareHeader(mode);

        state.bufferCut = generateHardCut(state.trackA.buffer, state.trackB.buffer);
        // Baseline is intentionally "dumb": no beat alignment.
        state.bufferFade = generateCrossfade(state.trackA.buffer, state.trackB.buffer, 0);
        
        if (mode === 'DROP') state.bufferAI = generateDropTransition(state.trackA.buffer, state.trackB.buffer, trimOffset);
        else if (mode === 'ECHO_FREEZE') state.bufferAI = generateEchoFreeze(state.trackA.buffer, state.trackB.buffer, trimOffset);
        else state.bufferAI = generateCrossfade(state.trackA.buffer, state.trackB.buffer, trimOffset);

        state.outputBuffer = state.bufferAI;
        
        drawWaveform(elements.spectrogram, state.bufferAI, '#000000', '#FFFFFF');
        drawSpectrogram(elements.specCut, state.bufferCut);
        drawSpectrogram(elements.specFade, state.bufferFade);
        drawSpectrogram(elements.specAI, state.bufferAI);
        
        // Update data tabs
        updateExportUI();
        updateAnalysisUI();
        
        elements.processBtn.innerHTML = 'TRANSITION COMPLETE';
        elements.processBtn.disabled = false;
        elements.playBtn.disabled = false;
        elements.downloadBtn.disabled = false;
        elements.playCutBtn.disabled = false;
        elements.playFadeBtn.disabled = false;
        elements.playAIBtn.disabled = false;
        
        setTimeout(() => { elements.processBtn.innerHTML = 'INITIATE SEQUENCE'; }, 2000);
        
    } catch (error) {
        console.error(error);
        elements.processBtn.innerHTML = 'ERROR';
    } finally {
        state.isProcessing = false;
    }
}

function generateHardCut(bA, bB) {
    const out = state.audioContext.createBuffer(2, bA.length+bB.length, CONFIG.SAMPLE_RATE);
    for(let c=0;c<2;c++) {
        const dA=bA.getChannelData(c%bA.numberOfChannels), dB=bB.getChannelData(c%bB.numberOfChannels), o=out.getChannelData(c);
        o.set(dA,0); o.set(dB,bA.length);
    } return out;
}
function generateCrossfade(bA, bB, tr) {
    const fl=Math.floor(5*CONFIG.SAMPLE_RATE), sf=Math.min(bA.length,bB.length-tr,fl), len=bA.length+bB.length-tr-sf;
    const out=state.audioContext.createBuffer(2,len,CONFIG.SAMPLE_RATE);
    for(let c=0;c<2;c++) {
        const dA=bA.getChannelData(c%bA.numberOfChannels), dB=bB.getChannelData(c%bB.numberOfChannels), o=out.getChannelData(c);
        const sl=bA.length-sf; o.set(dA.subarray(0,sl),0);
        for(let i=0;i<sf;i++) o[sl+i]=dA[sl+i]*(1-(i/sf))+dB[tr+i]*(i/sf);
        if(bB.length-tr-sf>0) o.set(dB.subarray(tr+sf),sl+sf);
    } return out;
}

// function generateDropTransition(bufferA, bufferB, trimB) {
//     const sweepSamples = Math.floor(4.0 * CONFIG.SAMPLE_RATE);
//     const gapSamples = Math.floor(0.2 * CONFIG.SAMPLE_RATE);
//     const lengthA = bufferA.length, lengthB = bufferB.length - trimB;
//     const output = state.audioContext.createBuffer(2, lengthA + gapSamples + lengthB, CONFIG.SAMPLE_RATE);
//     const f1 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);
//     const f2 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);
//     const f3 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);

//     for (let c = 0; c < 2; c++) {
//         const dataA = bufferA.getChannelData(c % bufferA.numberOfChannels);
//         const dataB = bufferB.getChannelData(c % bufferB.numberOfChannels);
//         const out = output.getChannelData(c);
        
//         const sweepStart = lengthA - sweepSamples;
//         out.set(dataA.subarray(0, sweepStart), 0);
//         const tailA = new Float32Array(sweepSamples);
//         for (let i = 0; i < sweepSamples; i++) {
//             let s = dataA[sweepStart + i];
//             s = processFilter(f1, s); s = processFilter(f2, s); s = processFilter(f3, s);
//             s *= (1.0 - (i / sweepSamples)); out[sweepStart + i] = s; tailA[i] = s;
//         }
        
//         const revLen = Math.floor(2.0 * CONFIG.SAMPLE_RATE);
//         const revSnippet = tailA.slice(sweepSamples - Math.floor(0.2 * CONFIG.SAMPLE_RATE));
//         const revTail = new Float32Array(revLen);
//         for(let r=0;r<15;r++) {
//             const dec = Math.pow(0.6, r), off = r * revSnippet.length;
//             if (off>=revLen) break;
//             for(let k=0;k<revSnippet.length;k++) if (off+k<revLen) revTail[off+k] = revSnippet[k]*dec;
//         }
        
//         const bStart = lengthA + gapSamples;
//         out.set(dataB.subarray(trimB, trimB + lengthB), bStart);
//         for(let i=0;i<revLen;i++) if(bStart+i<output.length) out[bStart+i] += revTail[i];
//     }
//     return output;
// }

function generateDropTransition(bufferA, bufferB, trimB) {
    const sweepSamples = Math.floor(4.0 * CONFIG.SAMPLE_RATE);
    const gapSamples = Math.floor(0.2 * CONFIG.SAMPLE_RATE);
    const lengthA = bufferA.length, lengthB = bufferB.length - trimB;
    const output = state.audioContext.createBuffer(2, lengthA + gapSamples + lengthB, CONFIG.SAMPLE_RATE);

    for (let c = 0; c < 2; c++) {
        // --- FIX: Create NEW filters for EVERY channel ---
        const f1 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);
        const f2 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);
        const f3 = createHighPassFilter(800, CONFIG.SAMPLE_RATE, 0.707);
        // -------------------------------------------------

        const dataA = bufferA.getChannelData(c % bufferA.numberOfChannels);
        const dataB = bufferB.getChannelData(c % bufferB.numberOfChannels);
        const out = output.getChannelData(c);
        
        const sweepStart = lengthA - sweepSamples;
        out.set(dataA.subarray(0, sweepStart), 0);
        
        const tailA = new Float32Array(sweepSamples);
        for (let i = 0; i < sweepSamples; i++) {
            let s = dataA[sweepStart + i];
            // Apply 3-stage aggressive filtering
            s = processFilter(f1, s); 
            s = processFilter(f2, s); 
            s = processFilter(f3, s);
            
            // Fade out volume
            s *= (1.0 - (i / sweepSamples)); 
            out[sweepStart + i] = s; 
            tailA[i] = s;
        }
        
        // Dub Throw (Reverb) Logic
        const revLen = Math.floor(2.0 * CONFIG.SAMPLE_RATE);
        // Safety check: ensure we don't slice out of bounds
        const sliceStart = Math.max(0, sweepSamples - Math.floor(0.2 * CONFIG.SAMPLE_RATE));
        const revSnippet = tailA.subarray(sliceStart);
        
        const revTail = new Float32Array(revLen);
        for(let r=0;r<15;r++) {
            const dec = Math.pow(0.6, r), off = r * revSnippet.length;
            if (off>=revLen) break;
            for(let k=0;k<revSnippet.length;k++) {
                if (off+k<revLen) revTail[off+k] += revSnippet[k]*dec; // Use += for overlap mixing
            }
        }
        
        const bStart = lengthA + gapSamples;
        // Safety check: trimB might exceed buffer length
        if (trimB < dataB.length) {
            out.set(dataB.subarray(trimB, Math.min(dataB.length, trimB + lengthB)), bStart);
        }
        
        // Mix Reverb over Track B
        for(let i=0;i<revLen;i++) if(bStart+i<output.length) out[bStart+i] += revTail[i];
    }
    return output;
}

function generateEchoFreeze(bA, bB, tr) {
    const bl=Math.floor(0.5*CONFIG.SAMPLE_RATE), ov=Math.floor(1*CONFIG.SAMPLE_RATE), xf=Math.floor(0.1*CONFIG.SAMPLE_RATE);
    const el=bl*6, brl=Math.max(el, ov+bB.length-tr), out=state.audioContext.createBuffer(2,bA.length+brl-xf,CONFIG.SAMPLE_RATE);
    for(let c=0;c<2;c++) {
        const dA=bA.getChannelData(c%bA.numberOfChannels), dB=bB.getChannelData(c%bB.numberOfChannels), o=out.getChannelData(c);
        const lb=dA.subarray(bA.length-bl), et=new Float32Array(el);
        for(let r=0;r<6;r++) { const v=Math.pow(0.63,r), off=r*bl; for(let k=0;k<bl;k++) et[off+k]=lb[k]*v; }
        const br=new Float32Array(brl); br.set(et,0);
        for(let i=0;i<bB.length-tr;i++) if(ov+i<brl) br[ov+i]+=dB[tr+i];
        
        const fs=bA.length-xf; o.set(dA.subarray(0,fs),0);
        for(let i=0;i<xf;i++) o[fs+i]=dA[fs+i]*(1-(i/xf));
        for(let i=0;i<brl;i++) { let s=br[i]; if(i<xf) s*=i/xf; if(fs+i<out.length) o[fs+i]+=s; }
    } return out;
}

function findFirstBeat(b) {
    const d=b.getChannelData(0), l=Math.min(d.length,CONFIG.SAMPLE_RATE*10), w=Math.floor(0.05*CONFIG.SAMPLE_RATE);
    let mx=0; for(let i=0;i<l;i+=w) { let s=0; for(let j=0;j<w&&i+j<l;j++) s+=d[i+j]*d[i+j]; const r=Math.sqrt(s/w); if(r>mx) mx=r; }
    const th=mx*0.1; for(let i=0;i<l;i+=w) { let s=0; for(let j=0;j<w&&i+j<l;j++) s+=d[i+j]*d[i+j]; if(Math.sqrt(s/w)>th) return i; }
    return 0;
}
function createHighPassFilter(f,r,q){const w=2*Math.PI*f/r,a=Math.sin(w)/(2*q),c=Math.cos(w),n=1+a;return{b0:((1+c)/2)/n,b1:(-(1+c))/n,b2:((1+c)/2)/n,a1:(-2*c)/n,a2:(1-a)/n,z1:0,z2:0};}
function processFilter(f,s){const o=f.b0*s+f.b1*f.z1+f.b2*f.z2-f.a1*f.z1-f.a2*f.z2;f.z2=f.z1;f.z1=o;return o;}

// ============================================
// PLAYBACK & PLAYHEAD ANIMATION
// ============================================
async function togglePlayback(buffer, btn, canvas, type) {
    // --- FIX: Resume Context on Click ---
    if (state.audioContext.state === 'suspended') {
        await state.audioContext.resume();
    }
    if (!buffer) return;
    
    if (state.isPlaying) {
        const wasBtn = state.activeBtn;
        stopPlayback();
        if (wasBtn === btn) return; 
    }
    
    startPlayback(buffer, btn, canvas, type);
}

function startPlayback(buffer, btn, canvas, type) {
    const source = state.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(state.audioContext.destination);
    state.currentSource = source;
    state.isPlaying = true;
    state.activeBtn = btn;
    state.activeCanvas = canvas;
    state.activeBuffer = buffer;
    state.playStartTime = state.audioContext.currentTime; 
    
    // SNAPSHOT: Capture the current visual state (Spectrogram or Waveform)
    if (canvas) {
        const ctx = canvas.getContext('2d');
        state.baseImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    
    source.start();
    btn.innerHTML = '■';
    btn.classList.add('playing');
    
    // Start Animation Loop
    animatePlayhead();
    
    source.onended = () => { if (state.currentSource === source) stopPlayback(); };
}

function stopPlayback() {
    if (state.currentSource) { try { state.currentSource.stop(); } catch(e){} state.currentSource = null; }
    state.isPlaying = false;
    
    // Cancel Animation
    if (state.playheadAnimationId) {
        cancelAnimationFrame(state.playheadAnimationId);
        state.playheadAnimationId = null;
    }
    
    // Restore the clean snapshot one last time (removes playhead)
    if (state.activeCanvas && state.baseImageData) {
        const ctx = state.activeCanvas.getContext('2d');
        ctx.putImageData(state.baseImageData, 0, 0);
    }
    
    // Reset Button
    if (state.activeBtn) {
        state.activeBtn.innerHTML = '▶';
        state.activeBtn.classList.remove('playing');
        state.activeBtn = null;
    }
    
    state.activeCanvas = null;
    state.activeBuffer = null;
    state.baseImageData = null;
}

function animatePlayhead() {
    if (!state.isPlaying || !state.activeCanvas || !state.activeBuffer || !state.baseImageData) return;
    
    const elapsed = state.audioContext.currentTime - state.playStartTime;
    const duration = state.activeBuffer.duration;
    const progress = elapsed / duration;
    
    if (progress >= 1) return; // onended handles stop logic
    
    const ctx = state.activeCanvas.getContext('2d');
    
    // 1. Restore the base image (Physical Pixels)
    ctx.putImageData(state.baseImageData, 0, 0);
    
    // 2. Reset Transform to Physical Pixels (Identity)
    // This prevents double-scaling issues on Retina/High-DPI screens
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    
    // 3. Calculate Position in Physical Pixels
    const x = Math.floor(progress * state.activeCanvas.width);
    
    // 4. Draw Playhead
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, state.activeCanvas.height);
    ctx.strokeStyle = '#000000'; 
    ctx.lineWidth = 2 * window.devicePixelRatio; 
    ctx.stroke();
    
    state.playheadAnimationId = requestAnimationFrame(animatePlayhead);
}

function updateButtonStates() {
    const ready = state.trackA.buffer && state.trackB.buffer;
    if (elements.processBtn) elements.processBtn.disabled = !ready;
}

function exportWAV() {
    if(!state.outputBuffer) return;
    const wav = audioBufferToWav(state.outputBuffer);
    const blob = new Blob([wav], {type:'audio/wav'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href=url; a.download='flowstate_mix.wav'; a.click();
}
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const length = buffer.length * numChannels * 2;
    const bufferLength = 44 + length;
    const view = new DataView(new ArrayBuffer(bufferLength));
    writeString(view, 0, 'RIFF'); view.setUint32(4, 36+length, true); writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true); view.setUint32(24, 22050, true);
    view.setUint32(28, 22050*numChannels*2, true); view.setUint16(32, numChannels*2, true);
    view.setUint16(34, 16, true); writeString(view, 36, 'data'); view.setUint32(40, length, true);
    let offset = 44;
    for (let i=0; i<buffer.length; i++) for (let c=0; c<numChannels; c++) {
        let s = Math.max(-1, Math.min(1, buffer.getChannelData(c)[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true); offset += 2;
    }
    return view.buffer;
}
function writeString(v, o, s) { for (let i=0; i<s.length; i++) v.setUint8(o+i, s.charCodeAt(i)); }

function exportLogs() {
    console.log('[FlowState] Generating logs...');
    const logs = {
        timestamp: new Date().toISOString(),
        configuration: CONFIG,
        model: {
            type: state.modelType,
            status: elements.modelStatusText?.textContent || 'UNKNOWN'
        },
        session: {
            transitionDuration: state.transitionDuration,
            filterIntensity: state.filterIntensity,
            mode: elements.compareModeInfo?.textContent || 'UNKNOWN'
        },
        trackA: {
            filename: state.trackA.filename,
            classification: state.trackA.classification,
            confidence: state.trackA.confidence
        },
        trackB: {
            filename: state.trackB.filename,
            classification: state.trackB.classification,
            confidence: state.trackB.confidence
        },
        features: {
            // These would be populated if you store them in state during analyzeFeatures
            // currently they are just in the DOM, so we can scrape them or leave as future work
            fluxA: document.getElementById('featFluxA')?.textContent,
            zcrA: document.getElementById('featZcrA')?.textContent,
            fluxB: document.getElementById('featFluxB')?.textContent,
            zcrB: document.getElementById('featZcrB')?.textContent
        }
    };

    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flowstate_logs_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Ensure AudioContext is resumed (fixing Safari/Apple autoplay policy)
document.addEventListener('click', async () => {
    if (state.audioContext && state.audioContext.state === 'suspended') {
        await state.audioContext.resume();
        console.log('[FlowState] AudioContext resumed by user interaction.');
    }
}, { once: true }); // Only try once


document.addEventListener('DOMContentLoaded', init);
