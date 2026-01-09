/**
 * FlowState Node.js Test Script - With FFT
 * Tests the ONNX model with proper mel spectrogram
 */

const ort = require('onnxruntime-node');
const fs = require('fs');
const { execSync } = require('child_process');

// Config
const CONFIG = {
    MODEL_PATH: './model/flowstate_brain_v2.onnx',
    SAMPLE_RATE: 22050,
    MEL_BANDS: 128,
    SPEC_WIDTH: 128,
    TARGET_SAMPLES: 65536,
};

// Simple FFT (Cooley-Tukey radix-2)
function fft(input) {
    const n = input.length;
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    
    // Bit-reversal permutation
    for (let i = 0; i < n; i++) {
        let j = 0;
        for (let k = 0; k < Math.log2(n); k++) {
            j = (j << 1) | ((i >> k) & 1);
        }
        real[j] = input[i];
    }
    
    // Cooley-Tukey FFT
    for (let size = 2; size <= n; size *= 2) {
        const halfSize = size / 2;
        const angleStep = -2 * Math.PI / size;
        
        for (let i = 0; i < n; i += size) {
            for (let j = 0; j < halfSize; j++) {
                const angle = angleStep * j;
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                
                const evenIdx = i + j;
                const oddIdx = i + j + halfSize;
                
                const tReal = cos * real[oddIdx] - sin * imag[oddIdx];
                const tImag = sin * real[oddIdx] + cos * imag[oddIdx];
                
                real[oddIdx] = real[evenIdx] - tReal;
                imag[oddIdx] = imag[evenIdx] - tImag;
                real[evenIdx] = real[evenIdx] + tReal;
                imag[evenIdx] = imag[evenIdx] + tImag;
            }
        }
    }
    
    return { real, imag };
}

// Create mel filterbank
function createMelFilterbank(nFft, nMels, sr) {
    const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
    const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);
    
    const melMin = hzToMel(0);
    const melMax = hzToMel(sr / 2);
    
    const nMelPoints = nMels + 2;
    const melPoints = [];
    for (let i = 0; i < nMelPoints; i++) {
        melPoints.push(melMin + (melMax - melMin) * i / (nMelPoints - 1));
    }
    
    const hzPoints = melPoints.map(melToHz);
    const binPoints = hzPoints.map(hz => Math.floor((nFft + 1) * hz / sr));
    
    const filters = [];
    for (let m = 0; m < nMels; m++) {
        const filter = new Float32Array(nFft / 2 + 1);
        const startBin = binPoints[m];
        const centerBin = binPoints[m + 1];
        const endBin = binPoints[m + 2];
        
        for (let k = startBin; k < centerBin; k++) {
            if (k >= 0 && k < filter.length && centerBin > startBin) {
                filter[k] = (k - startBin) / (centerBin - startBin);
            }
        }
        for (let k = centerBin; k < endBin; k++) {
            if (k >= 0 && k < filter.length && endBin > centerBin) {
                filter[k] = (endBin - k) / (endBin - centerBin);
            }
        }
        filters.push(filter);
    }
    return filters;
}

// Load audio
function loadAudio(filePath) {
    const tmpFile = '/tmp/flowstate_audio.raw';
    execSync(`ffmpeg -y -i "${filePath}" -ar ${CONFIG.SAMPLE_RATE} -ac 1 -f f32le -acodec pcm_f32le "${tmpFile}"`, { stdio: 'pipe' });
    const buffer = fs.readFileSync(tmpFile);
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
}

// Generate mel spectrogram with FFT
function generateMelSpectrogram(samples, segment = 'start') {
    // Extract chunk
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
    
    const nFft = 2048;
    const hopLength = 512;
    const nMels = CONFIG.MEL_BANDS;
    const sr = CONFIG.SAMPLE_RATE;
    
    const melFilters = createMelFilterbank(nFft, nMels, sr);
    const nFrames = Math.floor((chunk.length - nFft) / hopLength) + 1;
    const targetFrames = CONFIG.SPEC_WIDTH;
    const melSpec = new Float32Array(nMels * targetFrames);
    
    for (let t = 0; t < Math.min(nFrames, targetFrames); t++) {
        const frameStart = t * hopLength;
        
        // Extract frame with Hann window
        const frame = new Float32Array(nFft);
        for (let i = 0; i < nFft; i++) {
            const sample = (frameStart + i < chunk.length) ? chunk[frameStart + i] : 0;
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)));
            frame[i] = sample * window;
        }
        
        // FFT
        const { real, imag } = fft(frame);
        
        // Power spectrum
        const powerSpec = new Float32Array(nFft / 2 + 1);
        for (let k = 0; k <= nFft / 2; k++) {
            powerSpec[k] = real[k] * real[k] + imag[k] * imag[k];
        }
        
        // Apply mel filters
        for (let m = 0; m < nMels; m++) {
            let energy = 0;
            for (let k = 0; k < powerSpec.length; k++) {
                energy += powerSpec[k] * melFilters[m][k];
            }
            melSpec[m * targetFrames + t] = energy;
        }
    }
    
    // Convert to dB
    let maxPower = 1e-10;
    for (let i = 0; i < melSpec.length; i++) {
        if (melSpec[i] > maxPower) maxPower = melSpec[i];
    }
    
    for (let i = 0; i < melSpec.length; i++) {
        let db = 10 * Math.log10(Math.max(melSpec[i], 1e-10) / maxPower);
        melSpec[i] = Math.max(db, -80.0);
    }
    
    // Min-Max normalize
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
    
    // Stats
    let sum = 0;
    for (let i = 0; i < melSpec.length; i++) sum += melSpec[i];
    console.log(`  dB=[${minVal.toFixed(1)}, ${maxVal.toFixed(1)}], mean=${(sum/melSpec.length).toFixed(4)}`);
    
    return melSpec;
}

async function main() {
    const trackAPath = process.argv[2] || '/Users/jeff/Desktop/FlowState/research_data/class_ambient/classical.00001.wav';
    const trackBPath = process.argv[3] || '/Users/jeff/Desktop/FlowState/research_data/class_beat/hiphop.00001.wav';
    
    console.log('=== FlowState Node.js Test (FFT) ===\n');
    
    // Load model
    console.log('Loading ONNX model...');
    const session = await ort.InferenceSession.create(CONFIG.MODEL_PATH);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    console.log(`  Input: ${inputName}, Output: ${outputName}\n`);
    
    // Process Track A
    console.log('=== TRACK A (end segment - classical) ===');
    const samplesA = loadAudio(trackAPath);
    console.log(`  Loaded ${samplesA.length} samples`);
    const specA = generateMelSpectrogram(samplesA, 'end');
    const tensorA = new ort.Tensor('float32', specA, [1, 1, 128, 128]);
    const resultA = await session.run({ [inputName]: tensorA });
    const predA = resultA[outputName].data[0];
    console.log(`  Prediction: ${predA.toFixed(6)} -> ${predA < 0.5 ? 'RHYTHMIC' : 'HARMONIC'}\n`);
    
    // Process Track B
    console.log('=== TRACK B (start segment - hiphop) ===');
    const samplesB = loadAudio(trackBPath);
    console.log(`  Loaded ${samplesB.length} samples`);
    const specB = generateMelSpectrogram(samplesB, 'start');
    const tensorB = new ort.Tensor('float32', specB, [1, 1, 128, 128]);
    const resultB = await session.run({ [inputName]: tensorB });
    const predB = resultB[outputName].data[0];
    console.log(`  Prediction: ${predB.toFixed(6)} -> ${predB < 0.5 ? 'RHYTHMIC' : 'HARMONIC'}\n`);
    
    // Mode selection
    const typeA = predA < 0.5 ? 'RHYTHMIC' : 'HARMONIC';
    const typeB = predB < 0.5 ? 'RHYTHMIC' : 'HARMONIC';
    
    let mode = 'DROP';
    if (typeA === 'RHYTHMIC' && typeB === 'RHYTHMIC') mode = 'BEAT_MATCH';
    else if (typeA === 'HARMONIC' && typeB === 'HARMONIC') mode = 'LIQUID_BLEND';
    else if (typeA === 'RHYTHMIC' && typeB === 'HARMONIC') mode = 'ECHO_FREEZE';
    else mode = 'DROP';
    
    console.log('=== RESULT ===');
    console.log(`A=${predA.toFixed(4)} (${typeA}) | B=${predB.toFixed(4)} (${typeB}) -> Mode: ${mode}`);
    
    // Compare with expected (Python output was: A=1.0000, B=-0.0000 -> DROP)
    console.log('\n=== EXPECTED (from Python) ===');
    console.log('A=1.0000 (HARMONIC) | B=0.0000 (RHYTHMIC) -> Mode: DROP');
}

main().catch(console.error);
