/**
 * KremCheats Server v2.3.0
 * Uses pure TensorFlow.js (no native bindings) for maximum compatibility
 */

const express = require('express');
const cors = require('cors');
const Jimp = require('jimp');

const app = express();

app.use(cors({ origin: '*', methods: ['GET', 'POST'] }));
app.use(express.json({ limit: '10mb' }));

const CONFIG = {
    port: process.env.PORT || 3000,
    version: '2.3.0',
    modelLoaded: false,
    startTime: Date.now(),
    loadError: null
};

const stats = {
    totalConnections: 0,
    activeUsers: new Set(),
    totalDetections: 0,
    peakUsers: 0
};

let tf = null;
let cocoSsd = null;
let detectionModel = null;

// Load model
async function loadModel() {
    console.log('[KremCheats] Starting model load...');
    
    try {
        // Use pure tfjs (no native bindings - more compatible)
        console.log('[KremCheats] Loading @tensorflow/tfjs...');
        tf = require('@tensorflow/tfjs');
        
        // Set backend to CPU (most compatible)
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('[KremCheats] TensorFlow.js ready, backend:', tf.getBackend());
        
        // Load coco-ssd
        console.log('[KremCheats] Loading @tensorflow-models/coco-ssd...');
        cocoSsd = require('@tensorflow-models/coco-ssd');
        
        console.log('[KremCheats] Loading detection model...');
        detectionModel = await cocoSsd.load({
            base: 'lite_mobilenet_v2'
        });
        
        CONFIG.modelLoaded = true;
        CONFIG.loadError = null;
        console.log('[KremCheats] Model loaded successfully!');
        
    } catch (e) {
        console.error('[KremCheats] Model load error:', e.message);
        console.error(e.stack);
        CONFIG.loadError = e.message;
        CONFIG.modelLoaded = false;
    }
}

// Decode base64 image to tensor using Jimp
async function decodeImage(base64Data) {
    // Remove data URL prefix if present
    if (base64Data.includes(',')) {
        base64Data = base64Data.split(',')[1];
    }
    
    const buffer = Buffer.from(base64Data, 'base64');
    const image = await Jimp.read(buffer);
    
    const width = image.getWidth();
    const height = image.getHeight();
    
    // Get pixel data as RGB array
    const pixels = new Uint8Array(width * height * 3);
    let idx = 0;
    
    image.scan(0, 0, width, height, function(x, y, i) {
        pixels[idx++] = this.bitmap.data[i];     // R
        pixels[idx++] = this.bitmap.data[i + 1]; // G
        pixels[idx++] = this.bitmap.data[i + 2]; // B
    });
    
    // Create tensor [1, height, width, 3]
    return tf.tensor3d(pixels, [height, width, 3], 'int32');
}

function formatUptime(ms) {
    const s = Math.floor(ms / 1000);
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m ${s % 60}s`;
}

// ==================== ENDPOINTS ====================

app.get('/', (req, res) => {
    res.json({
        name: 'KremCheats GPU Server',
        version: CONFIG.version,
        status: 'online',
        modelLoaded: CONFIG.modelLoaded,
        backend: 'tfjs-cpu',
        activeUsers: stats.activeUsers.size,
        error: CONFIG.loadError
    });
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', modelReady: CONFIG.modelLoaded });
});

app.get('/stats', (req, res) => {
    res.json({
        activeUsers: stats.activeUsers.size,
        peakUsers: stats.peakUsers,
        totalConnections: stats.totalConnections,
        totalDetections: stats.totalDetections,
        uptime: formatUptime(Date.now() - CONFIG.startTime),
        modelLoaded: CONFIG.modelLoaded
    });
});

app.post('/connect', (req, res) => {
    const userId = req.body.userId || `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    stats.activeUsers.add(userId);
    stats.totalConnections++;
    if (stats.activeUsers.size > stats.peakUsers) stats.peakUsers = stats.activeUsers.size;
    console.log(`[KremCheats] Connected: ${userId} (${stats.activeUsers.size} active)`);
    res.json({ success: true, userId, activeUsers: stats.activeUsers.size, modelReady: CONFIG.modelLoaded });
});

app.post('/disconnect', (req, res) => {
    if (req.body.userId) {
        stats.activeUsers.delete(req.body.userId);
        console.log(`[KremCheats] Disconnected: ${req.body.userId}`);
    }
    res.json({ success: true });
});

app.post('/heartbeat', (req, res) => {
    const userId = req.body.userId;
    if (userId && !stats.activeUsers.has(userId)) {
        stats.activeUsers.add(userId);
    }
    res.json({ success: true, modelReady: CONFIG.modelLoaded });
});

// Detection endpoint
app.post('/detect', async (req, res) => {
    if (!CONFIG.modelLoaded || !detectionModel) {
        return res.json({ predictions: [], error: 'Model not loaded' });
    }
    
    try {
        const imageData = req.body.image;
        if (!imageData || imageData.length < 100) {
            return res.json({ predictions: [], error: 'Invalid image' });
        }
        
        // Decode image
        const tensor = await decodeImage(imageData);
        
        // Run detection
        const maxDetections = req.body.maxDetections || 5;
        const minScore = req.body.confidence || 0.35;
        
        const predictions = await detectionModel.detect(tensor, maxDetections, minScore);
        
        // Cleanup
        tensor.dispose();
        
        stats.totalDetections++;
        
        // Filter for persons
        const persons = predictions.filter(p => p.class === 'person');
        
        res.json({ predictions: persons, count: persons.length });
        
    } catch (e) {
        console.error('[KremCheats] Detection error:', e.message);
        res.json({ predictions: [], error: e.message });
    }
});

// Status page
app.get('/status', (req, res) => {
    const uptime = Date.now() - CONFIG.startTime;
    res.send(`
<!DOCTYPE html>
<html>
<head>
    <title>KremCheats Status</title>
    <meta http-equiv="refresh" content="10">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            min-height: 100vh;
            color: #fff;
            padding: 40px 20px;
        }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        .logo {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(135deg, #39ff14, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .version { color: #666; margin: 10px 0 30px; }
        .status {
            display: inline-block;
            padding: 12px 30px;
            border-radius: 30px;
            font-weight: bold;
            font-size: 18px;
        }
        .online { background: rgba(57,255,20,0.2); color: #39ff14; border: 2px solid #39ff14; }
        .offline { background: rgba(255,50,50,0.2); color: #ff3232; border: 2px solid #ff3232; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .stat {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(57,255,20,0.2);
            border-radius: 15px;
            padding: 25px;
        }
        .stat-value { font-size: 32px; color: #39ff14; font-weight: bold; }
        .stat-label { color: #888; margin-top: 8px; font-size: 12px; text-transform: uppercase; }
        .error {
            background: rgba(255,50,50,0.1);
            border: 1px solid #ff3232;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            color: #ff6b6b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">KREMCHEATS</div>
        <div class="version">Server v${CONFIG.version}</div>
        <div class="status ${CONFIG.modelLoaded ? 'online' : 'offline'}">
            ${CONFIG.modelLoaded ? '● MODEL READY' : '● MODEL NOT LOADED'}
        </div>
        ${CONFIG.loadError ? `<div class="error"><strong>Error:</strong> ${CONFIG.loadError}</div>` : ''}
        <div class="stats">
            <div class="stat"><div class="stat-value">${stats.activeUsers.size}</div><div class="stat-label">Active Users</div></div>
            <div class="stat"><div class="stat-value">${stats.totalDetections}</div><div class="stat-label">Detections</div></div>
            <div class="stat"><div class="stat-value">${formatUptime(uptime)}</div><div class="stat-label">Uptime</div></div>
            <div class="stat"><div class="stat-value">${stats.peakUsers}</div><div class="stat-label">Peak Users</div></div>
        </div>
    </div>
</body>
</html>
    `);
});

// Start
console.log('[KremCheats] Starting...');
loadModel().then(() => {
    app.listen(CONFIG.port, '0.0.0.0', () => {
        console.log(`[KremCheats] Running on port ${CONFIG.port}`);
        console.log(`[KremCheats] Model: ${CONFIG.modelLoaded ? 'READY' : 'FAILED'}`);
        if (CONFIG.loadError) console.log(`[KremCheats] Error: ${CONFIG.loadError}`);
    });
});
