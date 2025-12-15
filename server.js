/**
 * KremCheats Server - Render.com Edition
 * Fixed version with better error handling and tfjs fallback
 */

const express = require('express');
const cors = require('cors');

const app = express();

app.use(cors({ origin: '*', methods: ['GET', 'POST'] }));
app.use(express.json({ limit: '10mb' }));

const CONFIG = {
    port: process.env.PORT || 3000,
    version: '2.2.0',
    modelLoaded: false,
    startTime: Date.now(),
    loadError: null,
    backend: 'none'
};

// Track connected users
const stats = {
    totalConnections: 0,
    activeUsers: new Set(),
    totalDetections: 0,
    peakUsers: 0
};

let tf = null, cocoSsd = null, detectionModel = null;

// Load TensorFlow with fallback
async function loadModel() {
    console.log('[KremCheats] Starting model load...');
    
    // Try tfjs-node first (faster, uses native bindings)
    try {
        console.log('[KremCheats] Attempting to load @tensorflow/tfjs-node...');
        tf = require('@tensorflow/tfjs-node');
        CONFIG.backend = 'tfjs-node';
        console.log('[KremCheats] tfjs-node loaded successfully');
    } catch (nodeError) {
        console.log('[KremCheats] tfjs-node failed:', nodeError.message);
        console.log('[KremCheats] Falling back to @tensorflow/tfjs...');
        
        // Fallback to pure JS tfjs (slower but more compatible)
        try {
            tf = require('@tensorflow/tfjs');
            CONFIG.backend = 'tfjs';
            console.log('[KremCheats] tfjs (pure JS) loaded successfully');
        } catch (jsError) {
            console.error('[KremCheats] Both TensorFlow backends failed!');
            console.error('[KremCheats] tfjs-node error:', nodeError.message);
            console.error('[KremCheats] tfjs error:', jsError.message);
            CONFIG.loadError = 'TensorFlow failed to load: ' + nodeError.message;
            return;
        }
    }
    
    // Load Coco-SSD model
    try {
        console.log('[KremCheats] Loading @tensorflow-models/coco-ssd...');
        cocoSsd = require('@tensorflow-models/coco-ssd');
        
        console.log('[KremCheats] Loading detection model (lite_mobilenet_v2)...');
        detectionModel = await cocoSsd.load({ 
            base: 'lite_mobilenet_v2'
        });
        
        CONFIG.modelLoaded = true;
        CONFIG.loadError = null;
        console.log('[KremCheats] Model loaded successfully!');
        console.log('[KremCheats] Backend:', CONFIG.backend);
        console.log('[KremCheats] Ready for detections!');
    } catch (modelError) {
        console.error('[KremCheats] Model loading failed:', modelError.message);
        console.error('[KremCheats] Full error:', modelError);
        CONFIG.loadError = 'Model failed to load: ' + modelError.message;
    }
}

// Format uptime
function formatUptime(ms) {
    const seconds = Math.floor(ms / 1000);
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
}

// ==================== API ENDPOINTS ====================

// Root endpoint - JSON status
app.get('/', (req, res) => {
    res.json({
        name: 'KremCheats GPU Server',
        version: CONFIG.version,
        status: 'online',
        modelLoaded: CONFIG.modelLoaded,
        backend: CONFIG.backend,
        activeUsers: stats.activeUsers.size,
        loadError: CONFIG.loadError
    });
});

// Health check
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        modelReady: CONFIG.modelLoaded,
        backend: CONFIG.backend,
        error: CONFIG.loadError
    });
});

// Stats endpoint
app.get('/stats', (req, res) => {
    res.json({
        activeUsers: stats.activeUsers.size,
        peakUsers: stats.peakUsers,
        totalConnections: stats.totalConnections,
        totalDetections: stats.totalDetections,
        uptime: formatUptime(Date.now() - CONFIG.startTime),
        modelLoaded: CONFIG.modelLoaded,
        backend: CONFIG.backend,
        error: CONFIG.loadError
    });
});

// Connect endpoint - track users
app.post('/connect', (req, res) => {
    const userId = req.body.userId || `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    stats.activeUsers.add(userId);
    stats.totalConnections++;
    if (stats.activeUsers.size > stats.peakUsers) {
        stats.peakUsers = stats.activeUsers.size;
    }
    console.log(`[KremCheats] User connected: ${userId} (${stats.activeUsers.size} active)`);
    res.json({ 
        success: true, 
        userId, 
        activeUsers: stats.activeUsers.size,
        modelReady: CONFIG.modelLoaded
    });
});

// Disconnect endpoint
app.post('/disconnect', (req, res) => {
    const userId = req.body.userId;
    if (userId) {
        stats.activeUsers.delete(userId);
        console.log(`[KremCheats] User disconnected: ${userId} (${stats.activeUsers.size} active)`);
    }
    res.json({ success: true, activeUsers: stats.activeUsers.size });
});

// Heartbeat to keep user active
app.post('/heartbeat', (req, res) => {
    const userId = req.body.userId;
    if (userId && !stats.activeUsers.has(userId)) {
        stats.activeUsers.add(userId);
        stats.totalConnections++;
    }
    res.json({ 
        success: true, 
        activeUsers: stats.activeUsers.size,
        modelReady: CONFIG.modelLoaded
    });
});

// Detection endpoint
app.post('/detect', async (req, res) => {
    // Return empty if model not loaded
    if (!CONFIG.modelLoaded || !detectionModel) {
        return res.json({ 
            predictions: [],
            error: CONFIG.loadError || 'Model not loaded'
        });
    }
    
    try {
        let img = req.body.image || '';
        
        // Handle base64 data URL format
        if (img.includes(',')) {
            img = img.split(',')[1];
        }
        
        if (!img || img.length < 100) {
            return res.json({ predictions: [], error: 'Invalid image data' });
        }
        
        const buffer = Buffer.from(img, 'base64');
        
        // Decode image based on backend
        let tensor;
        if (CONFIG.backend === 'tfjs-node') {
            tensor = tf.node.decodeImage(buffer, 3);
        } else {
            // For pure tfjs, we need a different approach
            // This is a simplified version - may need canvas for full support
            const { createCanvas, loadImage } = require('canvas');
            const image = await loadImage(buffer);
            const canvas = createCanvas(image.width, image.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            const imageData = ctx.getImageData(0, 0, image.width, image.height);
            tensor = tf.browser.fromPixels({ data: imageData.data, width: image.width, height: image.height });
        }
        
        const maxDetections = req.body.maxDetections || 5;
        const confidence = req.body.confidence || 0.35;
        
        const predictions = await detectionModel.detect(tensor, maxDetections, confidence);
        
        // Clean up tensor to prevent memory leaks
        tensor.dispose();
        
        stats.totalDetections++;
        
        // Filter for persons only
        const personPredictions = predictions.filter(p => p.class === 'person');
        
        res.json({ 
            predictions: personPredictions,
            count: personPredictions.length,
            backend: CONFIG.backend
        });
        
    } catch (e) {
        console.error('[KremCheats] Detection error:', e.message);
        res.json({ 
            predictions: [], 
            error: e.message 
        });
    }
});

// Retry loading model endpoint
app.post('/reload-model', async (req, res) => {
    console.log('[KremCheats] Manual model reload requested...');
    CONFIG.modelLoaded = false;
    CONFIG.loadError = null;
    await loadModel();
    res.json({
        success: CONFIG.modelLoaded,
        backend: CONFIG.backend,
        error: CONFIG.loadError
    });
});

// ==================== STATUS PAGE ====================
app.get('/status', (req, res) => {
    const uptime = Date.now() - CONFIG.startTime;
    const statusClass = CONFIG.modelLoaded ? 'status-online' : 'status-offline';
    const statusText = CONFIG.modelLoaded ? '● OPERATIONAL' : '● MODEL NOT LOADED';
    
    res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KremCheats Server Status</title>
    <meta http-equiv="refresh" content="10">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header { text-align: center; padding: 40px 0; }
        .logo {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(135deg, #39ff14, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-top: 10px; font-size: 14px; }
        .status-badge {
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 20px;
            font-size: 16px;
        }
        .status-online {
            background: rgba(57, 255, 20, 0.2);
            color: #39ff14;
            border: 2px solid #39ff14;
            box-shadow: 0 0 20px rgba(57, 255, 20, 0.3);
        }
        .status-offline {
            background: rgba(255, 50, 50, 0.2);
            color: #ff3232;
            border: 2px solid #ff3232;
            box-shadow: 0 0 20px rgba(255, 50, 50, 0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(57, 255, 20, 0.2);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #39ff14;
        }
        .stat-label {
            color: #888;
            margin-top: 10px;
            font-size: 14px;
            text-transform: uppercase;
        }
        .error-box {
            background: rgba(255, 50, 50, 0.1);
            border: 1px solid rgba(255, 50, 50, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
            color: #ff6b6b;
        }
        .error-box h3 { color: #ff3232; margin-bottom: 10px; }
        .info-box {
            background: rgba(57, 255, 20, 0.05);
            border: 1px solid rgba(57, 255, 20, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }
        .info-box h3 { color: #39ff14; margin-bottom: 10px; }
        code {
            background: rgba(0,0,0,0.3);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">KREMCHEATS</div>
            <div class="subtitle">GPU Detection Server v${CONFIG.version}</div>
            <div class="status-badge ${statusClass}">${statusText}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats.activeUsers.size}</div>
                <div class="stat-label">Active Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.totalDetections}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatUptime(uptime)}</div>
                <div class="stat-label">Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${CONFIG.backend || 'N/A'}</div>
                <div class="stat-label">Backend</div>
            </div>
        </div>
        
        ${CONFIG.loadError ? `
        <div class="error-box">
            <h3>⚠️ Model Load Error</h3>
            <p>${CONFIG.loadError}</p>
            <p style="margin-top: 10px; font-size: 12px; color: #888;">
                Try redeploying or check Render logs for more details.
            </p>
        </div>
        ` : ''}
        
        <div class="info-box">
            <h3>📡 API Endpoints</h3>
            <p><code>GET /</code> - Server status (JSON)</p>
            <p><code>GET /health</code> - Health check</p>
            <p><code>GET /stats</code> - Detailed statistics</p>
            <p><code>POST /connect</code> - Register user</p>
            <p><code>POST /detect</code> - Run detection</p>
            <p><code>POST /reload-model</code> - Retry model loading</p>
        </div>
    </div>
</body>
</html>
    `);
});

// Start server
console.log('[KremCheats] Starting server...');
loadModel().then(() => {
    app.listen(CONFIG.port, '0.0.0.0', () => {
        console.log(`[KremCheats] Server running on port ${CONFIG.port}`);
        console.log(`[KremCheats] Model loaded: ${CONFIG.modelLoaded}`);
        console.log(`[KremCheats] Backend: ${CONFIG.backend}`);
        if (CONFIG.loadError) {
            console.log(`[KremCheats] Error: ${CONFIG.loadError}`);
        }
    });
});
