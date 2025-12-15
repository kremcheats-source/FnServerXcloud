/**
 * KremCheats Server - Render.com Edition
 * With Public Status Page
 */

const express = require('express');
const cors = require('cors');

const app = express();

app.use(cors({ origin: '*', methods: ['GET', 'POST'] }));
app.use(express.json({ limit: '10mb' }));

const CONFIG = {
    port: process.env.PORT || 3000,
    version: '2.1.0',
    modelLoaded: false,
    startTime: Date.now()
};

// Track connected users
const stats = {
    totalConnections: 0,
    activeUsers: new Set(),
    totalDetections: 0,
    peakUsers: 0
};

let tf = null, cocoSsd = null, detectionModel = null;

// Load TensorFlow
async function loadModel() {
    try {
        console.log('[KremCheats] Loading TensorFlow.js...');
        tf = require('@tensorflow/tfjs-node');
        console.log('[KremCheats] Loading model...');
        cocoSsd = require('@tensorflow-models/coco-ssd');
        detectionModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        CONFIG.modelLoaded = true;
        console.log('[KremCheats] Ready!');
    } catch (e) {
        console.error('[KremCheats] Error:', e.message);
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

// ==================== STATUS PAGE ====================
app.get('/status', (req, res) => {
    const uptime = Date.now() - CONFIG.startTime;
    
    res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KremCheats Status</title>
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
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; padding: 40px 0; }
        .logo {
            font-size: 48px;
            font-weight: bold;
            background: linear-gradient(135deg, #39ff14, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(57, 255, 20, 0.5);
        }
        .subtitle { color: #888; margin-top: 10px; font-size: 14px; }
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 20px;
            font-size: 14px;
        }
        .status-online {
            background: rgba(57, 255, 20, 0.2);
            color: #39ff14;
            border: 1px solid #39ff14;
        }
        .status-offline {
            background: rgba(255, 50, 50, 0.2);
            color: #ff3232;
            border: 1px solid #ff3232;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(57, 255, 20, 0.2);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .stat-card:hover {
            border-color: #39ff14;
            box-shadow: 0 0 20px rgba(57, 255, 20, 0.2);
            transform: translateY(-5px);
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            color: #39ff14;
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
        }
        .stat-label {
            color: #888;
            margin-top: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .info-section {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
        }
        .info-title { color: #39ff14; font-size: 18px; margin-bottom: 15px; }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .info-row:last-child { border-bottom: none; }
        .info-label { color: #888; }
        .info-value { color: #fff; font-weight: 500; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .footer { text-align: center; margin-top: 40px; color: #555; font-size: 12px; }
        .refresh-note { color: #39ff14; font-size: 11px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">KremCheats</div>
            <div class="subtitle">GPU Detection Server</div>
            <div class="status-badge ${CONFIG.modelLoaded ? 'status-online' : 'status-offline'}">
                <span class="pulse">●</span> ${CONFIG.modelLoaded ? 'OPERATIONAL' : 'LOADING'}
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats.activeUsers.size}</div>
                <div class="stat-label">Active Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.peakUsers}</div>
                <div class="stat-label">Peak Users</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.totalConnections}</div>
                <div class="stat-label">Total Connections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.totalDetections.toLocaleString()}</div>
                <div class="stat-label">Total Detections</div>
            </div>
        </div>

        <div class="info-section">
            <div class="info-title">Server Information</div>
            <div class="info-row">
                <span class="info-label">Status</span>
                <span class="info-value" style="color: ${CONFIG.modelLoaded ? '#39ff14' : '#ffaa00'}">
                    ${CONFIG.modelLoaded ? '● Online' : '● Loading Model...'}
                </span>
            </div>
            <div class="info-row">
                <span class="info-label">Version</span>
                <span class="info-value">v${CONFIG.version}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Uptime</span>
                <span class="info-value">${formatUptime(uptime)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Model</span>
                <span class="info-value">${CONFIG.modelLoaded ? 'COCO-SSD (MobileNet)' : 'Loading...'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Backend</span>
                <span class="info-value">${tf ? tf.getBackend() : 'Initializing...'}</span>
            </div>
        </div>

        <div class="footer">
            KremCheats &copy; 2024 | All Rights Reserved
            <div class="refresh-note">Auto-refreshes every 10 seconds</div>
        </div>
    </div>
</body>
</html>
    `);
});

// ==================== API ROUTES ====================
app.get('/', (req, res) => {
    res.json({
        name: 'KremCheats GPU Server',
        version: CONFIG.version,
        status: 'online',
        modelLoaded: CONFIG.modelLoaded,
        backend: tf ? tf.getBackend() : 'loading',
        activeUsers: stats.activeUsers.size
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

// Connect endpoint - track users
app.post('/connect', (req, res) => {
    const userId = req.body.userId || `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    stats.activeUsers.add(userId);
    stats.totalConnections++;
    if (stats.activeUsers.size > stats.peakUsers) {
        stats.peakUsers = stats.activeUsers.size;
    }
    console.log(`[KremCheats] User connected: ${userId} (${stats.activeUsers.size} active)`);
    res.json({ success: true, userId, activeUsers: stats.activeUsers.size });
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
    res.json({ success: true, activeUsers: stats.activeUsers.size });
});

// Detection endpoint
app.post('/detect', async (req, res) => {
    if (!CONFIG.modelLoaded) return res.json({ predictions: [] });
    
    try {
        let img = req.body.image || '';
        if (img.includes(',')) img = img.split(',')[1];
        
        const tensor = tf.node.decodeImage(Buffer.from(img, 'base64'), 3);
        const preds = await detectionModel.detect(tensor, req.body.maxDetections || 5, req.body.confidence || 0.35);
        tensor.dispose();
        
        stats.totalDetections++;
        
        res.json({ predictions: preds.filter(p => p.class === 'person') });
    } catch (e) {
        res.json({ predictions: [] });
    }
});

// Start server
loadModel().then(() => {
    app.listen(CONFIG.port, '0.0.0.0', () => {
        console.log(`[KremCheats] Server running on port ${CONFIG.port}`);
        console.log(`[KremCheats] Status page: /status`);
    });
});
