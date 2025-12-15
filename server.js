/**
 * KremCheats Server - Render.com Edition
 * Optimized for Render's free tier (CPU-based)
 */

const express = require('express');
const cors = require('cors');

const app = express();

// CORS - allow all origins
app.use(cors({ origin: '*', methods: ['GET', 'POST'] }));
app.use(express.json({ limit: '10mb' }));

const CONFIG = {
    port: process.env.PORT || 3000,
    version: '2.1.0',
    modelLoaded: false
};

let tf = null;
let cocoSsd = null;
let detectionModel = null;

// Load TensorFlow (CPU mode for Render free tier)
async function loadModel() {
    try {
        console.log('[KremCheats] Loading TensorFlow.js (CPU)...');
        tf = require('@tensorflow/tfjs-node');
        console.log('[KremCheats] Backend:', tf.getBackend());
        
        console.log('[KremCheats] Loading Coco-SSD model...');
        cocoSsd = require('@tensorflow-models/coco-ssd');
        detectionModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        
        CONFIG.modelLoaded = true;
        console.log('[KremCheats] Model loaded!');
        return true;
    } catch (error) {
        console.error('[KremCheats] Load error:', error.message);
        return false;
    }
}

// Routes
app.get('/', (req, res) => {
    res.json({
        name: 'KremCheats GPU Server',
        version: CONFIG.version,
        status: 'online',
        modelLoaded: CONFIG.modelLoaded,
        backend: tf ? tf.getBackend() : 'loading'
    });
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', modelReady: CONFIG.modelLoaded });
});

app.post('/detect', async (req, res) => {
    if (!CONFIG.modelLoaded) {
        return res.status(503).json({ error: 'Model loading', predictions: [] });
    }
    
    try {
        const { image, confidence = 0.35, maxDetections = 5 } = req.body;
        
        if (!image) {
            return res.status(400).json({ error: 'No image', predictions: [] });
        }
        
        // Remove data URL prefix if present
        let imageData = image;
        if (image.includes(',')) {
            imageData = image.split(',')[1];
        }
        
        const buffer = Buffer.from(imageData, 'base64');
        const tensor = tf.node.decodeImage(buffer, 3);
        
        const predictions = await detectionModel.detect(tensor, maxDetections, confidence);
        tensor.dispose();
        
        const persons = predictions.filter(p => p.class === 'person');
        
        res.json({ predictions: persons });
    } catch (error) {
        console.error('[KremCheats] Error:', error.message);
        res.json({ error: error.message, predictions: [] });
    }
});

// Start
async function start() {
    await loadModel();
    app.listen(CONFIG.port, '0.0.0.0', () => {
        console.log(`[KremCheats] Running on port ${CONFIG.port}`);
    });
}

start();
