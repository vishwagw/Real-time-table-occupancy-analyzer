import webview
import threading
from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import os
import tempfile
from pathlib import Path

# Try to import ultralytics, install if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

# HTML template as a string
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table Occupancy Detection with YOLO</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
    <style>
        .loading-spinner {
            border: 2px solid #f3f3f3;
            border-top: 2px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .video-container {
            position: relative;
            display: inline-block;
        }
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .timeline {
            width: 100%;
            height: 60px;
            background: #f3f4f6;
            border-radius: 10px;
            position: relative;
            margin-top: 10px;
        }
        .timeline-progress {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            border-radius: 10px;
            width: 0%;
            transition: width 0.1s;
        }
        .timeline-marker {
            position: absolute;
            top: -5px;
            width: 4px;
            height: 70px;
            background: #ef4444;
        }
        .model-status {
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-ready {
            background: #d1fae5;
            color: #065f46;
        }
        .status-loading {
            background: #fef3c7;
            color: #92400e;
        }
        .status-error {
            background: #fee2e2;
            color: #991b1b;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-50 to-slate-100 p-8">
    <div class="max-w-6xl mx-auto">
        <div class="bg-white rounded-2xl shadow-xl p-8">
            <div class="flex items-center justify-between mb-6">
                <div class="flex items-center gap-3">
                    <i data-lucide="table" class="w-8 h-8 text-blue-600"></i>
                    <h1 class="text-3xl font-bold text-gray-800">YOLO Table Occupancy Detection</h1>
                </div>
                <div id="modelStatus" class="model-status status-loading">
                    <i data-lucide="cpu" class="w-4 h-4 inline mr-1"></i>
                    Loading YOLO Model...
                </div>
            </div>
            
            <p class="text-gray-600 mb-8">
                Real-time table occupancy detection using YOLO deep learning model. Upload images or videos for analysis.
            </p>

            <!-- Stats Dashboard -->
            <div id="statsDashboard" class="hidden grid grid-cols-3 gap-4 mb-8">
                <div class="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="table" class="w-5 h-5 text-blue-600"></i>
                        <span class="text-sm font-medium text-blue-900">Total Tables</span>
                    </div>
                    <p id="totalTables" class="text-3xl font-bold text-blue-600">0</p>
                </div>
                
                <div class="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl border border-red-200">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="users" class="w-5 h-5 text-red-600"></i>
                        <span class="text-sm font-medium text-red-900">Occupied</span>
                    </div>
                    <p id="occupiedTables" class="text-3xl font-bold text-red-600">0</p>
                    <p id="occupancyRate" class="text-sm text-red-700 mt-1">0% occupancy</p>
                </div>
                
                <div class="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                    <div class="flex items-center gap-2 mb-2">
                        <i data-lucide="check-circle" class="w-5 h-5 text-green-600"></i>
                        <span class="text-sm font-medium text-green-900">Vacant</span>
                    </div>
                    <p id="vacantTables" class="text-3xl font-bold text-green-600">0</p>
                    <p class="text-sm text-green-700 mt-1">Available for seating</p>
                </div>
            </div>

            <!-- Model Confidence Settings -->
            <div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-6">
                <div class="flex items-center gap-2 mb-2">
                    <i data-lucide="settings" class="w-4 h-4 text-yellow-600"></i>
                    <span class="text-sm font-medium text-yellow-900">Detection Settings</span>
                </div>
                <div class="flex items-center gap-4">
                    <div class="flex items-center gap-2">
                        <label class="text-sm text-yellow-800">Confidence Threshold:</label>
                        <input type="range" id="confidenceSlider" min="0.1" max="0.9" step="0.1" value="0.5" 
                               class="w-32" onchange="updateConfidenceValue(this.value)">
                        <span id="confidenceValue" class="text-sm font-mono text-yellow-800">0.5</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <label class="text-sm text-yellow-800">IOU Threshold:</label>
                        <input type="range" id="iouSlider" min="0.1" max="0.9" step="0.1" value="0.5" 
                               class="w-32" onchange="updateIouValue(this.value)">
                        <span id="iouValue" class="text-sm font-mono text-yellow-800">0.5</span>
                    </div>
                </div>
            </div>

            <!-- Upload Section -->
            <div class="mb-8">
                <div class="flex gap-4 flex-wrap">
                    <button
                        onclick="document.getElementById('imageInput').click()"
                        class="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                        <i data-lucide="upload" class="w-5 h-5"></i>
                        Upload Image
                    </button>
                    
                    <button
                        onclick="document.getElementById('videoInput').click()"
                        class="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                        <i data-lucide="video" class="w-5 h-5"></i>
                        Upload Video
                    </button>
                    
                    <button
                        onclick="loadDemoImage()"
                        class="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                    >
                        <i data-lucide="camera" class="w-5 h-5"></i>
                        Load Demo Image
                    </button>
                    
                    <button
                        id="detectButton"
                        onclick="processCurrentFrame()"
                        disabled
                        class="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                        <i data-lucide="zap" class="w-5 h-5"></i>
                        Run YOLO Detection
                    </button>
                </div>
                <input
                    id="imageInput"
                    type="file"
                    accept="image/*"
                    onchange="handleImageUpload(event)"
                    class="hidden"
                />
                <input
                    id="videoInput"
                    type="file"
                    accept="video/*"
                    onchange="handleVideoUpload(event)"
                    class="hidden"
                />
            </div>

            <!-- Detection Results -->
            <div id="resultsContainer" class="hidden bg-gray-50 rounded-xl p-6 border-2 border-gray-200 mb-4">
                <h3 class="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <i data-lucide="bar-chart" class="w-5 h-5"></i>
                    Detection Results
                </h3>
                <div id="detectionDetails" class="text-sm text-gray-700">
                    <!-- Results will be populated here -->
                </div>
            </div>

            <!-- Canvas Display -->
            <div id="canvasContainer" class="hidden bg-gray-50 rounded-xl p-6 border-2 border-gray-200 mb-4">
                <canvas
                    id="detectionCanvas"
                    class="max-w-full h-auto mx-auto rounded-lg shadow-lg"
                ></canvas>
            </div>

            <!-- Video Display -->
            <div id="videoContainer" class="hidden bg-gray-50 rounded-xl p-6 border-2 border-gray-200 mb-4">
                <div class="video-container">
                    <video
                        id="videoPlayer"
                        class="max-w-full h-auto mx-auto rounded-lg shadow-lg"
                        controls
                        ontimeupdate="updateVideoProgress()"
                    >
                        Your browser does not support the video tag.
                    </video>
                    <canvas
                        id="videoOverlay"
                        class="video-overlay max-w-full h-auto mx-auto rounded-lg"
                    ></canvas>
                </div>
            </div>

            <!-- Legend -->
            <div id="legend" class="hidden mt-8 flex items-center gap-6 justify-center flex-wrap">
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-red-500 rounded"></div>
                    <span class="text-sm text-gray-700">Occupied Table</span>
                </div>
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-green-500 rounded"></div>
                    <span class="text-sm text-gray-700">Vacant Table</span>
                </div>
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-blue-500 rounded"></div>
                    <span class="text-sm text-gray-700">Person</span>
                </div>
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-yellow-500 rounded"></div>
                    <span class="text-sm text-gray-700">Chair</span>
                </div>
            </div>

            <!-- Model Information -->
            <div class="mt-8 p-6 bg-gray-50 rounded-xl">
                <h3 class="font-semibold text-gray-900 mb-3">YOLO Model Information</h3>
                <div class="grid grid-cols-2 gap-4 text-sm text-gray-700">
                    <div>
                        <p><strong>Model:</strong> YOLOv8</p>
                        <p><strong>Framework:</strong> Ultralytics</p>
                        <p><strong>Task:</strong> Object Detection</p>
                    </div>
                    <div>
                        <p><strong>Classes:</strong> Table, Person, Chair</p>
                        <p><strong>Input Size:</strong> 640x640</p>
                        <p><strong>Backend:</strong> PyTorch</p>
                    </div>
                </div>
                <div class="mt-4 text-sm text-gray-600">
                    <p>The YOLO model detects tables and determines occupancy based on the presence of people and chairs around tables.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentImage = null;
        let currentVideo = null;
        let currentPredictions = [];
        let isProcessingVideo = false;
        let confidenceThreshold = 0.5;
        let iouThreshold = 0.5;

        // Initialize Lucide icons
        lucide.createIcons();

        // Check model status on load
        checkModelStatus();

        async function checkModelStatus() {
            try {
                const response = await fetch('/api/model-status');
                const data = await response.json();
                
                const statusElement = document.getElementById('modelStatus');
                if (data.ready) {
                    statusElement.className = 'model-status status-ready';
                    statusElement.innerHTML = '<i data-lucide="check-circle" class="w-4 h-4 inline mr-1"></i> YOLO Model Ready';
                } else {
                    statusElement.className = 'model-status status-error';
                    statusElement.innerHTML = '<i data-lucide="alert-triangle" class="w-4 h-4 inline mr-1"></i> ' + data.message;
                }
                lucide.createIcons();
            } catch (error) {
                console.error('Error checking model status:', error);
            }
        }

        function updateConfidenceValue(value) {
            confidenceThreshold = parseFloat(value);
            document.getElementById('confidenceValue').textContent = value;
        }

        function updateIouValue(value) {
            iouThreshold = parseFloat(value);
            document.getElementById('iouValue').textContent = value;
        }

        async function handleImageUpload(event) {
            const file = event.target.files[0];
            if (file) {
                // Reset video if any
                resetVideo();
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    currentImage = e.target.result;
                    currentPredictions = [];
                    resetStats();
                    showCanvas();
                    document.getElementById('detectButton').disabled = false;
                    document.getElementById('instructions').classList.add('hidden');
                };
                reader.readAsDataURL(file);
            }
        }

        async function handleVideoUpload(event) {
            const file = event.target.files[0];
            if (file) {
                // Reset image if any
                resetImage();
                
                const url = URL.createObjectURL(file);
                currentVideo = url;
                
                const video = document.getElementById('videoPlayer');
                video.src = url;
                
                video.onloadedmetadata = function() {
                    showVideo();
                    document.getElementById('detectButton').disabled = false;
                    document.getElementById('instructions').classList.add('hidden');
                    updateVideoInfo();
                };
            }
        }

        function resetImage() {
            currentImage = null;
            document.getElementById('canvasContainer').classList.add('hidden');
            document.getElementById('resultsContainer').classList.add('hidden');
        }

        function resetVideo() {
            if (currentVideo) {
                URL.revokeObjectURL(currentVideo);
                currentVideo = null;
            }
            stopVideoAnalysis();
            document.getElementById('videoContainer').classList.add('hidden');
            document.getElementById('videoDashboard').classList.add('hidden');
            document.getElementById('resultsContainer').classList.add('hidden');
            const video = document.getElementById('videoPlayer');
            video.src = '';
        }

        async function loadDemoImage() {
            try {
                const response = await fetch('/api/demo-image');
                const data = await response.json();
                
                if (data.success) {
                    // Reset video if any
                    resetVideo();
                    
                    currentImage = data.image;
                    currentPredictions = [];
                    resetStats();
                    showCanvas();
                    document.getElementById('detectButton').disabled = false;
                }
            } catch (error) {
                console.error('Error loading demo image:', error);
                alert('Error loading demo image. Please make sure the backend server is running.');
            }
        }

        async function processCurrentFrame() {
            if (currentVideo) {
                // Process current video frame
                const video = document.getElementById('videoPlayer');
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const imageData = canvas.toDataURL('image/jpeg');
                await processImageData(imageData);
                
                // Draw predictions on video overlay
                drawVideoPredictions();
            } else if (currentImage) {
                // Process current image
                await processImageData(currentImage);
                drawPredictions();
            }
        }

        async function processImageData(imageData) {
            const button = document.getElementById('detectButton');
            button.disabled = true;
            button.innerHTML = '<div class="loading-spinner"></div> Processing...';
            
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        image: imageData,
                        confidence: confidenceThreshold,
                        iou: iouThreshold
                    }),
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentPredictions = data.predictions;
                    updateStats(data.stats);
                    updateDetectionDetails(data.detection_info);
                    document.getElementById('legend').classList.remove('hidden');
                    document.getElementById('resultsContainer').classList.remove('hidden');
                } else {
                    alert('Error processing image: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image. Please make sure the backend server is running.');
            } finally {
                button.disabled = false;
                button.innerHTML = '<i data-lucide="zap" class="w-5 h-5"></i> Run YOLO Detection';
                lucide.createIcons();
            }
        }

        function updateDetectionDetails(detectionInfo) {
            const detailsElement = document.getElementById('detectionDetails');
            let html = `
                <div class="grid grid-cols-3 gap-4 mb-4">
                    <div><strong>Inference Time:</strong> ${detectionInfo.inference_time}ms</div>
                    <div><strong>Total Detections:</strong> ${detectionInfo.total_detections}</div>
                    <div><strong>Model:</strong> ${detectionInfo.model_name}</div>
                </div>
                <div class="mb-3">
                    <strong>Class Distribution:</strong>
                    <div class="flex gap-2 mt-1 flex-wrap">
            `;
            
            for (const [className, count] of Object.entries(detectionInfo.class_distribution)) {
                const color = getClassColor(className);
                html += `<span class="px-2 py-1 rounded text-xs text-white" style="background: ${color}">${className}: ${count}</span>`;
            }
            
            html += `</div></div>`;
            
            if (detectionInfo.advanced_metrics) {
                html += `<div><strong>Advanced Metrics:</strong> ${JSON.stringify(detectionInfo.advanced_metrics)}</div>`;
            }
            
            detailsElement.innerHTML = html;
        }

        function getClassColor(className) {
            const colors = {
                'table': '#3b82f6',
                'occupied_table': '#ef4444',
                'vacant_table': '#22c55e',
                'person': '#8b5cf6',
                'chair': '#f59e0b'
            };
            return colors[className] || '#6b7280';
        }

        function showCanvas() {
            document.getElementById('canvasContainer').classList.remove('hidden');
            const canvas = document.getElementById('detectionCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };
            
            img.src = currentImage;
        }

        function showVideo() {
            document.getElementById('videoContainer').classList.remove('hidden');
        }

        function drawPredictions() {
            const canvas = document.getElementById('detectionCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                currentPredictions.forEach(pred => {
                    const color = getClassColor(pred.class);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(pred.x, pred.y, pred.width, pred.height);
                    
                    // Draw label background
                    const label = `${pred.class} ${(pred.confidence * 100).toFixed(0)}%`;
                    ctx.font = 'bold 14px Arial';
                    const textWidth = ctx.measureText(label).width;
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(pred.x, pred.y - 25, textWidth + 10, 25);
                    
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(label, pred.x + 5, pred.y - 7);
                });
            };
            
            img.src = currentImage;
        }

        function drawVideoPredictions() {
            const video = document.getElementById('videoPlayer');
            const canvas = document.getElementById('videoOverlay');
            const ctx = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            currentPredictions.forEach(pred => {
                const color = getClassColor(pred.class);
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(pred.x, pred.y, pred.width, pred.height);
                
                // Draw label background
                const label = `${pred.class} ${(pred.confidence * 100).toFixed(0)}%`;
                ctx.font = 'bold 14px Arial';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = color;
                ctx.fillRect(pred.x, pred.y - 25, textWidth + 10, 25);
                
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, pred.x + 5, pred.y - 7);
            });
        }

        function updateStats(stats) {
            document.getElementById('statsDashboard').classList.remove('hidden');
            document.getElementById('totalTables').textContent = stats.total_tables || 0;
            document.getElementById('occupiedTables').textContent = stats.occupied_tables || 0;
            document.getElementById('vacantTables').textContent = stats.vacant_tables || 0;
            
            const totalTables = stats.total_tables || 1;
            const occupancyRate = ((stats.occupied_tables || 0) / totalTables) * 100;
            document.getElementById('occupancyRate').textContent = `${occupancyRate.toFixed(0)}% occupancy`;
        }

        function resetStats() {
            document.getElementById('statsDashboard').classList.add('hidden');
            document.getElementById('legend').classList.add('hidden');
        }

        // Initialize Lucide icons
        lucide.createIcons();
    </script>
</body>
</html>
'''

app = Flask(__name__)

class YOLOTableDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load YOLO model - will use pretrained model for demo"""
        try:
            if YOLO_AVAILABLE:
                # Load a pretrained YOLOv8 model
                # You can replace this with your custom trained model
                self.model = YOLO('yolov8n.pt')  # Using nano version for speed
                self.model_loaded = True
                print("YOLO model loaded successfully!")
            else:
                print("YOLO not available. Using mock mode.")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model_loaded = False
    
    def process_image(self, image_data, confidence=0.5, iou=0.5):
        """Process image with YOLO model"""
        try:
            start_time = time.time()
            
            # Convert base64 to image
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            if self.model_loaded and YOLO_AVAILABLE:
                # Run YOLO inference
                results = self.model(image_np, conf=confidence, iou=iou, verbose=False)
                
                # Process results
                predictions = []
                class_distribution = {}
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]
                            
                            # Update class distribution
                            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                            
                            prediction = {
                                "x": int(x1),
                                "y": int(y1),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1),
                                "confidence": float(conf),
                                "class": class_name,
                                "class_id": cls
                            }
                            predictions.append(prediction)
                
                # Calculate table occupancy based on detected objects
                stats = self.calculate_occupancy_stats(predictions)
                
                inference_time = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "predictions": predictions,
                    "stats": stats,
                    "detection_info": {
                        "inference_time": inference_time,
                        "total_detections": len(predictions),
                        "class_distribution": class_distribution,
                        "model_name": "YOLOv8n" if self.model_loaded else "Mock Model",
                        "advanced_metrics": {
                            "confidence_threshold": confidence,
                            "iou_threshold": iou
                        }
                    }
                }
            else:
                # Fallback to mock mode
                return self.mock_detection(image_np, confidence)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def calculate_occupancy_stats(self, predictions):
        """Calculate table occupancy statistics from YOLO detections"""
        tables = [p for p in predictions if p['class'] in ['dining table', 'table']]
        people = [p for p in predictions if p['class'] in ['person']]
        chairs = [p for p in predictions if p['class'] in ['chair']]
        
        # Simple occupancy logic: table is occupied if people are detected near it
        occupied_tables = 0
        vacant_tables = 0
        
        for table in tables:
            table_center_x = table['x'] + table['width'] / 2
            table_center_y = table['y'] + table['height'] / 2
            
            # Check if any person is near this table
            is_occupied = False
            for person in people:
                person_center_x = person['x'] + person['width'] / 2
                person_center_y = person['y'] + person['height'] / 2
                
                # Calculate distance between table and person
                distance = ((table_center_x - person_center_x) ** 2 + 
                           (table_center_y - person_center_y) ** 2) ** 0.5
                
                # If person is close to table, consider it occupied
                if distance < max(table['width'], table['height']) * 1.5:
                    is_occupied = True
                    break
            
            if is_occupied:
                occupied_tables += 1
                # Update class to indicate occupancy
                table['class'] = 'occupied_table'
            else:
                vacant_tables += 1
                table['class'] = 'vacant_table'
        
        return {
            "total_tables": len(tables),
            "occupied_tables": occupied_tables,
            "vacant_tables": vacant_tables,
            "total_people": len(people),
            "total_chairs": len(chairs)
        }
    
    def mock_detection(self, image_np, confidence):
        """Mock detection for when YOLO is not available"""
        height, width = image_np.shape[:2]
        
        # Generate mock predictions based on image size
        predictions = []
        class_distribution = {}
        
        # Create some mock tables
        table_positions = [
            (width//4, height//4), (width//2, height//4), (3*width//4, height//4),
            (width//4, height//2), (width//2, height//2), (3*width//4, height//2),
        ]
        
        for i, (x, y) in enumerate(table_positions):
            table_w, table_h = 150, 100
            table_pred = {
                "x": x - table_w//2,
                "y": y - table_h//2,
                "width": table_w,
                "height": table_h,
                "confidence": max(confidence, 0.7 + i*0.05),
                "class": "occupied_table" if i % 2 == 0 else "vacant_table",
                "class_id": 0
            }
            predictions.append(table_pred)
            class_distribution[table_pred["class"]] = class_distribution.get(table_pred["class"], 0) + 1
        
        # Add some mock people near occupied tables
        for i, pred in enumerate(predictions):
            if pred["class"] == "occupied_table":
                person_pred = {
                    "x": pred["x"] + 20,
                    "y": pred["y"] - 30,
                    "width": 40,
                    "height": 80,
                    "confidence": 0.8,
                    "class": "person",
                    "class_id": 1
                }
                predictions.append(person_pred)
                class_distribution["person"] = class_distribution.get("person", 0) + 1
        
        stats = {
            "total_tables": len([p for p in predictions if 'table' in p['class']]),
            "occupied_tables": len([p for p in predictions if p['class'] == 'occupied_table']),
            "vacant_tables": len([p for p in predictions if p['class'] == 'vacant_table']),
            "total_people": len([p for p in predictions if p['class'] == 'person']),
            "total_chairs": 0
        }
        
        return {
            "success": True,
            "predictions": predictions,
            "stats": stats,
            "detection_info": {
                "inference_time": 50,
                "total_detections": len(predictions),
                "class_distribution": class_distribution,
                "model_name": "Mock Model (YOLO not available)",
                "advanced_metrics": {
                    "confidence_threshold": confidence,
                    "note": "Install ultralytics for real YOLO detection"
                }
            }
        }
    
    def generate_demo_image(self):
        """Generate a demo restaurant image with tables and people"""
        width, height = 800, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw floor pattern
        for i in range(0, width, 50):
            cv2.line(image, (i, 0), (i, height), (220, 220, 220), 1)
        for i in range(0, height, 50):
            cv2.line(image, (0, i), (width, i), (220, 220, 220), 1)
        
        # Draw tables
        tables = [
            (200, 150, True), (400, 150, False), (600, 150, True),
            (200, 350, False), (400, 350, True), (600, 350, False),
        ]
        
        for i, (x, y, occupied) in enumerate(tables):
            # Table
            cv2.rectangle(image, (x-75, y-50), (x+75, y+50), (139, 69, 19), -1)
            cv2.rectangle(image, (x-65, y-40), (x+65, y+40), (160, 82, 45), -1)
            
            if occupied:
                # Draw people
                cv2.ellipse(image, (x-30, y-70), (15, 15), 0, 0, 360, (74, 107, 227), -1)
                cv2.ellipse(image, (x+30, y-70), (15, 15), 0, 0, 360, (74, 107, 227), -1)
                # Draw plates
                cv2.ellipse(image, (x-20, y), (10, 10), 0, 0, 360, (255, 255, 255), -1)
                cv2.ellipse(image, (x+20, y), (10, 10), 0, 0, 360, (255, 255, 255), -1)
            
            # Table number
            cv2.putText(image, f"T{i+1}", (x-10, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64

# Initialize detector
detector = YOLOTableDetector()

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect_tables():
    """API endpoint for table occupancy detection with YOLO"""
    data = request.json
    image_data = data.get('image')
    confidence = data.get('confidence', 0.5)
    iou = data.get('iou', 0.5)
    
    if not image_data:
        return jsonify({"success": False, "error": "No image data provided"})
    
    result = detector.process_image(image_data, confidence, iou)
    return jsonify(result)

@app.route('/api/demo-image', methods=['GET'])
def get_demo_image():
    """API endpoint to generate demo image"""
    try:
        image_base64 = detector.generate_demo_image()
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_base64}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """API endpoint to check YOLO model status"""
    return jsonify({
        "ready": detector.model_loaded,
        "message": "YOLO model loaded successfully" if detector.model_loaded else "YOLO not available. Using mock mode."
    })

def run_flask():
    """Run Flask server"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Create webview window
    window = webview.create_window(
        'YOLO Table Occupancy Detection System',
        'http://127.0.0.1:5000',
        width=1400,
        height=1000,
        min_size=(1000, 800)
    )
    
    # Start the application
    webview.start(debug=True)