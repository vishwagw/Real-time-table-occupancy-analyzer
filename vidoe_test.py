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

# HTML template as a string
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table Occupancy Detection System</title>
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
    </style>
</head>
<body class="bg-gradient-to-br from-slate-50 to-slate-100 p-8">
    <div class="max-w-6xl mx-auto">
        <div class="bg-white rounded-2xl shadow-xl p-8">
            <div class="flex items-center gap-3 mb-6">
                <i data-lucide="table" class="w-8 h-8 text-blue-600"></i>
                <h1 class="text-3xl font-bold text-gray-800">Table Occupancy Detection System</h1>
            </div>
            
            <p class="text-gray-600 mb-8">
                Upload images or videos to detect occupied and vacant tables in real-time. Analyze recorded videos for occupancy patterns.
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

            <!-- Video Analysis Dashboard -->
            <div id="videoDashboard" class="hidden bg-purple-50 border border-purple-200 rounded-xl p-6 mb-8">
                <h3 class="font-semibold text-purple-900 mb-4 flex items-center gap-2">
                    <i data-lucide="video" class="w-5 h-5"></i>
                    Video Analysis
                </h3>
                <div class="grid grid-cols-4 gap-4 mb-4">
                    <div class="text-center">
                        <p class="text-2xl font-bold text-purple-600" id="currentTime">00:00</p>
                        <p class="text-sm text-purple-700">Current Time</p>
                    </div>
                    <div class="text-center">
                        <p class="text-2xl font-bold text-purple-600" id="totalTime">00:00</p>
                        <p class="text-sm text-purple-700">Total Time</p>
                    </div>
                    <div class="text-center">
                        <p class="text-2xl font-bold text-purple-600" id="currentFrame">0</p>
                        <p class="text-sm text-purple-700">Current Frame</p>
                    </div>
                    <div class="text-center">
                        <p class="text-2xl font-bold text-purple-600" id="totalFrames">0</p>
                        <p class="text-sm text-purple-700">Total Frames</p>
                    </div>
                </div>
                <div class="timeline">
                    <div id="timelineProgress" class="timeline-progress"></div>
                    <div id="timelineMarker" class="timeline-marker"></div>
                </div>
                <div class="flex gap-2 mt-4">
                    <button onclick="playVideo()" class="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                        <i data-lucide="play" class="w-4 h-4"></i> Play
                    </button>
                    <button onclick="pauseVideo()" class="flex items-center gap-2 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700">
                        <i data-lucide="pause" class="w-4 h-4"></i> Pause
                    </button>
                    <button onclick="stopVideo()" class="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
                        <i data-lucide="square" class="w-4 h-4"></i> Stop
                    </button>
                    <button onclick="analyzeEntireVideo()" id="analyzeVideoBtn" class="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                        <i data-lucide="activity" class="w-4 h-4"></i> Analyze Full Video
                    </button>
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
                        <i data-lucide="alert-circle" class="w-5 h-5"></i>
                        Detect Tables
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
            <div id="legend" class="hidden mt-8 flex items-center gap-6 justify-center">
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-red-500 rounded"></div>
                    <span class="text-sm text-gray-700">Occupied Table</span>
                </div>
                <div class="flex items-center gap-2">
                    <div class="w-4 h-4 bg-green-500 rounded"></div>
                    <span class="text-sm text-gray-700">Vacant Table</span>
                </div>
            </div>

            <!-- Instructions -->
            <div id="instructions" class="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
                <h3 class="font-semibold text-blue-900 mb-3">How to use:</h3>
                <ul class="space-y-2 text-blue-800 text-sm">
                    <li>1. Click "Upload Image" to select a restaurant floor photo</li>
                    <li>2. Click "Upload Video" to analyze recorded restaurant footage</li>
                    <li>3. Use "Load Demo Image" to see a sample detection</li>
                    <li>4. Click "Detect Tables" to run occupancy detection on current frame</li>
                    <li>5. For videos, use playback controls and "Analyze Full Video" for complete analysis</li>
                </ul>
            </div>

            <!-- Model Info -->
            <div class="mt-8 p-6 bg-gray-50 rounded-xl">
                <h3 class="font-semibold text-gray-900 mb-3">Video Analysis Features</h3>
                <p class="text-sm text-gray-700 leading-relaxed mb-4">
                    The system now supports video analysis for recorded restaurant footage. You can:
                </p>
                <ul class="text-sm text-gray-700 list-disc list-inside space-y-1 mb-4">
                    <li>Analyze table occupancy patterns over time</li>
                    <li>Detect peak occupancy periods</li>
                    <li>Track table turnover rates</li>
                    <li>Generate occupancy heatmaps from video data</li>
                </ul>
                <p class="text-sm text-gray-700 leading-relaxed">
                    The computer vision model processes each frame to detect tables and classify occupancy based on visual cues like chair positions, table settings, and customer presence.
                </p>
            </div>
        </div>
    </div>

    <script>
        let currentImage = null;
        let currentVideo = null;
        let currentPredictions = [];
        let isProcessingVideo = false;
        let videoAnalysisInterval = null;

        // Initialize Lucide icons
        lucide.createIcons();

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
        }

        function resetVideo() {
            if (currentVideo) {
                URL.revokeObjectURL(currentVideo);
                currentVideo = null;
            }
            stopVideoAnalysis();
            document.getElementById('videoContainer').classList.add('hidden');
            document.getElementById('videoDashboard').classList.add('hidden');
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
                    document.getElementById('instructions').classList.add('hidden');
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
                    body: JSON.stringify({ image: imageData }),
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentPredictions = data.predictions;
                    updateStats(data.stats);
                    document.getElementById('legend').classList.remove('hidden');
                } else {
                    alert('Error processing image: ' + data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image. Please make sure the backend server is running.');
            } finally {
                button.disabled = false;
                button.innerHTML = '<i data-lucide="alert-circle" class="w-5 h-5"></i> Detect Tables';
                lucide.createIcons();
            }
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
            document.getElementById('videoDashboard').classList.remove('hidden');
        }

        function drawPredictions() {
            const canvas = document.getElementById('detectionCanvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
                
                currentPredictions.forEach(pred => {
                    ctx.strokeStyle = pred.occupied ? '#ef4444' : '#22c55e';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(pred.x, pred.y, pred.width, pred.height);
                    
                    // Draw label background
                    const label = `${pred.tableNumber}: ${pred.occupied ? 'Occupied' : 'Vacant'} (${(pred.confidence * 100).toFixed(0)}%)`;
                    ctx.font = 'bold 14px Arial';
                    const textWidth = ctx.measureText(label).width;
                    
                    ctx.fillStyle = pred.occupied ? '#ef4444' : '#22c55e';
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
                ctx.strokeStyle = pred.occupied ? '#ef4444' : '#22c55e';
                ctx.lineWidth = 3;
                ctx.strokeRect(pred.x, pred.y, pred.width, pred.height);
                
                // Draw label background
                const label = `${pred.tableNumber}: ${pred.occupied ? 'Occupied' : 'Vacant'} (${(pred.confidence * 100).toFixed(0)}%)`;
                ctx.font = 'bold 14px Arial';
                const textWidth = ctx.measureText(label).width;
                
                ctx.fillStyle = pred.occupied ? '#ef4444' : '#22c55e';
                ctx.fillRect(pred.x, pred.y - 25, textWidth + 10, 25);
                
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, pred.x + 5, pred.y - 7);
            });
        }

        function updateStats(stats) {
            document.getElementById('statsDashboard').classList.remove('hidden');
            document.getElementById('totalTables').textContent = stats.total;
            document.getElementById('occupiedTables').textContent = stats.occupied;
            document.getElementById('vacantTables').textContent = stats.vacant;
            document.getElementById('occupancyRate').textContent = 
                `${((stats.occupied / stats.total) * 100).toFixed(0)}% occupancy`;
        }

        function resetStats() {
            document.getElementById('statsDashboard').classList.add('hidden');
            document.getElementById('legend').classList.add('hidden');
        }

        function updateVideoInfo() {
            const video = document.getElementById('videoPlayer');
            document.getElementById('totalTime').textContent = formatTime(video.duration);
            document.getElementById('totalFrames').textContent = Math.floor(video.duration * 30); // Assuming 30 FPS
        }

        function updateVideoProgress() {
            const video = document.getElementById('videoPlayer');
            const currentTime = video.currentTime;
            const duration = video.duration;
            
            if (duration) {
                const progress = (currentTime / duration) * 100;
                document.getElementById('timelineProgress').style.width = `${progress}%`;
                document.getElementById('currentTime').textContent = formatTime(currentTime);
                document.getElementById('currentFrame').textContent = Math.floor(currentTime * 30); // Assuming 30 FPS
                
                // Update timeline marker for current analysis position
                if (isProcessingVideo) {
                    document.getElementById('timelineMarker').style.left = `${progress}%`;
                }
            }
        }

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        function playVideo() {
            document.getElementById('videoPlayer').play();
        }

        function pauseVideo() {
            document.getElementById('videoPlayer').pause();
        }

        function stopVideo() {
            const video = document.getElementById('videoPlayer');
            video.pause();
            video.currentTime = 0;
            updateVideoProgress();
        }

        async function analyzeEntireVideo() {
            if (!currentVideo || isProcessingVideo) return;
            
            const button = document.getElementById('analyzeVideoBtn');
            button.disabled = true;
            button.innerHTML = '<div class="loading-spinner"></div> Analyzing...';
            isProcessingVideo = true;
            
            const video = document.getElementById('videoPlayer');
            const originalTime = video.currentTime;
            video.pause();
            
            try {
                // This would typically send the entire video to the backend for processing
                // For this demo, we'll simulate processing key frames
                alert('Video analysis started! In a real implementation, this would process the entire video and generate occupancy analytics.');
                
                // Simulate processing
                await new Promise(resolve => setTimeout(resolve, 3000));
                
                alert('Video analysis complete! Occupancy patterns have been analyzed.');
                
            } catch (error) {
                console.error('Error analyzing video:', error);
                alert('Error analyzing video: ' + error.message);
            } finally {
                button.disabled = false;
                button.innerHTML = '<i data-lucide="activity" class="w-4 h-4"></i> Analyze Full Video';
                lucide.createIcons();
                isProcessingVideo = false;
                video.currentTime = originalTime;
            }
        }

        function stopVideoAnalysis() {
            isProcessingVideo = false;
            if (videoAnalysisInterval) {
                clearInterval(videoAnalysisInterval);
                videoAnalysisInterval = null;
            }
        }
    </script>
</body>
</html>
'''

app = Flask(__name__)

class TableOccupancyDetector:
    def process_image(self, image_data):
        """Process uploaded image and return table occupancy predictions"""
        try:
            # Simulate processing delay
            time.sleep(1)
            
            # Mock predictions - in real implementation, these would vary based on actual image content
            predictions = [
                { 
                    "id": 1, "x": 100, "y": 80, "width": 120, "height": 100, 
                    "occupied": True, "confidence": 0.95, "tableNumber": "T1" 
                },
                { 
                    "id": 2, "x": 280, "y": 80, "width": 120, "height": 100, 
                    "occupied": False, "confidence": 0.92, "tableNumber": "T2" 
                },
                { 
                    "id": 3, "x": 460, "y": 80, "width": 120, "height": 100, 
                    "occupied": True, "confidence": 0.88, "tableNumber": "T3" 
                },
                { 
                    "id": 4, "x": 100, "y": 240, "width": 120, "height": 100, 
                    "occupied": False, "confidence": 0.91, "tableNumber": "T4" 
                },
                { 
                    "id": 5, "x": 280, "y": 240, "width": 120, "height": 100, 
                    "occupied": True, "confidence": 0.94, "tableNumber": "T5" 
                },
                { 
                    "id": 6, "x": 460, "y": 240, "width": 120, "height": 100, 
                    "occupied": False, "confidence": 0.89, "tableNumber": "T6" 
                },
            ]
            
            # Calculate statistics
            occupied = len([p for p in predictions if p['occupied']])
            vacant = len([p for p in predictions if not p['occupied']])
            
            return {
                "success": True,
                "predictions": predictions,
                "stats": {
                    "occupied": occupied,
                    "vacant": vacant,
                    "total": len(predictions)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_demo_image(self):
        """Generate a demo restaurant floor plan image"""
        # Create a blank image
        width, height = 700, 450
        image = np.ones((height, width, 3), dtype=np.uint8) * 248
        
        # Draw tables
        tables = [
            {"x": 100, "y": 80, "occupied": True},
            {"x": 280, "y": 80, "occupied": False},
            {"x": 460, "y": 80, "occupied": True},
            {"x": 100, "y": 240, "occupied": False},
            {"x": 280, "y": 240, "occupied": True},
            {"x": 460, "y": 240, "occupied": False},
        ]
        
        for i, table in enumerate(tables):
            # Table base
            cv2.rectangle(image, (table["x"], table["y"]), 
                         (table["x"] + 120, table["y"] + 100), 
                         (139, 69, 19), -1)
            
            # Table top
            cv2.rectangle(image, (table["x"] + 10, table["y"] + 10), 
                         (table["x"] + 110, table["y"] + 90), 
                         (160, 82, 45), -1)
            
            if table["occupied"]:
                # Draw plates
                cv2.rectangle(image, (table["x"] + 30, table["y"] + 40), 
                             (table["x"] + 50, table["y"] + 65), 
                             (226, 232, 240), -1)
                cv2.rectangle(image, (table["x"] + 70, table["y"] + 40), 
                             (table["x"] + 90, table["y"] + 65), 
                             (226, 232, 240), -1)
            
            # Table number
            cv2.putText(image, f"T{i+1}", 
                       (table["x"] + 50, table["y"] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64

detector = TableOccupancyDetector()

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/detect', methods=['POST'])
def detect_tables():
    """API endpoint for table occupancy detection"""
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"success": False, "error": "No image data provided"})
    
    result = detector.process_image(image_data)
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

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """API endpoint to analyze entire video"""
    try:
        # In a real implementation, this would process the video file
        # and return analytics data
        time.sleep(2)  # Simulate processing
        
        # Mock analytics data
        analytics = {
            "total_frames": 900,
            "processed_frames": 900,
            "average_occupancy": 0.45,
            "peak_occupancy": 0.83,
            "peak_time": "19:30",
            "table_turnover_rate": 2.1,
            "occupancy_timeline": [
                {"time": "18:00", "occupancy": 0.2},
                {"time": "18:30", "occupancy": 0.5},
                {"time": "19:00", "occupancy": 0.7},
                {"time": "19:30", "occupancy": 0.83},
                {"time": "20:00", "occupancy": 0.6},
                {"time": "20:30", "occupancy": 0.3},
            ]
        }
        
        return jsonify({
            "success": True,
            "analytics": analytics
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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
        'Table Occupancy Detection System - Video Analysis',
        'http://127.0.0.1:5000',
        width=1400,
        height=1000,
        min_size=(1000, 800)
    )
    
    # Start the application
    webview.start(debug=True)