import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, AlertCircle, CheckCircle, Users, Table } from 'lucide-react';

const TableOccupancyDetector = () => {
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [stats, setStats] = useState({ occupied: 0, vacant: 0, total: 0 });
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Simulated ML model - In production, this would call your trained model
  const detectTableOccupancy = (imageData) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Simulate detection results for a restaurant floor plan
        const mockPredictions = [
          { id: 1, x: 100, y: 80, width: 120, height: 100, occupied: true, confidence: 0.95, tableNumber: 'T1' },
          { id: 2, x: 280, y: 80, width: 120, height: 100, occupied: false, confidence: 0.92, tableNumber: 'T2' },
          { id: 3, x: 460, y: 80, width: 120, height: 100, occupied: true, confidence: 0.88, tableNumber: 'T3' },
          { id: 4, x: 100, y: 240, width: 120, height: 100, occupied: false, confidence: 0.91, tableNumber: 'T4' },
          { id: 5, x: 280, y: 240, width: 120, height: 100, occupied: true, confidence: 0.94, tableNumber: 'T5' },
          { id: 6, x: 460, y: 240, width: 120, height: 100, occupied: false, confidence: 0.89, tableNumber: 'T6' },
        ];
        resolve(mockPredictions);
      }, 1500);
    });
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setPredictions([]);
        setStats({ occupied: 0, vacant: 0, total: 0 });
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!image) return;
    
    setIsProcessing(true);
    const results = await detectTableOccupancy(image);
    setPredictions(results);
    
    const occupied = results.filter(p => p.occupied).length;
    const vacant = results.filter(p => !p.occupied).length;
    setStats({ occupied, vacant, total: results.length });
    
    setIsProcessing(false);
  };

  useEffect(() => {
    if (image && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw predictions
        predictions.forEach(pred => {
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
      
      img.src = image;
    }
  }, [image, predictions]);

  const generateDemoImage = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 700;
    canvas.height = 450;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw tables
    const tables = [
      { x: 100, y: 80, occupied: true },
      { x: 280, y: 80, occupied: false },
      { x: 460, y: 80, occupied: true },
      { x: 100, y: 240, occupied: false },
      { x: 280, y: 240, occupied: true },
      { x: 460, y: 240, occupied: false },
    ];
    
    tables.forEach((table, i) => {
      // Table
      ctx.fillStyle = '#8b4513';
      ctx.fillRect(table.x, table.y, 120, 100);
      
      // Table top
      ctx.fillStyle = '#a0522d';
      ctx.fillRect(table.x + 10, table.y + 10, 100, 80);
      
      if (table.occupied) {
        // Chairs (represented as circles)
        ctx.fillStyle = '#4a5568';
        ctx.beginPath();
        ctx.arc(table.x + 60, table.y - 10, 15, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(table.x + 60, table.y + 110, 15, 0, Math.PI * 2);
        ctx.fill();
        
        // Items on table
        ctx.fillStyle = '#e2e8f0';
        ctx.fillRect(table.x + 30, table.y + 40, 20, 25);
        ctx.fillRect(table.x + 70, table.y + 40, 20, 25);
      }
      
      // Table number
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 16px Arial';
      ctx.fillText(`T${i + 1}`, table.x + 50, table.y + 60);
    });
    
    setImage(canvas.toDataURL());
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <Table className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-800">Table Occupancy Detection System</h1>
          </div>
          
          <p className="text-gray-600 mb-8">
            Upload a restaurant floor image or use the demo to detect occupied and vacant tables in real-time.
          </p>

          {/* Stats Dashboard */}
          {stats.total > 0 && (
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
                <div className="flex items-center gap-2 mb-2">
                  <Table className="w-5 h-5 text-blue-600" />
                  <span className="text-sm font-medium text-blue-900">Total Tables</span>
                </div>
                <p className="text-3xl font-bold text-blue-600">{stats.total}</p>
              </div>
              
              <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl border border-red-200">
                <div className="flex items-center gap-2 mb-2">
                  <Users className="w-5 h-5 text-red-600" />
                  <span className="text-sm font-medium text-red-900">Occupied</span>
                </div>
                <p className="text-3xl font-bold text-red-600">{stats.occupied}</p>
                <p className="text-sm text-red-700 mt-1">
                  {((stats.occupied / stats.total) * 100).toFixed(0)}% occupancy
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-medium text-green-900">Vacant</span>
                </div>
                <p className="text-3xl font-bold text-green-600">{stats.vacant}</p>
                <p className="text-sm text-green-700 mt-1">Available for seating</p>
              </div>
            </div>
          )}

          {/* Upload Section */}
          <div className="mb-8">
            <div className="flex gap-4">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Upload className="w-5 h-5" />
                Upload Image
              </button>
              
              <button
                onClick={generateDemoImage}
                className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <Camera className="w-5 h-5" />
                Load Demo
              </button>
              
              {image && (
                <button
                  onClick={processImage}
                  disabled={isProcessing}
                  className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5" />
                      Detect Tables
                    </>
                  )}
                </button>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

          {/* Canvas Display */}
          {image && (
            <div className="bg-gray-50 rounded-xl p-6 border-2 border-gray-200">
              <canvas
                ref={canvasRef}
                className="max-w-full h-auto mx-auto rounded-lg shadow-lg"
              />
            </div>
          )}

          {/* Legend */}
          {predictions.length > 0 && (
            <div className="mt-8 flex items-center gap-6 justify-center">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 rounded"></div>
                <span className="text-sm text-gray-700">Occupied Table</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded"></div>
                <span className="text-sm text-gray-700">Vacant Table</span>
              </div>
            </div>
          )}

          {/* Instructions */}
          {!image && (
            <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
              <h3 className="font-semibold text-blue-900 mb-3">How to use:</h3>
              <ul className="space-y-2 text-blue-800 text-sm">
                <li>1. Click "Upload Image" to select a restaurant floor photo</li>
                <li>2. Or click "Load Demo" to see a sample detection</li>
                <li>3. Click "Detect Tables" to run the occupancy detection</li>
                <li>4. View real-time statistics and visual bounding boxes</li>
              </ul>
            </div>
          )}

          {/* Model Info */}
          <div className="mt-8 p-6 bg-gray-50 rounded-xl">
            <h3 className="font-semibold text-gray-900 mb-3">Model Architecture</h3>
            <p className="text-sm text-gray-700 leading-relaxed">
              This system uses a computer vision model trained to detect tables and classify their occupancy status. 
              In production, you would integrate a trained YOLO, Faster R-CNN, or similar object detection model fine-tuned 
              on restaurant imagery. The model detects tables and uses features like chair positions, plate/glass presence, 
              and people detection to determine occupancy status.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TableOccupancyDetector;