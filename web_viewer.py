"""Web viewer for FreeTimeGS 4D Gaussian Splatting"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from pathlib import Path
import uvicorn

app = FastAPI(title="FreeTimeGS 4D Gaussian Splatting Viewer")

# Setup directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/api/gaussians/{model_name}")
async def get_gaussians(model_name: str):
    gaussians_path = Path("trained_model") / "gaussians.json"
    if not gaussians_path.exists():
        return {"error": "Model not found"}
    
    with open(gaussians_path, 'r') as f:
        return json.load(f)

def create_templates():
    """Create HTML and JS templates"""
    
    # HTML template
    html = '''<!DOCTYPE html>
<html><head><title>FreeTimeGS 4D Viewer</title>
<style>
body { margin:0; background:#000; font-family:Arial; overflow:hidden; }
#container { position:relative; width:100vw; height:100vh; }
#canvas { display:block; width:100%; height:100%; }
#controls { position:absolute; top:20px; left:20px; background:rgba(0,0,0,0.8); 
           padding:15px; border-radius:8px; color:white; z-index:100; }
.control-group { margin-bottom:10px; }
label { display:inline-block; width:80px; font-size:14px; }
input[type="range"] { width:120px; }
#info { position:absolute; bottom:20px; left:20px; background:rgba(0,0,0,0.8);
        padding:10px; border-radius:8px; color:white; font-size:12px; }
.loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
          color:white; font-size:18px; }
</style></head>
<body>
<div id="container">
<canvas id="canvas"></canvas>
<div id="controls">
<h3>FreeTimeGS 4D Viewer</h3>
<div class="control-group">
<label>Time:</label>
<input type="range" id="timeSlider" min="0" max="1" step="0.01" value="0.5">
<span id="timeValue">0.50</span>
</div>
<div class="control-group">
<button id="playButton">Play</button>
<button id="pauseButton">Pause</button>
<button id="resetButton">Reset</button>
</div>
</div>
<div id="info">
<div>Mouse: rotate, scroll: zoom</div>
<div>Gaussians: <span id="gaussianCount">0</span></div>
</div>
<div id="loading" class="loading">Loading model...</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="/static/viewer.js"></script>
</body></html>'''
    
    with open("templates/viewer.html", "w") as f:
        f.write(html)
    
    # JavaScript viewer
    js = '''class GaussianViewer {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({canvas: document.getElementById('canvas')});
        this.currentTime = 0.5;
        this.isPlaying = false;
        this.setupRenderer();
        this.setupControls();
        this.loadData();
        this.animate();
    }
    
    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000);
        this.camera.position.set(0, 0, 3);
    }
    
    setupControls() {
        const timeSlider = document.getElementById('timeSlider');
        const timeValue = document.getElementById('timeValue');
        
        timeSlider.addEventListener('input', (e) => {
            this.currentTime = parseFloat(e.target.value);
            timeValue.textContent = this.currentTime.toFixed(2);
            this.updateGaussians();
        });
        
        document.getElementById('playButton').addEventListener('click', () => this.isPlaying = true);
        document.getElementById('pauseButton').addEventListener('click', () => this.isPlaying = false);
        document.getElementById('resetButton').addEventListener('click', () => this.resetView());
        
        let mouseDown = false, mouseX = 0, mouseY = 0;
        document.addEventListener('mousedown', (e) => { mouseDown = true; mouseX = e.clientX; mouseY = e.clientY; });
        document.addEventListener('mouseup', () => mouseDown = false);
        document.addEventListener('mousemove', (e) => {
            if (!mouseDown) return;
            this.camera.position.x += (e.clientX - mouseX) * 0.01;
            this.camera.position.y -= (e.clientY - mouseY) * 0.01;
            mouseX = e.clientX; mouseY = e.clientY;
        });
        document.addEventListener('wheel', (e) => {
            this.camera.position.z += e.deltaY * 0.01;
            this.camera.position.z = Math.max(0.1, Math.min(10, this.camera.position.z));
        });
    }
    
    async loadData() {
        try {
            const response = await fetch('/api/gaussians/default');
            this.gaussianData = await response.json();
            if (this.gaussianData.error) {
                document.getElementById('loading').textContent = 'No model found';
                return;
            }
            this.createMesh();
            document.getElementById('loading').style.display = 'none';
            document.getElementById('gaussianCount').textContent = this.gaussianData.num_gaussians;
        } catch (error) {
            document.getElementById('loading').textContent = 'Load failed';
        }
    }
    
    createMesh() {
        const positions = new Float32Array(this.gaussianData.positions.flat());
        const colors = new Float32Array(this.gaussianData.colors.flat());
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 0.05, vertexColors: true, transparent: true, opacity: 0.8
        });
        
        this.gaussianMesh = new THREE.Points(geometry, material);
        this.scene.add(this.gaussianMesh);
    }
    
    updateGaussians() {
        if (!this.gaussianMesh || !this.gaussianData) return;
        const positions = this.gaussianMesh.geometry.attributes.position;
        
        for (let i = 0; i < this.gaussianData.num_gaussians; i++) {
            const basePos = this.gaussianData.positions[i];
            const velocity = this.gaussianData.velocities[i];
            positions.array[i*3] = basePos[0] + velocity[0] * this.currentTime;
            positions.array[i*3+1] = basePos[1] + velocity[1] * this.currentTime;
            positions.array[i*3+2] = basePos[2] + velocity[2] * this.currentTime;
        }
        positions.needsUpdate = true;
    }
    
    resetView() {
        this.camera.position.set(0, 0, 3);
        this.currentTime = 0.5;
        document.getElementById('timeSlider').value = 0.5;
        document.getElementById('timeValue').textContent = '0.50';
        this.updateGaussians();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.isPlaying) {
            this.currentTime += 0.01;
            if (this.currentTime > 1) this.currentTime = 0;
            document.getElementById('timeSlider').value = this.currentTime;
            document.getElementById('timeValue').textContent = this.currentTime.toFixed(2);
            this.updateGaussians();
        }
        this.renderer.render(this.scene, this.camera);
    }
}

document.addEventListener('DOMContentLoaded', () => new GaussianViewer());'''
    
    with open("static/viewer.js", "w") as f:
        f.write(js)

if __name__ == "__main__":
    create_templates()
    print("Starting viewer at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
#!/usr/bin/env python3
 