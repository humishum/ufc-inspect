class GaussianViewer {
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

document.addEventListener('DOMContentLoaded', () => new GaussianViewer());