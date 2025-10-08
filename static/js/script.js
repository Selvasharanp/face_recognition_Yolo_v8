class FaceRecognitionApp {
    constructor() {
        this.isCameraOn = false;
        this.unknownFaceDetected = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.startDetectionPolling();
    }

    bindEvents() {
        document.getElementById('startBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('saveFaceBtn').addEventListener('click', () => this.saveUnknownFace());
    }

    async startCamera() {
        try {
            const response = await fetch('/start_camera');
            const data = await response.json();
            
            if (data.status === 'camera started') {
                this.isCameraOn = true;
                document.getElementById('videoFeed').style.display = 'block';
                document.getElementById('placeholder').style.display = 'none';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                
                this.showNotification('Camera started successfully!', 'success');
            }
        } catch (error) {
            this.showNotification('Error starting camera: ' + error.message, 'error');
        }
    }

    async stopCamera() {
        try {
            const response = await fetch('/stop_camera');
            const data = await response.json();
            
            if (data.status === 'camera stopped') {
                this.isCameraOn = false;
                document.getElementById('videoFeed').style.display = 'none';
                document.getElementById('placeholder').style.display = 'flex';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                this.showNotification('Camera stopped', 'info');
            }
        } catch (error) {
            this.showNotification('Error stopping camera: ' + error.message, 'error');
        }
    }

    async startDetectionPolling() {
        setInterval(async () => {
            if (this.isCameraOn) {
                await this.updateDetections();
            }
        }, 2000); // Update every 2 seconds
    }

    async updateDetections() {
        try {
            const response = await fetch('/get_detections');
            const detections = await response.json();
            
            this.updateDetectionList(detections);
            this.checkForUnknownFaces(detections);
        } catch (error) {
            console.error('Error fetching detections:', error);
        }
    }

    updateDetectionList(detections) {
        const detectionList = document.getElementById('detectionList');
        
        if (detections.length === 0) {
            detectionList.innerHTML = '<p class="no-detections">No detections yet</p>';
            return;
        }

        detectionList.innerHTML = detections
            .slice()
            .reverse()
            .map(detection => `
                <div class="detection-item ${detection.name === 'Unknown' ? 'unknown' : ''}">
                    <div class="name">${detection.name === 'Unknown' ? '❓ Unknown Person' : '✅ ' + detection.name}</div>
                    <div class="time">${detection.time}</div>
                </div>
            `)
            .join('');
    }

    checkForUnknownFaces(detections) {
        const recentUnknown = detections.find(detection => 
            detection.name === 'Unknown' && 
            this.isRecentDetection(detection.time)
        );

        if (recentUnknown && !this.unknownFaceDetected) {
            this.showUnknownFaceAlert();
        }
    }

    isRecentDetection(timeString) {
        const detectionTime = new Date(timeString);
        const currentTime = new Date();
        const timeDiff = (currentTime - detectionTime) / 1000; // Difference in seconds
        return timeDiff < 10; // Within last 10 seconds
    }

    showUnknownFaceAlert() {
        this.unknownFaceDetected = true;
        document.getElementById('unknownFaceAlert').style.display = 'block';
        this.showNotification('Unknown face detected! Please add a name.', 'warning');
    }

    async saveUnknownFace() {
        const nameInput = document.getElementById('faceName');
        const name = nameInput.value.trim();

        if (!name) {
            this.showNotification('Please enter a name for the face', 'error');
            return;
        }

        try {
            // In a real implementation, you would capture the current frame
            // For now, we'll simulate this
            const response = await fetch('/add_face', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: '', // You would add actual image data here
                    name: name
                })
            });

            if (response.ok) {
                this.showNotification(`Face saved as: ${name}`, 'success');
                nameInput.value = '';
                document.getElementById('unknownFaceAlert').style.display = 'none';
                this.unknownFaceDetected = false;
                
                // Refresh detections
                await this.updateDetections();
            }
        } catch (error) {
            this.showNotification('Error saving face: ' + error.message, 'error');
        }
    }

    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        // Set background color based on type
        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196F3'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(notification);

        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new FaceRecognitionApp();
});

// Add CSS for notification animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);