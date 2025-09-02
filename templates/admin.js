// Admin Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Form handling
    const rtspForm = document.querySelector('.admin-card:nth-child(1) .admin-form');
    const videoForm = document.querySelector('.admin-card:nth-child(2) .admin-form');
    
    // Camera controls
    const startCameraBtn = document.getElementById('start-camera');
    const stopCameraBtn = document.getElementById('stop-camera');
    const testCameraBtn = document.getElementById('test-camera');
    const capturePhotoBtn = document.getElementById('capture-photo');
    const liveCamera = document.getElementById('live-camera');
    const cameraCanvas = document.getElementById('camera-canvas');
    
    let stream = null;
    let cameras = [];

    // RTSP Camera Form Handler
    rtspForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const rtspUrl = document.getElementById('rtsp-url').value;
        const cameraName = document.getElementById('camera-name').value;
        const cameraLocation = document.getElementById('camera-location').value;
        
        if (!rtspUrl || !cameraName || !cameraLocation) {
            showNotification('Please fill in all fields', 'error');
            return;
        }
        
        // Validate RTSP URL format
        if (!rtspUrl.startsWith('rtsp://')) {
            showNotification('Please enter a valid RTSP URL', 'error');
            return;
        }
        
        // Add camera to list
        const camera = {
            id: Date.now(),
            name: cameraName,
            location: cameraLocation,
            url: rtspUrl,
            type: 'rtsp',
            status: 'online',
            addedAt: new Date()
        };
        
        cameras.push(camera);
        updateCameraList();
        rtspForm.reset();
        showNotification('Camera added successfully!', 'success');
    });

    // Video Upload Form Handler
    videoForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const videoFile = document.getElementById('video-file').files[0];
        const description = document.getElementById('video-description').value;
        const uploadDate = document.getElementById('upload-date').value;
        
        if (!videoFile) {
            showNotification('Please select a video file', 'error');
            return;
        }
        
        // Validate file type
        if (!videoFile.type.startsWith('video/')) {
            showNotification('Please select a valid video file', 'error');
            return;
        }
        
        // Simulate upload process
        showNotification('Uploading video...', 'info');
        
        setTimeout(() => {
            const video = {
                id: Date.now(),
                name: videoFile.name,
                description: description,
                uploadDate: uploadDate,
                size: videoFile.size,
                type: 'uploaded'
            };
            
            showNotification('Video uploaded successfully!', 'success');
            videoForm.reset();
        }, 2000);
    });

    // Test Camera
    testCameraBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/api/test_camera', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            showNotification(result.message, result.success ? 'success' : 'error');
        } catch (error) {
            console.error('Error testing camera:', error);
            showNotification('Failed to test camera. Please try again.', 'error');
        }
    });

    // Live Camera Controls
    startCameraBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/api/start_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                startCameraBtn.disabled = true;
                stopCameraBtn.disabled = false;
                showNotification(result.message, 'success');
            } else {
                showNotification(result.message, 'error');
            }
        } catch (error) {
            console.error('Error starting camera:', error);
            showNotification('Failed to start camera. Please try again.', 'error');
        }
    });

    stopCameraBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/api/stop_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                startCameraBtn.disabled = false;
                stopCameraBtn.disabled = true;
                showNotification(result.message, 'info');
            } else {
                showNotification(result.message, 'error');
            }
        } catch (error) {
            console.error('Error stopping camera:', error);
            showNotification('Failed to stop camera. Please try again.', 'error');
        }
    });

    capturePhotoBtn.addEventListener('click', function() {
        if (!stream) return;
        
        const context = cameraCanvas.getContext('2d');
        cameraCanvas.width = liveCamera.videoWidth;
        cameraCanvas.height = liveCamera.videoHeight;
        
        context.drawImage(liveCamera, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        // Convert to blob and download
        cameraCanvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = camera-capture-${Date.now()}.png;
            a.click();
            URL.revokeObjectURL(url);
        });
        
        showNotification('Photo captured and downloaded!', 'success');
    });

    // Camera Resolution Change
    document.getElementById('camera-resolution').addEventListener('change', function(e) {
        if (stream) {
            const [width, height] = e.target.value.split('x');
            const newConstraints = {
                video: {
                    width: { ideal: parseInt(width) },
                    height: { ideal: parseInt(height) }
                }
            };
            
            // Restart stream with new resolution
            stopCameraBtn.click();
            setTimeout(() => {
                startCameraBtn.click();
            }, 500);
        }
    });

    // Update Camera List
    function updateCameraList() {
        const cameraList = document.getElementById('camera-list');
        cameraList.innerHTML = '';
        
        cameras.forEach(camera => {
            const cameraItem = document.createElement('div');
            cameraItem.className = 'camera-item';
            cameraItem.innerHTML = `
                <h3>${camera.name}</h3>
                <p><strong>Location:</strong> ${camera.location}</p>
                <p><strong>Type:</strong> ${camera.type.toUpperCase()}</p>
                <p><strong>Added:</strong> ${camera.addedAt.toLocaleDateString()}</p>
                <span class="status ${camera.status}">${camera.status.toUpperCase()}</span>
                <button class="admin-btn" onclick="removeCamera(${camera.id})" style="margin-top: 10px;">Remove</button>
            `;
            cameraList.appendChild(cameraItem);
        });
        
        if (cameras.length === 0) {
            cameraList.innerHTML = '<p style="text-align: center; color: #ccc;">No cameras connected</p>';
        }
    }

    // Remove Camera Function (global scope)
    window.removeCamera = function(id) {
        cameras = cameras.filter(camera => camera.id !== id);
        updateCameraList();
        showNotification('Camera removed', 'info');
    };

    // Notification System
    function showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());
        
        const notification = document.createElement('div');
        notification.className = notification ${type};
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        
        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            info: '#007bff',
            warning: '#ffc107'
        };
        
        notification.style.backgroundColor = colors[type] || colors.info;
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Add CSS animations for notifications
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Check camera status on page load
    async function checkCameraStatus() {
        try {
            const response = await fetch('/api/camera_status');
            const result = await response.json();
            
            const statusIndicator = document.getElementById('camera-status-indicator');
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');
            
            if (result.running) {
                startCameraBtn.disabled = true;
                stopCameraBtn.disabled = false;
                statusDot.style.backgroundColor = '#28a745';
                statusText.textContent = 'Camera Online';
            } else {
                startCameraBtn.disabled = false;
                stopCameraBtn.disabled = true;
                statusDot.style.backgroundColor = '#dc3545';
                statusText.textContent = 'Camera Offline';
            }
        } catch (error) {
            console.error('Error checking camera status:', error);
        }
    }

    // Check status on page load
    checkCameraStatus();

    // Check status every 5 seconds
    setInterval(checkCameraStatus, 5000);

    // Initialize camera list
    updateCameraList();
});