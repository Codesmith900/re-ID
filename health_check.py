from flask import Flask, jsonify, request
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import psutil
import time
import threading
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonReIDHealthMonitor:
    def __init__(self, person_folders_path="D:\\re-ID\\persons\\id_108.0"):
        self.person_folders_path = Path(person_folders_path)
        self.system_start_time = datetime.now()
        
        # Re-ID specific metrics
        self.reid_stats = {
            "total_person_folders": 0,
            "backup_folders_count": 0,
            "last_merge_operation": None,
            "total_merges_performed": 0,
            "merge_errors": [],
            "duplicate_candidates": [],
            "average_images_per_folder": 0,
            "largest_folder_size": 0,
            "smallest_folder_size": 0,
            "folders_needing_review": []
        }
        
        # Processing status
        self.processing_status = {
            "is_processing": False,
            "current_operation": None,
            "progress_percentage": 0,
            "start_time": None,
            "estimated_completion": None
        }
        
        # Model/feature extraction health
        self.model_health = {
            "opencv_status": "unknown",
            "feature_extraction_working": False,
            "last_feature_test": None,
            "average_processing_time_ms": 0
        }
        
        # Storage and file system health
        self.storage_health = {
            "disk_space_available_gb": 0,
            "estimated_space_needed_gb": 0,
            "can_perform_operations": True,
            "backup_space_usage_gb": 0
        }
        
        # Start background monitoring
        self.monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitoring_thread.start()
        
        # Test feature extraction on startup
        self._test_feature_extraction()
    
    def _background_monitoring(self):
        """Background thread to continuously monitor re-ID system health"""
        while True:
            try:
                self._update_reid_stats()
                self._update_storage_health()
                self._check_model_health()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _test_feature_extraction(self):
        """Test if feature extraction is working properly"""
        try:
            # Create a test image
            test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            
            start_time = time.time()
            
            # Test basic OpenCV operations
            test_resized = cv2.resize(test_image, (64, 128))
            hsv = cv2.cvtColor(test_resized, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            
            processing_time = (time.time() - start_time) * 1000  
            
            self.model_health["opencv_status"] = "healthy"
            self.model_health["feature_extraction_working"] = True
            self.model_health["last_feature_test"] = datetime.now()
            self.model_health["average_processing_time_ms"] = processing_time
            
            logger.info(f"Feature extraction test passed in {processing_time:.2f}ms")
            
        except Exception as e:
            self.model_health["opencv_status"] = "error"
            self.model_health["feature_extraction_working"] = False
            self.model_health["last_feature_test"] = datetime.now()
            logger.error(f"Feature extraction test failed: {e}")
    
    def _update_reid_stats(self):
        """Update person re-identification statistics"""
        try:
            if not self.person_folders_path.exists():
                return
            
            person_folders = []
            backup_folders = 0
            total_images = 0
            folder_sizes = []
            
            for folder in self.person_folders_path.iterdir():
                if folder.is_dir():
                    if "backup" in folder.name.lower():
                        backup_folders += 1
                    elif folder.name.isdigit():
                        person_folders.append(folder)
                        
                        # Count images in folder
                        image_count = len([f for f in folder.iterdir() 
                                         if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}])
                        total_images += image_count
                        folder_sizes.append(image_count)
            
            self.reid_stats["total_person_folders"] = len(person_folders)
            self.reid_stats["backup_folders_count"] = backup_folders
            
            if folder_sizes:
                self.reid_stats["average_images_per_folder"] = total_images / len(person_folders)
                self.reid_stats["largest_folder_size"] = max(folder_sizes)
                self.reid_stats["smallest_folder_size"] = min(folder_sizes)
            
            # Find potential duplicates (folders with unusually high image counts)
            avg_size = self.reid_stats["average_images_per_folder"]
            threshold = avg_size * 2  # Folders with 2x more images might be duplicates
            
            large_folders = []
            for i, folder in enumerate(person_folders):
                if folder_sizes[i] > threshold:
                    large_folders.append({
                        "folder_id": folder.name,
                        "image_count": folder_sizes[i],
                        "size_ratio": folder_sizes[i] / avg_size if avg_size > 0 else 0
                    })
            
            self.reid_stats["duplicate_candidates"] = large_folders[:10]  # Top 10 candidates
            
            # Check for folders that might need manual review
            review_folders = []
            for folder in person_folders:
                folder_age = datetime.now() - datetime.fromtimestamp(folder.stat().st_mtime)
                if folder_age.days > 7:  # Folders older than 7 days
                    review_folders.append({
                        "folder_id": folder.name,
                        "days_old": folder_age.days,
                        "last_modified": datetime.fromtimestamp(folder.stat().st_mtime).isoformat()
                    })
            
            self.reid_stats["folders_needing_review"] = review_folders[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Error updating re-ID stats: {e}")
    
    def _update_storage_health(self):
        """Update storage and disk space health"""
        try:
            if self.person_folders_path.exists():
                disk_usage = shutil.disk_usage(self.person_folders_path)
                self.storage_health["disk_space_available_gb"] = disk_usage.free / (1024**3)
                
                # Calculate current storage usage
                total_size = 0
                backup_size = 0
                
                for folder in self.person_folders_path.rglob('*'):
                    if folder.is_file():
                        size = folder.stat().st_size
                        total_size += size
                        if "backup" in str(folder):
                            backup_size += size
                
                self.storage_health["backup_space_usage_gb"] = backup_size / (1024**3)
                
                # Estimate space needed for operations (assuming 20% growth)
                current_usage_gb = total_size / (1024**3)
                self.storage_health["estimated_space_needed_gb"] = current_usage_gb * 0.2
                
                # Check if we can perform operations
                available_gb = self.storage_health["disk_space_available_gb"]
                needed_gb = self.storage_health["estimated_space_needed_gb"]
                self.storage_health["can_perform_operations"] = available_gb > (needed_gb + 1)  # 1GB buffer
                
        except Exception as e:
            logger.error(f"Error updating storage health: {e}")
            self.storage_health["can_perform_operations"] = False
    
    def _check_model_health(self):
        """Periodically check model and processing health"""
        if datetime.now() - self.model_health.get("last_feature_test", datetime.min) > timedelta(hours=1):
            self._test_feature_extraction()
    
    def start_processing(self, operation_name, estimated_duration_minutes=None):
        """Mark the start of a processing operation"""
        self.processing_status["is_processing"] = True
        self.processing_status["current_operation"] = operation_name
        self.processing_status["progress_percentage"] = 0
        self.processing_status["start_time"] = datetime.now()
        
        if estimated_duration_minutes:
            estimated_completion = datetime.now() + timedelta(minutes=estimated_duration_minutes)
            self.processing_status["estimated_completion"] = estimated_completion
    
    def update_progress(self, percentage):
        """Update processing progress"""
        self.processing_status["progress_percentage"] = min(100, max(0, percentage))
    
    def finish_processing(self, success=True, error_message=None):
        """Mark the completion of a processing operation"""
        self.processing_status["is_processing"] = False
        self.processing_status["current_operation"] = None
        self.processing_status["progress_percentage"] = 100 if success else 0
        
        if success:
            self.reid_stats["total_merges_performed"] += 1
            self.reid_stats["last_merge_operation"] = datetime.now()
        else:
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "error": error_message or "Unknown error",
                "operation": self.processing_status.get("current_operation", "unknown")
            }
            self.reid_stats["merge_errors"].append(error_entry)
            # Keep only last 10 errors
            if len(self.reid_stats["merge_errors"]) > 10:
                self.reid_stats["merge_errors"] = self.reid_stats["merge_errors"][-10:]
    
    def get_comprehensive_health(self):
        """Get comprehensive health status for person re-ID system"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Calculate overall status
            overall_status = self._calculate_overall_status()
            
            uptime = datetime.now() - self.system_start_time
            
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "status": overall_status,
                "uptime": {
                    "seconds": int(uptime.total_seconds()),
                    "human_readable": str(uptime).split('.')[0]
                },
                "reid_system": {
                    "statistics": self.reid_stats,
                    "processing": self.processing_status,
                    "model_health": self.model_health,
                    "storage_health": self.storage_health
                },
                "system_resources": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "processes_count": len(psutil.pids())
                },
                "recommendations": self._get_recommendations()
            }
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive health: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _calculate_overall_status(self):
        """Calculate overall system health status"""
        try:
            # Check critical components
            if not self.model_health["feature_extraction_working"]:
                return "critical"
            
            if not self.storage_health["can_perform_operations"]:
                return "critical"
            
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90 or memory_percent > 90:
                return "critical"
            
            # Check for recent errors
            recent_errors = [e for e in self.reid_stats["merge_errors"] 
                           if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)]
            
            if len(recent_errors) > 3:
                return "warning"
            
            # Check processing status
            if self.processing_status["is_processing"]:
                return "processing"
            
            if cpu_percent > 70 or memory_percent > 70:
                return "warning"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Error calculating status: {e}")
            return "error"
    
    def _get_recommendations(self):
        """Get system recommendations based on current health"""
        recommendations = []
        
        try:
            # Storage recommendations
            if self.storage_health["disk_space_available_gb"] < 5:
                recommendations.append({
                    "type": "critical",
                    "message": "Low disk space! Less than 5GB available.",
                    "action": "Free up space or expand storage"
                })
            
            # Duplicate detection recommendations
            if len(self.reid_stats["duplicate_candidates"]) > 0:
                recommendations.append({
                    "type": "info",
                    "message": f"Found {len(self.reid_stats['duplicate_candidates'])} folders that might contain duplicates",
                    "action": "Run re-identification process on large folders"
                })
            
            # Performance recommendations
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:
                recommendations.append({
                    "type": "warning",
                    "message": f"High CPU usage: {cpu_percent:.1f}%",
                    "action": "Consider reducing concurrent operations"
                })
            
            # Backup recommendations
            if self.reid_stats["backup_folders_count"] > 10:
                recommendations.append({
                    "type": "info",
                    "message": f"Many backup folders ({self.reid_stats['backup_folders_count']}) taking up space",
                    "action": "Consider cleaning old backups"
                })
            
            # Model health recommendations
            if not self.model_health["feature_extraction_working"]:
                recommendations.append({
                    "type": "critical",
                    "message": "Feature extraction not working properly",
                    "action": "Check OpenCV installation and dependencies"
                })
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                "type": "error",
                "message": "Could not generate recommendations",
                "action": "Check system logs"
            })
        
        return recommendations

# Initialize health monitor
health_monitor = PersonReIDHealthMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Main health check endpoint for Person Re-ID system"""
    try:
        health_data = health_monitor.get_comprehensive_health()
        
        # Set HTTP status code based on health
        status_code = 200
        system_status = health_data.get('status', 'unknown')
        
        if system_status == 'processing':
            status_code = 202  # Accepted - processing
        elif system_status == 'warning':
            status_code = 200  # OK but with warnings
        elif system_status in ['critical', 'error']:
            status_code = 503  # Service Unavailable
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check endpoint error: {e}")
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/health/reid', methods=['GET'])
def reid_specific_health():
    """Re-ID system specific health check"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "reid_statistics": health_monitor.reid_stats,
        "processing_status": health_monitor.processing_status,
        "model_health": health_monitor.model_health
    })

@app.route('/health/storage', methods=['GET'])
def storage_health():
    """Storage and disk space health check"""
    health_monitor._update_storage_health()  # Force immediate update
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "storage_health": health_monitor.storage_health,
        "person_folders_path": str(health_monitor.person_folders_path)
    })

@app.route('/health/processing', methods=['GET'])
def processing_status():
    """Current processing status"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "processing": health_monitor.processing_status
    })

@app.route('/health/recommendations', methods=['GET'])
def get_recommendations():
    """Get system recommendations"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "recommendations": health_monitor._get_recommendations()
    })

# Processing control endpoints
@app.route('/processing/start', methods=['POST'])
def start_processing():
    """Start a processing operation"""
    try:
        data = request.get_json() or {}
        operation_name = data.get('operation', 'Unknown Operation')
        estimated_duration = data.get('estimated_duration_minutes')
        
        health_monitor.start_processing(operation_name, estimated_duration)
        
        return jsonify({
            "message": "Processing started",
            "operation": operation_name,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/processing/progress', methods=['POST'])
def update_progress():
    """Update processing progress"""
    try:
        data = request.get_json()
        percentage = data.get('percentage', 0)
        
        health_monitor.update_progress(percentage)
        
        return jsonify({
            "message": "Progress updated",
            "percentage": percentage,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/processing/finish', methods=['POST'])
def finish_processing():
    """Mark processing as finished"""
    try:
        data = request.get_json() or {}
        success = data.get('success', True)
        error_message = data.get('error_message')
        
        health_monitor.finish_processing(success, error_message)
        
        return jsonify({
            "message": "Processing finished",
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Configuration
    PERSON_FOLDERS_PATH = "D:\\re-ID\\persons\\id_108.0"  
    HOST = '0.0.0.0'
    PORT = 5001  # different port from camera health check
    
    # Update health monitor configuration
    health_monitor.person_folders_path = Path(PERSON_FOLDERS_PATH)
    
    print(f"Starting Person Re-ID Health Check API server...")
    print(f"Person folders path: {PERSON_FOLDERS_PATH}")
    print(f"Server will run on http://{HOST}:{PORT}")
    print("\nAvailable endpoints:")
    print("  GET  /health                 - Complete health check")
    print("  GET  /health/reid            - Re-ID system specific health")
    print("  GET  /health/storage         - Storage health")
    print("  GET  /health/processing      - Current processing status")
    print("  GET  /health/recommendations - System recommendations")
    print("  POST /processing/start       - Start processing operation")
    print("  POST /processing/progress    - Update processing progress")
    print("  POST /processing/finish      - Mark processing finished")
    
    app.run(host=HOST, port=PORT, debug=False)

    PERSON_FOLDERS_PATH = "D:\\re-ID\\persons\\id_108.0"  
    HOST = '0.0.0.0'
    PORT = 5001