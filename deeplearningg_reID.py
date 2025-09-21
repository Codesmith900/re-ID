import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
from health_check import PersonReIDHealthMonitor  # linking to the health check module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepFeatureReID:
    def __init__(self, base_folder_path, similarity_threshold=0.7, health_monitor=None):
        self.base_folder_path = Path(base_folder_path)
        self.similarity_threshold = similarity_threshold
        self.health_monitor = health_monitor

        # Load pre-trained ResNet50 model (without top layer for feature extraction)
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("Loaded ResNet50 model for feature extraction")
        
    def preprocess_image(self, img_path, target_size=(224, 224)):
        try:
            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {e}")
            return None
    
    def extract_features(self, img_path):
        preprocessed_img = self.preprocess_image(img_path)
        if preprocessed_img is None:
            return None
        try:
            features = self.model.predict(preprocessed_img, verbose=0)
            return features.flatten()
        except Exception as e:
            logger.error(f"Error extracting features from {img_path}: {e}")
            return None
    
    def get_folder_representative_features(self, folder_path, max_images=10):
        folder_features = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]
        image_files = image_files[:max_images] if len(image_files) > max_images else image_files
        
        for image_file in image_files:
            features = self.extract_features(str(image_file))
            if features is not None:
                folder_features.append(features)
        
        if not folder_features:
            logger.warning(f"No features extracted from folder {folder_path.name}")
            return None
        
        avg_features = np.mean(folder_features, axis=0)
        logger.info(f"Processed {len(folder_features)} images from folder {folder_path.name}")
        return avg_features
    
    def calculate_similarity(self, features1, features2):
        f1 = features1.reshape(1, -1)
        f2 = features2.reshape(1, -1)
        similarity = cosine_similarity(f1, f2)[0][0]
        return (similarity + 1) / 2
    
    def find_similar_folders(self, target_folder_id):
        target_folder_path = self.base_folder_path / target_folder_id
        if not target_folder_path.exists():
            logger.error(f"Target folder {target_folder_id} does not exist")
            return []
        
        target_features = self.get_folder_representative_features(target_folder_path)
        if target_features is None:
            logger.error(f"Could not extract features from target folder {target_folder_id}")
            return []
        
        similar_folders = [(target_folder_id, 1.0)]
        for folder_path in self.base_folder_path.iterdir():
            if folder_path.is_dir() and folder_path.name != target_folder_id:
                folder_features = self.get_folder_representative_features(folder_path)
                if folder_features is not None:
                    similarity = self.calculate_similarity(target_features, folder_features)
                    logger.info(f"Similarity between {target_folder_id} and {folder_path.name}: {similarity:.3f}")
                    if similarity >= self.similarity_threshold:
                        similar_folders.append((folder_path.name, similarity))
                        logger.info(f"Found similar folder: {folder_path.name} (similarity: {similarity:.3f})")
        
        similar_folders.sort(key=lambda x: x[1], reverse=True)
        return similar_folders
    
    def merge_folders(self, target_folder_id, folders_to_merge, backup=True):
        target_folder_path = self.base_folder_path / target_folder_id
        if backup:
            backup_folder = self.base_folder_path / "backup_before_merge"
            backup_folder.mkdir(exist_ok=True)
            logger.info(f"Creating backup in {backup_folder}")

        merged_count = 0

        for folder_id, similarity_score in folders_to_merge:
            if folder_id == target_folder_id:
                continue
            source_folder = self.base_folder_path / folder_id
            if not source_folder.exists():
                logger.warning(f"Source folder {folder_id} does not exist, skipping")
                continue

            if self.health_monitor:
                self.health_monitor.start_processing(
                    operation_name=f"Merging folder {folder_id} into {target_folder_id}"
                )

            try:
                if backup:
                    backup_dest = backup_folder / folder_id
                    shutil.copytree(source_folder, backup_dest, dirs_exist_ok=True)
                    logger.info(f"Backed up folder {folder_id}")

                image_files = [item for item in source_folder.iterdir() if item.is_file()]
                total_files = len(image_files)
                file_count = 0

                for item in image_files:
                    dest_file = target_folder_path / f"{folder_id}_{item.name}"
                    counter = 1
                    while dest_file.exists():
                        name_parts = item.name.rsplit('.', 1)
                        if len(name_parts) == 2:
                            dest_file = target_folder_path / f"{folder_id}_{name_parts[0]}_{counter}.{name_parts[1]}"
                        else:
                            dest_file = target_folder_path / f"{folder_id}_{item.name}_{counter}"
                        counter += 1
                    shutil.move(str(item), str(dest_file))
                    file_count += 1

                    if self.health_monitor and total_files > 0 and file_count % 10 == 0:
                        progress = int((file_count / total_files) * 100)
                        self.health_monitor.update_progress(progress)

                source_folder.rmdir()
                logger.info(f"Merged folder {folder_id} into {target_folder_id} "
                            f"(similarity: {similarity_score:.3f}, {file_count} files)")
                merged_count += 1

                if self.health_monitor:
                    self.health_monitor.update_progress(100)
                    self.health_monitor.finish_processing(success=True)

            except Exception as e:
                logger.error(f"Error merging folder {folder_id}: {e}")
                if self.health_monitor:
                    self.health_monitor.finish_processing(success=False, error_message=str(e))

        return merged_count


def main():
    # Configuration
    BASE_FOLDER_PATH = "D:\\re-ID\\persons"  
    TARGET_FOLDER_ID = "id_108.0"  
    SIMILARITY_THRESHOLD = 0.75  

    print("Initializing health monitor...")
    health_monitor = PersonReIDHealthMonitor(person_folders_path=BASE_FOLDER_PATH)

    print("Initializing Deep Feature Re-identification system...")
    reid_system = DeepFeatureReID(
        base_folder_path=BASE_FOLDER_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        health_monitor=health_monitor
    )

    print(f"\nProcessing re-identification for folder {TARGET_FOLDER_ID}...")
    similar_folders = reid_system.find_similar_folders(TARGET_FOLDER_ID)
    
    if len(similar_folders) <= 1:
        print(f"No similar folders found for {TARGET_FOLDER_ID}")
        return

    print(f"\nFound {len(similar_folders)} similar folders:")
    for folder_id, similarity in similar_folders:
        print(f"  - {folder_id}: {similarity:.3f}")

    folders_to_merge = [f for f in similar_folders if f[0] != TARGET_FOLDER_ID]
    if folders_to_merge:
        print(f"\nThe following folders will be merged into {TARGET_FOLDER_ID}:")
        for folder_id, similarity in folders_to_merge:
            print(f"  - {folder_id} (similarity: {similarity:.3f})")

        response = input("\nProceed with merging? (y/n): ").lower().strip()
        if response == 'y':
            merged_count = reid_system.merge_folders(TARGET_FOLDER_ID, folders_to_merge)
            print(f"\nSuccessfully merged {merged_count} folders into {TARGET_FOLDER_ID}")
            print(f"Backup created in: {BASE_FOLDER_PATH}/backup_before_merge")
        else:
            print("Operation cancelled.")
    else:
        print(f"No folders need to be merged for {TARGET_FOLDER_ID}")


if __name__ == "__main__":
    main()
