import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import tempfile
import os
from PIL import Image
import io
import warnings

# Suppress the use_column_width deprecation warning
warnings.filterwarnings("ignore", message=".*use_column_width.*")

class ImageStitcher:
    def __init__(self, max_dim=1600, step_size=8, lowe_ratio=0.75, 
                 ransac_thresh=4.0, min_inliers=30):
        self.MAX_DIM = max_dim
        self.step_size = step_size
        self.LOWE_RATIO = lowe_ratio
        self.RANSAC_THRESH = ransac_thresh
        self.MIN_INLIERS = min_inliers
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def preprocess_images(self, uploaded_files):
        """Load and preprocess images from uploaded files"""
        images_color = []
        images_gray = []
        image_names = []
        
        for uploaded_file in uploaded_files:
            # Read image from uploaded file
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3:
                color = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                color = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            # Resize if too large
            h, w = color.shape[:2]
            if self.MAX_DIM and max(h, w) > self.MAX_DIM:
                scale = self.MAX_DIM / float(max(h, w))
                color = cv2.resize(color, (int(w*scale), int(h*scale)), 
                                 interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            
            images_color.append(color)
            images_gray.append(gray)
            image_names.append(uploaded_file.name)
        
        return images_color, images_gray, image_names
    
    def dense_sift(self, img_gray):
        """Extract dense SIFT features"""
        h, w = img_gray.shape[:2]
        keypoints = [cv2.KeyPoint(float(x), float(y), self.step_size)
                     for y in range(0, h, self.step_size)
                     for x in range(0, w, self.step_size)]
        keypoints, descriptors = self.sift.compute(img_gray, keypoints)
        return keypoints, descriptors
    
    def extract_features(self, images_gray):
        """Extract features from all images"""
        keypoints_list = []
        descriptors_list = []
        
        for i, gray in enumerate(images_gray):
            kps, des = self.dense_sift(gray)
            if des is None:
                des = np.zeros((0, 128), dtype=np.float32)
            keypoints_list.append(kps)
            descriptors_list.append(des)
        
        return keypoints_list, descriptors_list
    
    def find_pairwise_homographies(self, keypoints_list, descriptors_list):
        """Find homographies between image pairs"""
        pairwise_H = {}
        pairwise_inliers = {}
        N = len(descriptors_list)
        
        for i in range(N):
            for j in range(i+1, N):
                des_i = descriptors_list[i]
                des_j = descriptors_list[j]
                
                if des_i.shape[0] == 0 or des_j.shape[0] == 0:
                    continue
                
                # Find matches
                knn_matches = self.bf.knnMatch(des_j, des_i, k=2)
                
                # Apply Lowe ratio test
                good = []
                for m_n in knn_matches:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < self.LOWE_RATIO * n.distance:
                        good.append(m)
                
                if len(good) < 8:
                    continue
                
                # Extract point correspondences
                pts_j = np.float32([keypoints_list[j][m.queryIdx].pt for m in good]).reshape(-1, 2)
                pts_i = np.float32([keypoints_list[i][m.trainIdx].pt for m in good]).reshape(-1, 2)
                
                # Find homography using RANSAC
                H, mask = cv2.findHomography(pts_j, pts_i, cv2.RANSAC, self.RANSAC_THRESH)
                
                if H is None or mask is None:
                    continue
                
                inliers = int(mask.sum())
                
                if inliers >= self.MIN_INLIERS:
                    pairwise_H[(i, j)] = H
                    pairwise_inliers[(i, j)] = inliers
        
        return pairwise_H, pairwise_inliers
    
    def find_center_image(self, pairwise_H, N):
        """Find the center image for stitching"""
        adj = defaultdict(list)
        for (i, j), H in pairwise_H.items():
            adj[i].append(j)
            adj[j].append(i)
        
        if len(adj) == 0:
            raise RuntimeError("No pairwise homographies found. Images might not have enough overlap.")
        
        center = max(range(N), key=lambda x: len(adj[x]) if x in adj else 0)
        return center, adj
    
    def compute_global_homographies(self, pairwise_H, center, adj, N):
        """Compute homographies to center image using BFS"""
        H_to_center = {center: np.eye(3, dtype=np.float64)}
        visited = set([center])
        q = deque([center])
        
        while q:
            p = q.popleft()
            for nb in adj.get(p, []):
                if nb in visited:
                    continue
                
                H_nb_to_p = None
                if (p, nb) in pairwise_H:
                    H_nb_to_p = pairwise_H[(p, nb)]
                elif (nb, p) in pairwise_H:
                    H_p_to_nb = pairwise_H[(nb, p)]
                    try:
                        H_nb_to_p = np.linalg.inv(H_p_to_nb)
                    except np.linalg.LinAlgError:
                        H_nb_to_p = None
                
                if H_nb_to_p is None:
                    continue
                
                H_p_to_center = H_to_center[p]
                H_nb_to_center = H_p_to_center @ H_nb_to_p
                H_nb_to_center = H_nb_to_center / H_nb_to_center[2, 2]
                
                H_to_center[nb] = H_nb_to_center
                visited.add(nb)
                q.append(nb)
        
        return H_to_center
    
    def compute_canvas_bounds(self, images_color, H_to_center):
        """Compute the bounds of the final panorama canvas"""
        all_corners = []
        for idx, img in enumerate(images_color):
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            H = H_to_center.get(idx, np.eye(3))
            trans = cv2.perspectiveTransform(corners, H)
            all_corners.append(trans.reshape(-1, 2))
        
        all_corners = np.vstack(all_corners)
        xmin, ymin = np.min(all_corners, axis=0)
        xmax, ymax = np.max(all_corners, axis=0)
        
        tx = -xmin if xmin < 0 else 0
        ty = -ymin if ymin < 0 else 0
        out_w = int(np.ceil(xmax + tx))
        out_h = int(np.ceil(ymax + ty))
        
        return out_w, out_h, tx, ty
    
    def stitch_images(self, images_color, H_to_center, out_w, out_h, tx, ty):
        """Stitch images using feathering blend"""
        accum_img = np.zeros((out_h, out_w, 3), dtype=np.float64)
        accum_weight = np.zeros((out_h, out_w), dtype=np.float64)
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        eps = 1e-6
        
        for idx, img in enumerate(images_color):
            H = H_to_center.get(idx, np.eye(3))
            H_full = T @ H
            h, w = img.shape[:2]
            
            # Warp image
            warped = cv2.warpPerspective(img, H_full, (out_w, out_h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT)
            
            # Create and warp mask
            mask = np.ones((h, w), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, H_full, (out_w, out_h), 
                                            flags=cv2.INTER_NEAREST, 
                                            borderMode=cv2.BORDER_CONSTANT)
            warped_mask_bin = (warped_mask > 0).astype(np.uint8) * 255
            
            # Compute weight using distance transform
            weight = np.zeros_like(warped_mask_bin, dtype=np.float32)
            if warped_mask_bin.sum() > 0:
                dt = cv2.distanceTransform(warped_mask_bin, 
                                         distanceType=cv2.DIST_L2, 
                                         maskSize=5)
                maxv = dt.max() if dt.max() > 0 else 1.0
                weight = dt / maxv
            
            # Accumulate
            warped_f = warped.astype(np.float64)
            for c in range(3):
                accum_img[:, :, c] += warped_f[:, :, c] * weight
            accum_weight += weight
        
        # Normalize
        norm = accum_weight[..., None] + eps
        panorama = (accum_img / norm).astype(np.uint8)
        
        return panorama
    
    def create_panorama(self, uploaded_files):
        """Main function to create panorama from uploaded files"""
        try:
            # Preprocess images
            images_color, images_gray, image_names = self.preprocess_images(uploaded_files)
            
            # Extract features
            keypoints_list, descriptors_list = self.extract_features(images_gray)
            
            # Find pairwise homographies
            pairwise_H, pairwise_inliers = self.find_pairwise_homographies(
                keypoints_list, descriptors_list)
            
            if not pairwise_H:
                raise RuntimeError("No matching pairs found. Images might not have enough overlap.")
            
            # Find center image
            center, adj = self.find_center_image(pairwise_H, len(images_color))
            
            # Compute global homographies
            H_to_center = self.compute_global_homographies(
                pairwise_H, center, adj, len(images_color))
            
            # Compute canvas bounds
            out_w, out_h, tx, ty = self.compute_canvas_bounds(images_color, H_to_center)
            
            # Stitch images
            panorama = self.stitch_images(images_color, H_to_center, out_w, out_h, tx, ty)
            
            # Convert BGR to RGB for display
            panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            
            return panorama_rgb, image_names, center
            
        except Exception as e:
            raise RuntimeError(f"Stitching failed: {str(e)}")

def main():
    st.set_page_config(page_title="Image Stitching App", layout="wide")
    
    st.title("🖼️ Image Stitching Application")
    st.markdown("Upload multiple overlapping images to create a panorama")
    
    # Sidebar for parameters
    st.sidebar.title("Stitching Parameters")
    max_dim = st.sidebar.slider("Max Image Dimension", 800, 2000, 1600, 100)
    step_size = st.sidebar.slider("SIFT Step Size", 4, 16, 8, 2)
    lowe_ratio = st.sidebar.slider("Lowe Ratio", 0.5, 0.9, 0.75, 0.05)
    min_inliers = st.sidebar.slider("Min Inliers", 10, 100, 30, 5)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        help="Upload at least 2 overlapping images"
    )
    
    if uploaded_files and len(uploaded_files) >= 2:
        st.success(f"Uploaded {len(uploaded_files)} images")
        
        # Show uploaded images
        st.subheader("Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, uploaded_file in enumerate(uploaded_files[:4]):
            with cols[i % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        if len(uploaded_files) > 4:
            st.info(f"Showing first 4 images. Total uploaded: {len(uploaded_files)}")
        
        # Stitch button
        if st.button("🔧 Stitch Images", type="primary"):
            with st.spinner("Stitching images... This may take a while."):
                try:
                    # Create stitcher
                    stitcher = ImageStitcher(
                        max_dim=max_dim,
                        step_size=step_size,
                        lowe_ratio=lowe_ratio,
                        min_inliers=min_inliers
                    )
                    
                    # Create panorama
                    panorama, image_names, center = stitcher.create_panorama(uploaded_files)
                    
                    # Display results
                    st.success("✅ Stitching completed successfully!")
                    st.subheader("Stitched Panorama")
                    
                    # Show center image info
                    st.info(f"Center image used: {image_names[center]}")
                    
                    # Display panorama
                    st.image(panorama, caption="Stitched Panorama", use_container_width=True)
                    
                    # Download button
                    pil_image = Image.fromarray(panorama)
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Panorama",
                        data=byte_im,
                        file_name="stitched_panorama.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during stitching: {str(e)}")
                    st.markdown("""
                    **Common issues:**
                    - Images don't have enough overlap
                    - Images are too different (lighting, angle, etc.)
                    - Not enough matching features detected
                    
                    **Try:**
                    - Using images with more overlap (30-50%)
                    - Adjusting the parameters in the sidebar
                    - Using images taken from similar viewpoints
                    """)
    
    elif uploaded_files and len(uploaded_files) < 2:
        st.warning("Please upload at least 2 images for stitching")
    
    else:
        st.info("Please upload images to get started")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. **Upload Images**: Choose 2 or more overlapping images
        2. **Adjust Parameters**: Use the sidebar to fine-tune stitching parameters
        3. **Stitch**: Click the "Stitch Images" button
        4. **Download**: Save your panorama
        
        ### Tips for best results:
        - Images should have 30-50% overlap
        - Take photos from the same position, rotating the camera
        - Avoid moving objects in the scene
        - Use consistent lighting and exposure
        - Images should be roughly aligned horizontally
        """)

if __name__ == "__main__":
    main()