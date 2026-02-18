# SNOW Reproduction Notes (VLM4D)

This file is the working reference for the SNOW reproduction. It captures the end-to-end pipeline, required parameters, and implementation details that must remain consistent with the paper.

## Scope
- Dataset: VLM4D (primary)
- Parameters and models: match the paper settings
- Output: 4D Scene Graph (4DSG) + VLM inference on 4DSG

## Pipeline (End-to-End)
1) Inputs (per timestep t)
   - Synchronized RGB images {I_t^c} and point cloud P_t
   - Assumption: temporal alignment + geometric calibration

2) Point Cloud Clustering and Sampling
   - Unmapped points U_t <- P_t
   - HDBSCAN over U_t to produce clusters R_t
   - Uniformly sample m points per cluster (paper: m = 4)
   - Sampled points act as SAM2 point prompts

3) Projection + SAM2 Mask Generation
   - Project each point in P_t to camera planes using intrinsics/extrinsics
   - For each cluster proposal, prompt SAM2 with {V_t^k}_img
   - SAM2 returns masks m_t,k,c
   - Enforce cross-view consistency via Hungarian matching

4) 3D-2D Association
   - Assign each projected point to a mask if (x_img, y_img) in m_t,k,c
   - Use this to associate 3D points with object masks

5) STEP Encoding (Spatio-Temporal Tokenized Patch Encoding)
   - Extract mask region
   - Partition masked image into 16x16 grid (256 patches)
   - Keep patches with IoU > 0.5 as image patch tokens
   - Append 3D tokens:
     - Centroid token c_t^k = (x_bar, y_bar, z_bar)
     - Shape token s_t^k with per-axis (mu, sigma, a_min, a_max)
   - Append temporal tokens:
     - theta_t^k = (t_start, t_end)
   - Full token set: S_t^k = {patch tokens, centroid, shape, temporal}

6) Iterative Refinement
   - For N_iter iterations (paper: 1)
   - Reintroduce residual points into SAM2
   - H_hop reasoning (paper: 1) to detect implausible geometry
   - Reassign implausible points to U_t

7) 4D Scene Graph (4DSG)
   - Per-frame scene graph G_t = (V_t, E_t)
   - Nodes = STEP token sets; edges from 3D proximity / relative orientation
   - Temporal linking over window T (paper: T = 10)
   - Object sequences F_k = {S_{t-T}^k ... S_t^k}
   - Aggregate to 4DSG M_t

8) SLAM Anchoring
   - Use SLAM backend for global coordinate alignment
   - Paper: KISS-SLAM for LiDAR, MapAnything for image-only

9) VLM Inference
   - Query on 4DSG directly
   - y_hat = VLM(q | M_t)

## Paper Configuration (Must Match)
- Window size: T = 10 frames
- Iterations: N_iter = 1
- Hop reasoning: H_hop = 1
- SAM2: Hiera Large video predictor
- VLM: Gemma3-4B-IT
- SLAM: KISS-SLAM (LiDAR) or MapAnything (image-only)
- Performance note: ~1.1 FPS on H100 (pipeline speed reference)

## Data and Evaluation
- VLM4D benchmark: evaluate reasoning accuracy
- Evaluation splits and protocols must match official VLM4D scoring
- Verify temporal window alignment and question subsets if used

## Implementation Notes
- Keep data flow modular: clustering -> segmentation -> STEP -> 4DSG -> VLM
- Track all coordinate frames explicitly (sensor -> camera -> global)
- Use deterministic sampling for reproducibility if possible
- Store STEP tokens as structured objects for easy serialization
- Keep logs for object linking, reassigned points, and masks per frame

## Planned Modules (Directory Mapping)
- fast_snow/data/ : loaders, calibration, projection
- fast_snow/vision/perception/ : clustering (HDBSCAN), segmentation (SAM2), association, refinement
- fast_snow/reasoning/tokens/ : STEP encoding, patch tokenizer, 3D token builder
- fast_snow/reasoning/graph/ : scene graph + temporal linking + 4DSG
- fast_snow/vision/slam/ : SLAM backend wrappers
- fast_snow/reasoning/vlm/ : prompt building + inference
- fast_snow/reasoning/eval/ : VLM4D evaluation scripts

## Open Items
- Confirm VLM4D dataset path and structure
- Confirm if official evaluation code is provided and how to integrate
- Confirm if VLM4D includes point clouds directly or requires reconstruction
