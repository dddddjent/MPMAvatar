# MPMAvatar

## Native 4D-DRESS preparation

`preprocess/prepare_4ddress.py` builds templates, garment splits and camera labels
from raw scans, with source files linked in place. `preprocess/track_4ddress.py`
runs native tracking, UV transfer, AO and skin-weight inpainting. See the
[workspace commands and subject-190 validation](../README.md#4d-dress--mpmavatar).
`preprocess/compare_4ddress_templates.py` compares geometry, UVs, connectivity and
garment partitions against author assets independently of vertex numbering.

## Local-grid patch fitting: subject 185 lower

`local_patch_fit.py` implements independent small MPM domains for the existing
subject-185 lower garment. The original-spacing contact-free survey found no
patch meeting the 16-scored-vertex minimum; results and commands are tracked
in the workspace README. Activate
`mpmavatar` and install the additional
proximity dependency from the workspace root:

```sh
pip install -r MPMAvatar/requirements-local-patches.txt
```

The `prepare` stage reads native tracked vertices plus the 30,000-update
appearance offsets. Specify the Take1 reference windows with `--windows`.
Clearance is sampled at lower vertices and triangle centroids
against SMPL-X and all tracking triangles outside lower cloth; the native
32-edge mouth opening is capped in an in-memory clearance-query copy, with the
cap required to lie above the complete lower garment. Lower self
proximity uses non-neighbor vertex pairs. The prescribed collar also meets the
sampled clearance criterion. This does not certify separation between samples
or frames. Selection fails clearly if the requested eligible patch is absent.
`--core-vertex-ids` selects explicit original tracking vertex IDs;
`--max-core-vertices`, `--count`, and `--collar-rings` control automatic selection.

Triangle topology is unchanged. Each patch has a fixed grid origin and a cube
covering its reference window with padding. Cell spacing now matches original
MPMAvatar: `h = 2 * global_bbox_extent / 200`, where the global extent is the
full lower garment's longest dimension at frame 45. For this subject h is
7.749 mm. The native whole-garment grid uses 200³ nodes in a cube of side twice
that global extent. `--padding-cells` sets margins in original-spacing units
(minimum/default 4). The default timestep is the original 400 substeps per frame.
The discarded 0.969-mm trials and their outputs have been removed.
Use the same padding for fitting, evaluation and convergence. The native
full-lower scale from frame 45 and rest geometry from frame 11 remain fixed.
Gravity, thickness, material units and anisotropic constants retain native
simulation units. Refining this grid does not renormalize the patch geometry.

The collar follows linearly interpolated tracked positions exactly at every
substep. Scored interior vertices have a three-cell Chebyshev separation from
prescribed vertex and fully prescribed face-center particles. The runner checks
the actual support on every substep and fails if collar grid velocities reach scored vertices or a
particle escapes its supported grid domain. Local vertex transfers have no
domain clipping. Colliders are omitted for these sampled contact-free patches.

The `fit` stage shares E and rest-height H across all patches and windows, with
fixed `--density`, central finite differences in log-E/H and Adam. E uses the
native exported modulus units. Loss excludes prescribed vertices and initialized
frames and is weighted by scored vertex-frame count. Grid-node arrays remain
dense within each small domain. Physics stages require an allocated CUDA device;
`--help` and preparation do not initialize CUDA.
The `survey` stage reports all eligible connected regions, including those with
no scored interior at the requested spacing. It does not start optimization.
It also caches per-frame clearance masks and full lower geometry.
`local_patch_support_survey.py` diagnoses all sliding windows using actual
3×3×3 footprints at reference interpolation substeps; production selection and
the simulated-trajectory leakage guard remain unchanged.
Fit history records base-rollout, four-probe gradient, and iteration wall time.

Commands are in the [workspace README](../README.md). Fits save `history.csv`,
`summary.json`, `initial_param.npz`, `best_param.npz`, `last_param.npz`, and best predictions. Each
parameter checkpoint's loss belongs to those exact parameters. The atomic
`optimizer_state.json` owns Adam moments, the next point and completed history;
resume regenerates derived reports and uses its frozen `patches/` input snapshot.
Use the same options with `--resume`, or `--stop-after N` for a resumable early
stop within the planned `--iterations` target.

`evaluate` reports interior mean vertex distance, 3D vertex RMSE, XYZ MSE,
per-frame errors, collar drift and grid leakage for a frozen checkpoint.
Grid dimensions and physical spacing are saved separately for each patch.
`convergence` compares frozen-material trajectories across `--cell-size-factors`
(default 1, 0.5 times the original spacing) and substep counts (400 / 800) against
the finest grid and largest substep count,
with `--convergence-tolerance-mm` (default 0.5). These are predictions conditioned
on recorded collar motion. Whole-garment future prediction must be evaluated
separately using the fitted D/E/H; the new NPZ files expose the native keys.
`local_patch_native_training.py` evaluates frozen material files on the original
full 45–56 training window with native contact and attachment movers; it matches
the original XYZ-MSE metric and saves trajectories and per-frame vertex RMSE.

## Differentiable material fitting (experimental)

2026-10-07: `--material-gradient ad` connects D, E and H to a checkpointed
PyTorch/Warp backward. Frame checkpoints contain positions, velocities, APIC C
and the evolving triangle deformation d. During backward, one frame's particle
states are replayed and a single substep tape is reused; grid buffers are rebuilt.
Density controls mass, E controls the elastic coefficients, and H controls the
rest metric. Internal E remains exported E / 100. The in-plane polar rotation
uses its exact 2D formula to avoid singular SVD-vector derivatives at equal stretches.

CPU local FD passes; the 800-step checkpoint/full-tape gradient error is
1.2e-10. Native central FD passes 19/24 two-frame cases at 3% relative / 1e-8
absolute tolerance, with failures at 40 and 400 substeps per frame. The
[complete CPU report](output/differentiable_cpu/report.json) retains those failures. Yield-surface branches and float32 trajectory sensitivity are under
investigation.

2026-10-08 GPU (A100), 4D-DRESS 190 material window (frames 19-30, 400 substeps):
AD loss matches native to 5e-4; AD vs native central FD (ε = 1% of value) is
D 1.8%, E 1.5%, H 0.6% ([report](output/differentiable_gpu/s190_t2_frames12.json)).
Smaller ε is limited by CUDA atomic nondeterminism (native repeat differs by 6e-8);
replay is checked per field at 1e-4 relative (measured 1.1e-5). A 10-update AD fit
lowered training loss 2.35e-4 → 5.62e-5 (no held-out eval).
Commands (run inside a GPU allocation):

```sh
FRAMES=12 bash output/differentiable_gpu/run_s190_check.sh
STOP=10 bash output/differentiable_gpu/run_s190_fit.sh
```

2026-10-08, 185 lower with the original FD fit settings (Take1 frames 45-56, 200 updates), held-out
Take7 garment CD / F (`../README.md` initial-vs-fitted table). Same A100, no contention: AD 25.5 s/update,
FD 36 s/update. AD and FD Adam paths agree through ~6 updates, then AD's E decreases more slowly;
both reach the same minimum (FD D/E/H 1.03/83.2/0.869, AD 1.08/90.4/0.873). The early-stopped row's better
test score comes from stopping at update 5 (E ≈ 229), not from AD.

| Material | Updates | Best train loss | Take7 lower CD / F |
| --- | ---: | ---: | --- |
| Initial (D 1, E 100, H 1) | 0 | 9.17e-4 | 0.921 / 88.69 |
| FD (`s185_t1_lower`) | 200 | 7.92e-5 (step 81) | 0.855 / 89.96 |
| AD, early stop 40 / 1% | 46 | 1.46e-4 (step 5) | 0.786 / 90.86 |
| AD, 200 updates | 200 | 7.90e-5 (step 176) | 0.855 / 89.96 |

Grey video (scan / initial / FD / AD 200 updates): `output/differentiable_gpu/185_lower/grey_comparison.mp4`.

```sh
bash output/differentiable_gpu/run_s185_lower_driver.sh > output/differentiable_gpu/185_lower_driver.log 2>&1   # login node: fit then Take7 eval on GPU holders
STAGE=eval_initial srun --jobid=<holder> --overlap --ntasks=1 --gpus=1 --cpus-per-task=4 --mem=32G bash output/differentiable_gpu/run_s185_lower.sh
cd .. && python MPMAvatar/scripts/eval/score_native_garment.py --prediction-dir MPMAvatar/output/differentiable_gpu/185_lower/s185_t1_lower/seed0 --prepared data/MPMAvatar/4DDress/examples/s185_t1 --split split_idx_lower.npz --label 4 --test-take 7 --test-start 11 --output MPMAvatar/output/differentiable_gpu/initial/scores/s185_t1_lower_ad.json
cd WarpXPBD && python render_mpmavatar_native_grey.py --prepared ../data/MPMAvatar/4DDress/examples/s185_t1 --source test --labels 4 --rollout "Initial D/E/H" ../MPMAvatar/output/differentiable_gpu/185_lower/s185_t1_lower_initial/seed0 --rollout "FD best" ../data/MPMAvatar/4DDress/examples/s185_t1/output/phys/s185_t1_lower/seed0 --rollout "AD best" ../MPMAvatar/output/differentiable_gpu/185_lower/s185_t1_lower/seed0 --output ../MPMAvatar/output/differentiable_gpu/185_lower/grey_comparison.mp4 --slowdown 8 --panel 480 640   # env warp_xpbd_render
```

Native FD remains
the default comparison mode. Start AD in a fresh output directory: its saved
optimizer context cannot resume an FD optimizer. Native direct training accepts
`--material_gradient ad`; the dataset launcher uses the hyphenated option below.
Activate `mpmavatar`, then reuse a trained appearance:

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000_ad --appearance-model output/ClothTransformer/sim_00000/appearance --stage material --material-gradient ad --material-iterations 200 --material-frames all --grid-size 200 --substeps 400
```

2026-10-09: [185 upper/lower sensitivity sweep](configs/material_sensitivity_185.json)
completed and audited. The native training windows are upper 19–30 and
lower 45–56, Take1. Soft/default/stiff D/E/H = 1/10/1, 1/100/1, 1/1000/1.
There are 270 native rollouts (five offsets × three absolute steps × three
D/E/H directions × three centers × two garments), and six center-only AD
evaluations of the training geometry MSE. Exports include all world-space
trajectories, per-vertex Euclidean differences from each stencil's center,
per-frame CSVs, loss FDs and trajectory FDs at h/2h/five-point, and AD errors.
E derivatives are in exported units; float32 rounding is exposed in the CSVs.
Independent garment/parameter cases save AD once and checkpoint each stencil;
`--resume` reuses those completed results across allocations.
[Results](output/material_sensitivity_185/summary.csv): all 270 trajectories,
54 FD stencils and 3,240 frame comparisons are saved. Five centers have finite
AD; lower/soft returns NaN for D/E/H, explicitly marked `ad_status=nonfinite`.
Its largest vertex difference from the stencil center is 1.51 m (maximum mean
76.0 mm). Float32 collapses 24 nonzero perturbations to their center, and eight
stencils have fewer than five distinct effective values.
[Preparation and run commands](../README.md#mpmavatar-material-sensitivity).

## Prepared dataset input

Activate the existing `mpmavatar` environment. `run.py` accepts completed
`cape_avatar`, `clothtransformer_avatar`, and `dgarments_avatar` exports using
their manifest; no dataset flag is needed. Keep `manifest.json`, `ActorsHQ/`,
`a<actor>_s<sequence>/`, and `tracking/` together. Fitted exports also need
`body_models/` with accessible symlink targets; raw exports need `body_motion/`.
From this directory:

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage appearance
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage material
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage evaluate
```

Change `--data` and `--output` for CAPE or D-Garment. The manifest supplies actor,
gender, cameras, tracking, and frame split. Defaults retain 30,000 appearance
iterations, 200 material iterations, a 200-cell grid,
400 substeps, and original Adam/finite-difference `D/E/H` fitting. Both fitting
stages use the complete training prefix by default. `--material-frames N` selects
its first N frames (2 through the training count); `all` selects the full prefix.
All cloth shares these material parameters. Fitted SMPL-X has the supplied 10 fixed
betas. Raw exports use their body mesh topology and per-interval velocities
directly, without SMPL-X, VPoser or skinning assets. No body fitting is added.
Appearance vertex offsets keep their original zero learning rate. Training
resumes `training_state.pt` automatically; completed appearance runs are skipped.
Evaluation still requires a fresh destination. Keep training settings unchanged
when resuming; iteration counts are total targets, not additional steps.

Material saves Adam, LR scheduler, parameters and next step after every update.
Appearance saves Gaussian/mesh/camera/shadow parameters and Adam state every
1,000 iterations and at exports; its shuffled image loader restarts on resume.
Old partial appearance runs without training state cannot resume. For the current
old material checkpoints only, explicitly authorize a fresh Adam/LR schedule:

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000_raw --output output/ClothTransformer/sim_00000_raw --stage material --resume-parameters
```

Steps 0–45 are retained; training continues at 46 toward the 200-step target.
Later reruns need no extra flag. During training, every material step automatically
updates `<output>/material/seed0/history.csv` and `summary.json` with the step
history and last/best D/E/H and losses. No separate export command is required.
Loss is measured before the update; D/E/H are after it (the original checkpoint
convention). New steps also record `evaluated_D/E/H`, the inputs to that loss;
old steps leave these columns blank. E matches terminal/NPZ units. Existing
histories have also been filled in for `sim_00000` (120 steps) and `sim_00000_raw`
(46 steps). View appearance loss curves with:

```sh
tensorboard --logdir output/ClothTransformer/sim_00000_raw/appearance
```

Material fitting uses only the chosen training prefix. Evaluation loads the highest
saved `last_param` iteration in `material/seed0/`, freezes `D/E/H`, and
scores a rollout initialized at the first held-out cloth mesh. Velocity is
estimated from the first two held-out meshes; the fitted rest reference and `H`
remain unchanged. Later free-cloth observations are used only for scoring.
Keep fitting options identical when invoking evaluation. Each export's manifest
defines the appearance-training prefix and future suffix.

2026-09-19: checkpoint selection is independent of `--material-iterations`.
Use `--checkpoint` to select any saved `last_param` or `best_param` file; relative
paths start at the current directory. For example:

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage evaluate --checkpoint output/ClothTransformer/sim_00000/material/seed0/last_param_00119.npz
```

Append `--evaluate-from-start` to save/render a separate continuous rollout from
frame 0. Both modes score only held-out frames using a rollout reset
at the first evaluation frame. `geometry_metrics.json` records
`evaluation_scope: held_out`, the initialization frames, and the separate render
frame range. The initialized first evaluation frame is included in the average
and has zero error. This protocol applies to original and tweaked-body exports.

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage evaluate --evaluate-from-start
```

Future body and any attachment motion remain prescribed driving inputs.
ClothTransformer and D-Garment retain their original visible body surfaces.
Collision uses the exported raw body or fitted SMPL-X; neither adds pins. Free-cloth
observations initialize the evaluation rollout only once using its first two
frames; subsequent observations never reset the simulation. The export supplies the static simulation domain;
ClothTransformer and D-Garment exclude future cloth from its bounds.
`--deterministic` enables Warp 1.17 `GPU_TO_GPU` and deterministic PyTorch
preparation for simulation on the same GPU architecture. The default retains
ordinary atomics; native simulation methods
and physical settings are unchanged. Add the flag to `run.py` simulation stages,
`train_material_params.py`, `replay_material_window.py` or `run_demo.py`. Use a
fresh process/output and keep the mode when resuming. Sensitivity campaigns can
set `"deterministic": true` in their input config before preparation. This flag
covers simulation; it does not guarantee deterministic appearance rendering.

`--precision fp64` selects double precision for simulation tensors, Warp state,
materials, grid, body contact and attachments; the default is `fp32`.
Use it with `run.py` simulation stages, `train_material_params.py`,
`replay_material_window.py` or `run_demo.py`, in a fresh process/output, and keep
the precision when resuming. Appearance preparation/rendering keeps its existing
precision. Sensitivity configs accept `"precision": "fp64"` and report the actual
material rounding in that precision. Forward timing commands and measurements:
[workspace README](../README.md#mpmavatar-fp64-forward-timing).
`benchmark_forward.py --deterministic` enables `GPU_TO_GPU` before kernel imports
and checks exact repeated losses and final `x/v/C/d` states. Both precisions pass
on A40; the workspace table compares deterministic and ordinary forward times.

`evaluation/seed0/` contains `evaluation_predictions.npz` for scoring,
`predictions.npz` and UV meshes for the selected rendering rollout, and
`geometry_metrics.json`/`.md` (per-frame/mean V2V and XYZ MSE on free cloth,
excluding prescribed attachments). Metrics are computed even with `--skip-render`
or the trainer's `--skip_sim` rendering mode. Standalone GT-lighting rendering
requires the current evaluation report; regenerate evaluation for older outputs.
The rendering branches are:

- `gt_lighting/<camera>/`: predicted mesh with the capture's original materials,
  lights, cameras and color management; references are the original capture PNGs.
- `fitted_appearance/<camera>/`: predicted mesh driving the fitted Gaussians,
  with predicted-mesh AO, learned shadow network and fitted camera color correction.

Each contains `pred/`, `gt/`, `pred.mp4`, `gt.mp4` and `comparison.mp4`
(prediction left, reference right; 25 FPS). The evaluate command above runs both;
GT lighting uses `render_python` from the dataset's `config.json` and requires
its retained `capture/` and `preparation/` assets. Image metrics are not computed
by this launcher; `geometry_metrics.json` reports geometry errors only.
Append `--skip-render` for geometry only, or `--skip-video` to omit video encoding.

2026-09-19: dual rendering is connected; three launcher tests passed. Bounded
GPU checks on CAPE, ClothTransformer and D-Garment reproduced one reference
frame per source within 0.001 mean RGB levels (0–255), confirmed changed geometry
changes the second frame, and decoded all three comparison videos correctly.
These are renderer checks; full fitted-checkpoint evaluation remains unrun.

Initial velocity is `25 * (x[1] - x[0])` from training geometry, scaled into the
simulation domain; triangle particles average their vertex velocities. Every
trial resets this velocity, geometry, affine state, rest metric, and solver time.
The export supplies domain scale/centre and prescribed skin/attachment motion.
Checkpoints record initial velocity and its source frames.

Initial tension uses the original `H`: it scales the first mesh's rest Y
coordinates while simulated initial positions stay fixed. `H=1` gives the
initial metric; `H<1` shortens vertical rest lengths and can introduce tension.
This is a single global rest-shape approximation, not recovered per-face stress
or stress-free sewing geometry. CAPE supplies neither measured stress nor material
parameters, so joint `D/E/H` fitting does not establish unique tension recovery.
Initial APIC affine velocity remains zero as upstream.

2026-09-18: six tests passed: shared dispatch for all three datasets/stages,
stage prerequisites and split boundaries, plus four contract/initialization
checks including actual GPU
rollouts for reset repeatability, motion sensitivity, forces induced by `H`, and
continuous prediction across the split. Changing held-out cloth targets changed
the saved errors while leaving predictions identical in the small solver test.
Warp 1.17 uses public tensor conversion and explicit column-matrix constructors.
Full training and evaluation on the new sources remain unrun. The public
launcher is `run.py`; shared input and prediction code lives in `dataset_input.py`
and `future_evaluation.py`.

2026-09-19: raw ClothTransformer colliders are supported in appearance, material
and continuous evaluation. The manifest selects the collider automatically;
the earlier export's `consumer_status` is a generation-time note.
Nine regression tests and a [bounded RTX 5080 check](../data/outputs/mpmavatar_raw_validation/report.json)
passed: the full 100-frame collider archive, two appearance iterations, one
material iteration on frames 0–1, and continuous prediction of frames 2–3
(40-cell grid, 400 substeps). Temporary model outputs were removed; full training,
held-out evaluation and raw-output videos remain unrun.
Use the same environment and stage options:

```sh
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000_raw --output output/ClothTransformer/sim_00000_raw --stage appearance
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000_raw --output output/ClothTransformer/sim_00000_raw --stage material
python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000_raw --output output/ClothTransformer/sim_00000_raw --stage evaluate
```

## MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics (NeurIPS 2025) ##

[Changmin Lee](https://github.com/KAISTChangmin/), [Jihyun Lee](https://jyunlee.github.io/), [Tae-Kyun (T-K) Kim](https://sites.google.com/view/tkkim/home)

KAIST
 
**[\[Project Page\]](https://kaistchangmin.github.io/MPMAvatar/) [\[Paper\]](https://arxiv.org/pdf/2510.01619.pdf) [\[Supplementary Video\]](https://youtu.be/ytrKDNqACqM)**

<p align="center">
  <img src="assets/teaser.jpg" alt="teaser" width="100%"/>
</p>

<p align="center">
  <img src="assets/novelpose.gif" alt="animated1" width="33%"/>
  <img src="assets/zeroshot.gif" alt="animated2" width="66%"/>
</p>

> we present **MPMAvatar**, a framework for creating 3D human avatars from multi-view videos that supports highly realistic, robust animation, as well as photorealistic rendering from free viewpoints. For accurate and robust dynamics modeling, our key idea is to use a Material Point Method-based simulator, which we carefully tailor to model garments with complex deformations and contact with the underlying body by incorporating an anisotropic constitutive model and a novel collision handling algorithm. We combine this dynamics modeling scheme with our canonical avatar that can be rendered using 3D Gaussian Splatting with quasi-shadowing, enabling high-fidelity rendering for physically realistic animations. In our experiments, we demonstrate that MPMAvatar significantly outperforms the existing state-of-the-art physics-based avatar in terms of (1) dynamics modeling accuracy, (2) rendering accuracy, and (3) robustness and efficiency. Additionally, we present a novel application in which our avatar generalizes to unseen interactions in a zero-shot manner—which was not achievable with previous learning-based methods due to their limited simulation generalizability.

&nbsp;

## Environment Setup  

From the workspace root, create the CUDA 13 environment. Native extensions
include A100 (8.0), A40 (8.6), H200 (9.0), and RTX 5080 (12.0) kernels.
Conda uses its configured environment and package directories. `ffmpeg` is
included; the separate `synthetic_avatar` environment provides Blender rendering.

```sh
bash MPMAvatar/setup.sh
conda activate mpmavatar
```

&nbsp;

## Data Preparation 

### ActorsHQ data
Download [ActorsHQ](https://actors-hq.com/) data by filling their request form in this [link](https://drive.google.com/file/d/1QpZIhsUIWLAgQCXWXdj865gSNkjpCRrd/view). We used four sequences: `Actor01-Sequence1`, `Actor02-Sequence1`, `Actor03-Sequence1`, `Actor06-Sequence1`, following the [PhysAvatar](https://qingqing-zhao.github.io/PhysAvatar)'s setting.

### 4D-DRESS data
Download [4D-DRESS](https://eth-ait.github.io/4d-dress/) data by filling their request form in this [link](https://4d-dress.ait.ethz.ch/). We used four sequences: `00170_Inner`, `00185_Inner`, `00190_Inner`, `00191_Inner`

### PhysAvatar data
Download the preprocessed data (e.g. SMPL-X parameters, template meshes, garment part segmentation, uv coordinates) for ActorsHQ dataset by PhysAvatar using their [drive link](https://drive.google.com/drive/folders/1Fl_WqNXAnZbAOHJwbcav5FYvFSLbh6OQ) and unpack it in `./data` folder.

### Additional preprocessed data
Download the additional preprocessed data for ActorsHQ and 4D-DRESS dataset using this [drive link](https://drive.google.com/drive/folders/1kP4mUgUA7rP4wGekMs-jQkg7ETSndQ7O) and unpack it in `./data` folder.

### SMPL-X models
Download SMPL-X model and [VPoser](https://github.com/nghorbani/human_body_prior) checkpoint from [its official website](https://smpl-x.is.tue.mpg.de/). You need to download **SMPL-X v1.1 (NPZ+PKL, 830 MB)** and **VPoser v1.0 - CVPR'19 (2.5MB)**. Unzip the downloaded files and move each body models and VPoser checkpoint (`vposer_v1_0/snapshots/TR00_E096.pt`) to `./data/body_models`.

In the end your `./data` folder should look like this:
```
data
    |-- ActorsHQ
        |-- Actor01
        |-- Actor02
        |-- Actor03
        |-- Actor06
    |-- 4D-DRESS
        |-- 00170_Inner
        |-- 00185_Inner
        |-- 00190_Inner
        |-- 00191_Inner
    |-- body_models
        |-- smplx
            |-- SMPLX_NEUTRAL.npz
            |-- SMPLX_FEMALE.npz
            |-- SMPLX_MALE.npz
        |-- TR00_E096.pt
    |-- a1_s1
    |-- a2_s1
    |-- a3_s1
    |-- a6_s1
    |-- s170_t1
    |-- s185_t1
    |-- s190_t2
    |-- s191_t2
    |-- demo
```
### Demo Data

To help users run the demo without executing the full preprocessing and training pipeline, we provide the necessary intermediate results (e.g., tracked meshes, blend weights, Gaussian point clouds, shadow networks, etc.).  
You can download them from this [link](https://drive.google.com/drive/folders/1ZylC3f8b3wg6Ae4MkVeHFodJc0OVuJfu?usp=drive_link) and place them according to the directory structure described in the [issue](https://github.com/KAISTChangmin/MPMAvatar/issues/1)

&nbsp;
   
## Preprocessing

<pre><code> $ cd preprocess
 $ bash ./scripts/actorshq_a1.sh
 $ cd .. </pre></code>

## Training

<pre><code> $ bash ./scripts/appearance/actorshq_a1.sh
 $ bash ./scripts/physics/actorshq_a1.sh </pre></code>

## Simulation & Rendering

<pre><code> $ bash ./scripts/sim/actorshq_a1.sh </pre></code>

## Evaluation

<pre><code> $ bash ./scripts/eval/actorshq_a1.sh </pre></code>

## Run Demo

<pre><code> $ python run_demo.py --save_name chair_sand --output_dir ./output/demo </pre></code>
