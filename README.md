# 3D Vision — Study Exercises (based on mint-lab/3dv_tutorial)

This repository contains my worked exercises, notes, and code written while going through **[mint-lab/3dv_tutorial](https://github.com/mint-lab/3dv_tutorial)** — *"An Invitation to 3D Vision"* by Sunglok Choi et al. It is a personal study log from learning 3D computer vision, not a standalone portfolio project.

## What's here

- **[`3d_processing_scripts/`](3d_processing_scripts)** — my solutions to the mint-lab tutorial exercises (camera calibration, epipolar geometry, feature matching/tracking, fundamental matrix estimation, bundle adjustment, structure-from-motion, distortion correction, etc.). The problem set and structure closely follow the original tutorial; see the source repo for the accompanying lecture slides and theory.
- **[`cv_engine/`](cv_engine)** — my own reusable engine written while working through the tutorial: calibration, feature detection (including a DNN-based detector variant), geometry utilities, and optimization/solver code. This part is my own design and implementation, built on top of what the tutorial covers.
- **[`data/`](data)** — sample images, video, and point-cloud data used by the scripts above.

## License

[Beerware](http://en.wikipedia.org/wiki/Beerware), inherited from the original tutorial — see [`LICENSE`](LICENSE).

## Credit

All tutorial content, lecture slides, and the original example structure are © [Sunglok Choi](https://mint-lab.github.io/sunglok/) and [JunHyeok Choi](https://github.com/cjh1995-ros) (mint-lab). Go to the [original repository](https://github.com/mint-lab/3dv_tutorial) for the full tutorial, slides, and further reading:

* Lecture slides: [Introduction](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/01_introduction.pdf) · [Single-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/02_single-view_geometry.pdf) · [Two-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/03_two-view_geometry.pdf) · [Solving Problems](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/04_solving_problems.pdf) · [Finding Correspondence](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/05_correspondence.pdf) · [Multiple-view Geometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/06_multi-view_geometry.pdf) · [Visual SLAM and Odometry](https://github.com/mint-lab/3dv_tutorial/blob/master/slides/07_visual_slam.pdf)
* Additional acknowledgements (dataset/media sources used by the original tutorial) are listed in the [original README](https://github.com/mint-lab/3dv_tutorial#acknowledgement).
