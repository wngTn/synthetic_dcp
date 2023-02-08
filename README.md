# Point-Set Alignment Using Weak Labels #

With the advent of recent technologies, multi-view RGB-D recordings have become the prevalent way of data acquisition in the operating room (OR).
The significant domain gap between standard data and OR data requires methods that are capable of effectively generalizing to this unique and challenging data domain.
Therefore, previous works have established methods to leverage 3D information to detect faces in an OR multi-view RGB-D setting. 
These methods rely on point set registrations; however, real-world 3D point clouds are often noisy and incomplete, which may yield erroneous alignments using existing point set registration methods. In this project, we aim to address this issue by adapting a deep learning-based point-set registration method to achieve more robust rigid transformations on real-world data.
We perform quantitative as well as qualitative evaluations of our proposed method and also give an outlook for future improvements.

For details see the full [project proposal](proposal/proposal.pdf).

This project was conducted as part of the 2022/23 Machine Learning for 3D Geometry course (IN2392) at the [Technical University of Munich](https://www.cit.tum.de/cit/startseite/).

## Problem ##

To following shows the real-world point cloud data of an OR (see "Operation Room Data" below for details on this data).
In the scene, a DL-based detector has identified a person and placed a [SMPL](https://smpl.is.tue.mpg.de/) mesh at the estimated position.
As one can see the estimation of the head is not very good and needs refinement to be usable for face detection.
In this project, different algorithms are compared in terms of their performance in computing a rotation and translation matrix that better aligns the head with the person in the OR.

<figure class="video_container">
 <video controls="false">
 <source src="assets/problem_vis.mp4" type="video/mp4" width="400">
 </video>
</figure>

## Synthetic Dataset ##

[DCP](https://arxiv.org/pdf/1905.03304.pdf) is originally trained on the [ModelNet40 dataset](https://modelnet.cs.princeton.edu/), which deviates too much from our medical setting.
Therefore, to train the DCP architecture, we create a synthetic dataset to imitate the real OR data as closely as possible.

The parameters for synthetically created SMPL poses can be found [here](https://nextcloud.in.tum.de/index.php/s/H9W8rAAoHiXHjfz) (128MB).
You may place it under `data/smpl_training_poses.pkl`

It has over 90k different poses sampled from the [HumanAct12 dataset](https://ericguo5513.github.io/action-to-motion/).

The following GIF visualizes 100 of these poses ([code used to create it](scripts/peak_synthetic_smpl.py))

<img src="assets/smpl_poses.gif" width="400"/>

We augment the SMPL meshes and accessories to further mimic the real data.
The mesh is cropped around the head as this is the part we want to do the alignment on.
We sample points on the meshes, add some noise and use the point cloud as input to the model.

<img src="assets/data_augmentation_vis.gif" width="400"/>


## Results ##

In the following, the red mesh is the point cloud we are trying to align to the recorded 3D scene.
The head is rendered into the scene as predicted by coarse full-body detection.


<img src="assets/OR_unaligned.gif" width="400"/>

After rigid alignment with DCP

<img src="assets/OR_aligned.gif" width="400"/>


For more quantitative as well as qualitative results see the full [project report](project_report.pdf) or the [supplementary materials](supplementary_materials.pdf).

We evaluated the performance of DCP and FilterReg both on real-world and synthetic data.

### Synthetic Data
- DCP-v2 outperformed DCP-v1
- DCP trained on our synthetic data outperformed DCP trained on ModelNet40
- synthetically trained DCP-v2 yields about equal performance as FilterReg

### Real-world Data
- FilterReg performs a lot better than DCP
- DCP doesn't seem to work well with real-world imperfections (other architectures like [PRNet](https://arxiv.org/abs/1910.12240) might be promising for further research)


## Dependencies 

```
pip install -r requirements.txt
```

You may want to use a virtual environment + dependency manager (e.g use Conda)

To install PyTorch3D see `https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md`

## SMPL Models ##

The SMPL models can be downloaded from: [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)

They are to be placed at `./data/smpl_models/`

We did not create them nor hold intellectual property on them.
We use them under the provided license for non-commercial scientific purposes as granted under:

- [https://smpl.is.tue.mpg.de/modellicense.html](https://smpl.is.tue.mpg.de/modellicense.html) 
- [https://smpl.is.tue.mpg.de/bodylicense.html](https://smpl.is.tue.mpg.de/bodylicense.html)

Therefore, we adhere to their demand not to publish/distribute their code/model.
To execute the SMPL model, one could be interested in the code provided in [EasyMocap](https://github.com/zju3dv/EasyMocap/tree/master/easymocap/smplmodel). Notice the emptiness of `./lib/smplmodel/` in this project.

Please review and comply with the license requirements of the SMPL authors as stated on their linked webpage before using any code.


## Operation Room Data ##

As the data captured in the OR is the proprietary property of the [chair of
Computer Aided Medical Procedures (CAMP)](https://www.cs.cit.tum.de/camp/start/) at TUM, we are not able to publish it here.

If you are in legal possession you may download and save the data under `./data/trial`.
The 2D face bounding box annotations are available at `./data/gt` as these are part of our contribution.

Without this data, you are not able to test the synthetically trained model on real-world data.
You can run `python ./demo/visualize_pointcloud.py` to see one frame as a demo.
To get a grasp of what the data would look like.
The rest of the project is working as is.

## Contributors

- [Johannes Volk](linkedin.com/in/jovo/)
- Tony Wang
- Yushan Zheng
- Yutong Hou

## Further Work

[PRNet](https://arxiv.org/pdf/1910.12240.pdf) is a Partial-to-Partial Registration DL-based approach.
We deemed it might be suitable for our application,
as our real-world measurements are also only partial but we were not able to reach reasonable results with it.

The [code of PRnet](https://github.com/WangYueFt/prnet) is included in this project and may be subject to further experiments.

As the detection algorithm that provides the first estimate on an alignment is also DL-based it should be possible to create a pipeline that is end-to-end trainable.

## References
This project is based on the following work,
we hereby acknowledge the intellectual property used in this project.
We tagged the parts of the code where their work was used directly or in refactored form.
For further references see the full [project report](project_report.pdf).
