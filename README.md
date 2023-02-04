# Point-Set Alignment Using Weak Labels #


With the advent of recent technologies, multi-view RGB-D recordings have become the
prevalent way of data acquisition in operating rooms (OR).
Previous works established the benefits of using 3D information to detect and anonymize faces of 2D images in
such multi-view settings. However, real-world 3D data often suffers from noisy and
incomplete point clouds, which yield erroneous alignments using probabilistic point set
alignment methods like [coherent point drift (CPD)](https://arxiv.org/pdf/0905.2635.pdf) and its variants.
In this project, we address this issue by creating and fine-tuning the deep learning-based point set registration method [Deep Closest Point (DCP)](https://arxiv.org/pdf/1905.03304.pdf) to achieve more robust rigid transformations on noisy and incomplete point clouds from real-world data.

For details see the full [project proposal](proposal/proposal.pdf).

This project was conducted as part of the 2022/23 Machine Learning for 3D Geometry course (IN2392) at the Technical University of Munich.

## Problem ##

To following shows the real-world point cloud data of an OR (see "Operation Room Data" below for details on this data).
In the scene, a DL-based detector has identified a person and placed a [SMPL](https://smpl.is.tue.mpg.de/) mesh at the estimated position.
As one can see the estimation of the head is not very good and needs refinement to be usable for anonymization.
In this project, different algorithms are compared in terms of their performance in computing a rotation and translation matrix that better aligns the head with the person in the OR.

<figure class="video_container">
 <video controls="false">
 <source src="assets/problem_vis.mp4" type="video/mp4" width="400">
 </video>
</figure>

## Synthetic Dataset ##

[DCP](https://arxiv.org/pdf/1905.03304.pdf) is originally trained on the [ModelNet40 dataset](https://modelnet.cs.princeton.edu/).

For our application, this dataset doesn't suffice, as our observed point clouds/targets for alignment (i.e. human heads) have very different geometry.
Therefore, to train the DCP architecture we create a synthetic dataset to imitate the real data of the OR setting as closely as possible.

The parameters for synthetically created SMPL poses can be found [here](https://nextcloud.in.tum.de/index.php/s/H9W8rAAoHiXHjfz) (128MB).
<!-- TODO maybe make download automatic -->

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


For details see the full [project report](TODO link) and the [presentation slides](TODO link).


<!-- ## File Structure ##

```
.
├── README.md
├── data
│   ├── mesh_files
│   ├── smpl_models 
│   └── trial
├── scripts
│   ├── ...
├── data.py
├── demo
├── main.py
├── model.py
├── pretrained
├── proposal
└── util.py
``` -->

## Dependencies 

```
pip install -r requirements.txt
```

You may want to use a virtual environment + dependency manager (e.g use Conda)

To install PyTorch3d see `https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md`

## SMPL Models ##

The SMPL models can be downloaded from: [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de).

We did not create them nor hold intellectual property on them.
We use them under the provided license for non-commercial scientific purposes as granted under:

- [https://smpl.is.tue.mpg.de/modellicense.html](https://smpl.is.tue.mpg.de/modellicense.html) 
- [https://smpl.is.tue.mpg.de/bodylicense.html](https://smpl.is.tue.mpg.de/bodylicense.html)

Therefore, we adhere to their demand not to publish/distribute their code/model.
To execute the SMPL model, one could be interested in the code provided in [EasyMocap](https://github.com/zju3dv/EasyMocap/tree/master/easymocap/smplmodel) Notice the emptiness of `./lib/smplmodel` in this project.

Please review and comply with the license requirements of the SMPL authors as stated on their linked webpage before using any code.


## Operation Room Data ##

As the data captured in the OR is the proprietary property of the [chair of
Computer Aided Medical Procedures (CAMP)](https://www.cs.cit.tum.de/camp/start/) at TUM, we are not able to publish it here.

If you are in legal possession you may download and save the trial data under `./data/trial`.
The 2D face bounding box annotations are at `./data/gt` as this is part of our contribution.

Without this data, you are not able to test the synthetically trained model on real-world data.
You can run `python ./demo/visualize_pointcloud.py` to see one frame as a demo.
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

As the detection algorithm that provides the first guess on an alignment is also DL-based it should be possible to create a pipeline that is end-to-end trainable.

## References
This project is based on the following work,
we hereby acknowledge the intellectual property used in this project.
We tagged the parts of the code where their work was used directly or in refactored form.


	@article{SMPL:2015,
      author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
      title = {{SMPL}: A Skinned Multi-Person Linear Model},
      journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
      month = oct,
      number = {6},
      pages = {248:1--248:16},
      publisher = {ACM},
      volume = {34},
      year = {2015}
    }

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

<!-- TODO add the rest (if there is more) -->
