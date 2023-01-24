# 3D Head Alignment Using Deep Closest Point #


With the advent of recent technologies, multi-view RGB-D recordings have become the
prevalent way of data acquisition in operating rooms (OR). [Previous works](TODO add Reference) established
the benefits of using 3D information to detect and anonymize faces of 2D images in
such multi-view settings. However, real-world 3D data often suffers from noisy and
incomplete point clouds, which yield erroneous alignments using probabilistic point set
alignment methods like [coherent point drift (CPD)](https://arxiv.org/pdf/0905.2635.pdf) and its variants.
In this project, we address this issue by creating and fine-tuning the deep learning-based point set registration method [Deep Closest Point (DCP)](https://arxiv.org/pdf/1905.03304.pdf) to achieve more robust rigid transformations on noisy and incomplete point clouds from real-world data.

For details see the full [project proposal](proposal/proposal.pdf).

This project was conducted as part of the 2022/23 Machine Learning for 3D Geometry course (IN2392) at the Technical University of Munich.


<!-- It would be nice to have one image here of the OR scene / our problem or smth along the lines  -->

## Synthetic Dataset ##

[DCP](https://arxiv.org/pdf/1905.03304.pdf) is originally trained on the [ModelNet40 dataset](https://modelnet.cs.princeton.edu/).

For our application, this dataset doesn't suffice, as our observed point clouds/targets for alignment (human head) have very different geometry.
Therefore, to train the DCP architecture we create a synthetic dataset to mimic the real data of the OR setting as closely as possible.

The parameters for synthetically created SMPL poses can be found [here](https://nextcloud.in.tum.de/index.php/s/H9W8rAAoHiXHjfz) (128MB).

It has over 90k different poses sampled from the [HumanAct12 dataset](https://ericguo5513.github.io/action-to-motion/).
The following GIF visualizes 100 of these poses.
 <!-- TODO decide if we want to reference it here ([code used to create it](scripts/peak_synthetic_smpl.py)) -->


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


For details see the full [project report](TODO link) and the [presentation](TODO link)


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

You may want to use a venv/dependency manager (e.g Conda)


## SMPL Models ##

`python scripts/download_all.py # downloads the SMPL models`

The SMPL models are taken from: [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de).

We did not create them nor hold intellectual property on these.
We use these under the provided license for non-commercial scientific purposes as granted under:

- [https://smpl.is.tue.mpg.de/modellicense.html](https://smpl.is.tue.mpg.de/modellicense.html) 
- [https://smpl.is.tue.mpg.de/bodylicense.html](https://smpl.is.tue.mpg.de/bodylicense.html)

Please review and comply with their license requirements as stated on the linked webpage before using our code.

## Operation Room Data ##

As the data captured in the OR is the proprietary property of the [chair of
Computer Aided Medical Procedures (CAMP)](https://www.cs.cit.tum.de/camp/start/) at TUM, we are not able to publish it here.

If you are in legal possession you may download and save the trial data under `./data/trial`.

Without this data, you are not able to test the synthetically trained model on real-world data.
The rest of the project is working as is.

<!-- ## Debugging ##

Use debug.py and the key maps are:
- A for advance
- S for going back
- Q for the next person
- D for closing the whole debugger -->

## Contributors

In no particular order

- Tony Wang
- Yushan Zheng
- Yutong Hou
- [Johannes Volk](linkedin.com/in/jovo/)


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
