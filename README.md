# ML 3D Group Project

## Downloading SMPL Models ##

`python scripts/download_all.py # downloads the SMPL models`

The SMPL models are taken from: [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de).

We did not create them, do not prosecute us.

Download the trial data under this location: `./data/trial`

## Debugging ##

Use debug.py and the key maps are:
- A for advance
- S for going bac
- Q for the next person
- D for closing the whole debugger

## Synthetic SMPL Poses ##

The parameters for synthetically created SMPL poses can be found [here](https://nextcloud.in.tum.de/index.php/s/H9W8rAAoHiXHjfz) (128MB).

It has over 90k different poses sampled from the HumanAct12 dataset.

![SMPL POses](assets/smpl_poses.gif)
This GIF visualizes 100 poses; it was created with `scripts/peak_synthetic_smpl.py`


## File Structure ##

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
```

## Prerequisites 


```
pip install -r requirements.txt
```

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

## License
MIT License
