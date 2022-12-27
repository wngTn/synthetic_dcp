import gdown


SMPL_MODEL_PATH = "data/smpl_models/smpl"

try:
    # smpl model
    gdown.download_folder(
        url="https://drive.google.com/drive/u/0/folders/1gFdZC4quxsAzqGWR-aY7NFEWC_FeFHVy",
        quiet=False,
        output=SMPL_MODEL_PATH,
    )
except IOError:
    print("There has been an error trying to install the SMPL Models")
