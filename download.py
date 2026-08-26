URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A/view?usp=share_link",
    "svox": "https://drive.google.com/file/d/16iuk8voW65GaywNUQlWAbDt6HZzAJ_t9/view?usp=drive_link",
}

MODELS_URL = "https://drive.google.com/drive/folders/1M43jAQBMmZ2_YUI0v2_Ww044E5oNWk-W?usp=sharing"


import os
import gdown
import shutil
import tempfile

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)

# trained regressors (model JSON files) used by VPR-Adaptive-ReRanking/validation/
if MODELS_URL:
    print("Downloading trained regressors (model JSON files for validation)")
    OUT_DIR = "VPR-Adaptive-ReRanking/validation"
    # subfolders already present in VPR-Adaptive-ReRanking/validation/
    SUBDIRS = ("logistic_hard", "logistic_help", "logistic_cost_sensitive",
               "su", "su_inliers", "sequential")
    with tempfile.TemporaryDirectory() as tmp:
        zip_filepath = os.path.join(tmp, "validation_models.zip")
        gdown.download(MODELS_URL, zip_filepath, fuzzy=True)
        shutil.unpack_archive(zip_filepath, extract_dir=tmp)
        # Drive may wrap the files in an extra folder (e.g. validation_models/su/...);
        # regardless of nesting, move every .json into OUT_DIR/<its parent folder name>/
        n = 0
        for root, _, files in os.walk(tmp):
            parent = os.path.basename(root)
            if parent not in SUBDIRS:
                continue
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                dest_dir = os.path.join(OUT_DIR, parent)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(os.path.join(root, fname), os.path.join(dest_dir, fname))
                n += 1
        print(f"Installed {n} JSON files into {OUT_DIR}/")
