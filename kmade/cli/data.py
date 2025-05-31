import os
import requests


def download_gw150914_data(save_dir="kmade/data"):
    base_url = "https://zenodo.org/records/6513631/files/"
    file_url = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
    url = base_url + file_url
    filename = "GW150914_posterior.h5"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    print(f"Downloading into {file_path} ...")
    r = requests.get(url)
    r.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(r.content)
    print("Done！")


if __name__ == "__main__":
    download_gw150914_data()
