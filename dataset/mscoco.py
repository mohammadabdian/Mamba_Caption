import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_size = None
        self.bytes_downloaded = 0

    def hook(self, block_num, block_size, total_size):
        if not self.file_size:
            self.file_size = total_size
        self.update(block_num * block_size - self.bytes_downloaded)
        self.bytes_downloaded = block_num * block_size


def download_file_with_progress(url, output_path):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}. Skipping download.")
        return

    print(f"Downloading: {url}")
    try:
        progress_bar = DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path))
        urllib.request.urlretrieve(url, output_path, reporthook=progress_bar.hook)
        progress_bar.close()
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # If download fails, remove the incomplete file
        if os.path.exists(output_path):
            os.remove(output_path)


def extract_zip(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}")
        return

    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress
            file_list = zip_ref.namelist()
            with tqdm(total=len(file_list), unit='file', desc=os.path.basename(zip_path)) as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_to)
                    pbar.update(1)
        print(f"Extracted to: {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is a bad zip file. Please re-download.")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    base_dir = "mscoco2014"  
    ensure_dir(base_dir)

    # Define files to download and their URLs for COCO 2014
    files_to_download = {
        "train_images": {
            "url": "http://images.cocodataset.org/zips/train2014.zip",
            "zip_path": os.path.join(base_dir, "train2014.zip"),
            "extract_to": base_dir,
            "extract_folder_name": "train2014"  # Expected folder after extraction
        },
        "val_images": {
            "url": "http://images.cocodataset.org/zips/val2014.zip",
            "zip_path": os.path.join(base_dir, "val2014.zip"),
            "extract_to": base_dir,
            "extract_folder_name": "val2014"  # Expected folder after extraction
        },
        "annotations": {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "zip_path": os.path.join(base_dir, "annotations_trainval2014.zip"),
            "extract_to": base_dir,
            "extract_folder_name": "annotations"  # Expected folder after extraction
        },
    }

    for name, info in files_to_download.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        
        # Download
        download_file_with_progress(info["url"], info["zip_path"])

        # Extract if not already extracted
        extracted_path = os.path.join(info["extract_to"], info["extract_folder_name"])
        if not os.path.exists(extracted_path):
            extract_zip(info["zip_path"], info["extract_to"])
        else:
            print(f"Already extracted: {info['extract_folder_name']}")

    print("\n" + "="*60)
    print("MS COCO 2014 download and extraction complete.")
    print(f"Data saved in: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()