import os
import gdown
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import shutil
import zipfile
import tarfile
from tqdm import tqdm
import base64
from enum import Enum

class FileInfo:
    def __init__(self, url, filename, uncompress=True):
        self.url = url
        self.filename = filename
        self.uncompress = uncompress

class FileCategory:
    def __init__(self, target_dir, files):
        self.target_dir = target_dir
        self.files = files

class FileDownloaderDecryptor:
    class FileCategoryEnum(Enum):
        DRID_DATA = FileCategory(
            target_dir="data/external",
            files=[
                FileInfo(
                    url="https://drive.google.com/file/d/1jm48RSCctyxtEkppS45Znh0wtdf9patA/view?usp=drive_link",
                    filename="DeepDRiD.zip.enc"
                )
            ]
        )
        CHALLENGE_DATA = FileCategory(
            target_dir="data/raw",
            files=[
                FileInfo(
                    url="https://drive.google.com/file/d/1K8xwscXQQo0KXEzFaC2wybgD-UYNXvfc/view?usp=sharing",
                    filename="UWF4DRChallengeData.zip.enc"
                )
            ]
        )
        AUTOMORPH_CHECKPOINTS = FileCategory(
            target_dir="models/AutoMorph",
            files=[
                FileInfo(
                    url="https://drive.google.com/file/d/1t7Dt8ViDAZ4fLYsFBWmU12Smv10hJyLo/view?usp=drive_link",
                    filename="automorph_best_loss_checkpoint.pth",
                    uncompress=False
                )
            ]
        )
        INTERFERNCE_MODEL = FileCategory(
            target_dir="models/submission_eval_model_weights",
            files=[
                FileInfo(
                    url="https://drive.google.com/file/d/1X5gekt4_BbIZLoj2fVMHwjWRRGWotzm1/view?usp=drive_link",
                    filename="model.pth",
                    uncompress=False
                )
            ]
        )

    def __init__(self, key_path="aes256.key", download_directory="data/downloads"):
        self.key_path = key_path
        self.download_directory = download_directory
        self.key = self.load_key()
        os.makedirs(download_directory, exist_ok=True)

        # Initialize target directories for each category
        for category in self.FileCategoryEnum:
            os.makedirs(category.value.target_dir, exist_ok=True)

    def load_key(self):
        with open(self.key_path, "rb") as key_file:
            return base64.urlsafe_b64decode(key_file.read())

    def download_file(self, url, filename):
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output = os.path.join(self.download_directory, filename)
        if os.path.exists(output):
            print(f"File '{output}' already exists. Skipping download.")
            return output
        print(f"Downloading from '{download_url}' to '{output}'")
        gdown.download(download_url, output, quiet=False)
        print(f"Downloaded to '{output}'")
        return output

    def decrypt_file(self, file_path):
        if not file_path.endswith(".enc"):
            print(f"File '{file_path}' is not encrypted. Skipping decryption.")
            return file_path

        with open(file_path, "rb") as file:
            iv = file.read(16)
            encrypted_data = file.read()

        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

        decrypted_file_path = file_path.replace(".enc", "")
        with open(decrypted_file_path, "wb") as file:
            file.write(decrypted_data)
        print(f"Decrypted from '{file_path}' to '{decrypted_file_path}'")
        return decrypted_file_path

    def extract_if_compressed(self, file_path, dest_folder, uncompress=True):
        if not uncompress:
            print(f"Skipping extraction for '{file_path}'")
            target_path = os.path.join(dest_folder, os.path.basename(file_path))
            if os.path.abspath(file_path) == os.path.abspath(target_path):
                print(f"Source and target paths are the same ('{file_path}'). Skipping move.")
                return
            if os.path.exists(target_path):
                print(f"File '{target_path}' already exists. Skipping move.")
            else:
                print(f"Moving '{file_path}' to '{dest_folder}'")
                shutil.move(file_path, dest_folder)
                print(f"Moved to '{target_path}'")
            return

        try:
            if zipfile.is_zipfile(file_path):
                print(f"Extracting '{file_path}' to '{dest_folder}'")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_folder)
            elif tarfile.is_tarfile(file_path):
                print(f"Extracting '{file_path}' to '{dest_folder}'")
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(dest_folder)
            else:
                target_path = os.path.join(dest_folder, os.path.basename(file_path))
                if os.path.abspath(file_path) == os.path.abspath(target_path):
                    print(f"Source and target paths are the same ('{file_path}'). Skipping move.")
                    return
                if os.path.exists(target_path):
                    print(f"File '{target_path}' already exists. Skipping move.")
                else:
                    print(f"Moving '{file_path}' to '{dest_folder}'")
                    shutil.move(file_path, dest_folder)
                    print(f"Moved to '{target_path}'")
        except Exception as e:
            print(f"An error occurred while extracting '{file_path}': {e}")
            raise

    def process_files(self):
        for category in self.FileCategoryEnum:
            dest_folder = category.value.target_dir
            for file_info in category.value.files:
                file_path = self.download_file(file_info.url, file_info.filename)
                decrypted_file_path = self.decrypt_file(file_path)
                self.extract_if_compressed(decrypted_file_path, dest_folder, file_info.uncompress)

if __name__ == "__main__":
    # Initialize the downloader/decryptor
    downloader_decryptor = FileDownloaderDecryptor()

    # Process the files
    downloader_decryptor.process_files()
