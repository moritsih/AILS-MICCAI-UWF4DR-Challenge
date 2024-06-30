from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import filecmp
from tqdm import tqdm
import base64

# Function to generate and save a 256-bit key in Base64 format if it does not exist
def generate_key_if_necessary():
    key_path = "aes256.key"
    if not os.path.exists(key_path):
        key = os.urandom(32)  # 256 bits
        with open(key_path, "wb") as key_file:
            key_file.write(base64.urlsafe_b64encode(key))
        print("Key generated and saved as 'aes256.key'.")
    else:
        print("Key already exists.")

# Function to load the key
def load_key():
    with open("aes256.key", "rb") as key_file:
        return base64.urlsafe_b64decode(key_file.read())

# Function to encrypt a file using AES-256
def encrypt_file(file_path, key):
    encrypted_file_path = file_path + ".enc"
    if os.path.exists(encrypted_file_path):
        print(f"Encrypted file '{encrypted_file_path}' already exists. Skipping encryption.")
        return encrypted_file_path

    with open(file_path, "rb") as file:
        data = file.read()

    # Add padding to the data
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()

    iv = os.urandom(16)  # Initialization vector for AES
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()

    with open(encrypted_file_path, "wb") as file:
        file.write(iv + encrypted)
    print(f"File '{file_path}' encrypted successfully.")
    return encrypted_file_path

# Function to decrypt a file using AES-256
def decrypt_file(file_path, key):
    with open(file_path, "rb") as file:
        iv = file.read(16)
        encrypted_data = file.read()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_data = unpadder.update(decrypted_padded) + unpadder.finalize()

    decrypted_file_path = file_path.replace(".enc", "")
    with open(decrypted_file_path, "wb") as file:
        file.write(decrypted_data)
    return decrypted_file_path

# Function to encrypt all files in the 'to_be_encrypted' folder
def encrypt_files_in_folder(folder_path, key):
    for root, dirs, files in os.walk(folder_path):
        #if no file found, create the folder_path and write nice message how to use the tool (using the absolute folder path)
        if not files:
            os.makedirs(folder_path, exist_ok=True)            
            print(f"No files found in '{folder_path}'. Please add files to encrypt in this folder.")
            print("You can encrypt files by running this script again, after that upload it to the gdrive and you can make the link to this file public so it can be downloaded by all team members easily.")
            return

        for file in tqdm(files, desc="Encrypting files"):
            file_path = os.path.join(root, file)
            # Skip already encrypted files
            if not file.endswith(".enc"):
                encrypted_file_path = encrypt_file(file_path, key)
                decrypted_file_path = decrypt_file(encrypted_file_path, key)
                if filecmp.cmp(file_path, decrypted_file_path, shallow=False):
                    print(f"Verification successful for '{file_path}'.")
                    os.remove(decrypted_file_path)
                else:
                    print(f"Verification failed for '{file_path}'.")

if __name__ == "__main__":
    # Generate and save the key (only if it does not already exist)
    generate_key_if_necessary()
    
    # Load the key
    key = load_key()
    
    # Folder containing files to be encrypted
    folder_to_encrypt = "tools/to_be_encrypted"
    
    # Encrypt each file in the folder
    encrypt_files_in_folder(folder_to_encrypt, key)
