import os
import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd

def download_and_extract_zip(url, output_folder, file_name):
    file_path = os.path.join(output_folder, file_name)

    # Create the 'data' folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download the ZIP file
    response = requests.get(url)
    
    # Extract the ZIP contents
    with ZipFile(BytesIO(response.content), 'r') as zip_ref:
        # Extract only 'train.csv' and 'test.csv' to the 'data' folder
        for member in zip_ref.namelist():
            if member.endswith(('train.csv', 'test.csv')):
                zip_ref.extract(member, output_folder)

    print(f"{file_name} has been downloaded and extracted to: {output_folder}")

def main():
    # Define URLs for training and testing datasets
    train_url = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip"
    test_url = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip"

    # Define folder names and file names
    output_folder = "data"
    train_file_name = "train.csv"
    test_file_name = "test.csv"

    # Download and extract the training dataset
    download_and_extract_zip(train_url, output_folder, train_file_name)

    # Download and extract the testing dataset
    download_and_extract_zip(test_url, output_folder, test_file_name)

    # Rename the extracted files to be directly in the 'data' folder
    os.rename(os.path.join(output_folder, 'final_project_train_dataset', 'train.csv'), os.path.join(output_folder, 'train.csv'))
    os.rename(os.path.join(output_folder, 'final_project_test_dataset', 'test.csv'), os.path.join(output_folder, 'test.csv'))

    # Remove the empty directories
    os.rmdir(os.path.join(output_folder, 'final_project_train_dataset'))
    os.rmdir(os.path.join(output_folder, 'final_project_test_dataset'))

    print("CSV files 'train.csv' and 'test.csv' moved to 'data' folder.")

if __name__ == "__main__":
    main()
