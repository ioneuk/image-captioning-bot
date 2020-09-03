import os

from models.utils import create_input_files

if __name__ == '__main__':
    dataset = os.environ['DATASET_NAME']
    captions_and_metadata_json_path = os.environ['CAPTIONS_AND_METADATA_JSON_PATH']
    image_folder = os.environ['IMAGE_FOLDER']
    output_folder = os.environ['PROCESSED_DATA_FOLDER']
    create_input_files(dataset=dataset,
                       captions_and_metadata_json_path=captions_and_metadata_json_path,
                       image_folder=image_folder,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=output_folder,
                       max_len=50)
