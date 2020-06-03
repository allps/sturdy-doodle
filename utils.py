import uuid
import boto3
import os
from boto3.s3.transfer import S3Transfer
from pathlib import Path
from dotenv import load_dotenv

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


def get_unique_filename():
    return str(uuid.uuid4())


def allowed_file_extensions(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_s3(file_path, file_name):
    if len(os.getenv('AWS_ACCESS_KEY_ID')) < 1:
        return file_path

    credentials = {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_ACCESS_KEY_SECRET')
    }

    client = boto3.client('s3', region_name='eu-central-1', **credentials)

    transfer = S3Transfer(client)
    print(file_name)
    transfer.upload_file(file_path, os.getenv('S3_BUCKET'), file_name, extra_args={'ACL': 'public-read'})

    file_url = '%s/%s/%s' % (client.meta.endpoint_url, os.getenv('S3_BUCKET'), file_name)
    print(file_url)
    return file_url
