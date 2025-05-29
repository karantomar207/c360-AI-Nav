import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

def download_from_s3(s3_key, local_path):
    bucket = os.getenv("FAISS_DB_BUCKET")
    print("///////////////", bucket, s3_key, local_path)
    if not os.path.exists(local_path):
        print(f"⬇️ Downloading {s3_key} from S3...")
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"✅ Downloaded to {local_path}")
    else:
        print(f"✔️ Found local copy: {local_path}")
