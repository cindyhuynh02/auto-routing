import pandas as pd
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import gspread
import warnings
from googleapiclient.discovery import build
from google.oauth2 import service_account
import pandas_gbq as pdgbq
from google.cloud import bigquery, storage
warnings.filterwarnings("ignore")

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/bigquery",
]


class google_bigquery_custom:
    def __init__(self, service_account_info):
        self.creds, self.client = self.get_auth_connection(service_account_info)

    def get_auth_connection(self, service_account_info):
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES
        )
        client = bigquery.Client(
            project="vn-bi-product",
            credentials=creds,
            _http=None,
            client_info=None,
            client_options=None,
        )
        return creds, client

    def query_data(self, query_body):
        # Perform a query.
        df = pd.read_gbq(
            query=query_body, project_id="vn-bi-product", credentials=self.creds
        )
        return df

    def insert_data(self, data, dest_table, type_insert):
        data.to_gbq(
            destination_table=dest_table,
            project_id="vn-bi-product",
            if_exists=type_insert,
            progress_bar=True,
            credentials=self.creds,
        )
        return None
    
class google_cloud_custom:
    def __init__(self, info):
        self.creds, self.drive_service, self.client = self.get_auth_connection(info)

    def get_auth_connection(self, info):
        creds = service_account.Credentials.from_service_account_info(info)
        drive_service = build('drive', 'v3', credentials=creds)
        client = storage.Client(
            project="vn-bi-product",
            credentials=creds,
            _http=None,
            client_info=None,
            client_options=None,
        )        
        return creds, drive_service, client

    def upload_gcs(self, bucket_name, folder_lv1, folder_lv2, data, f_name):
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(f"{folder_lv1}/{folder_lv2}/{f_name}")
        blob.upload_from_string(data.to_csv(index=False, encoding='utf-8'), content_type='text/csv')
        print(f"Done upload file {folder_lv1}/{folder_lv2}/{f_name}")

class google_drive_custom:
    def __init__(self, service_account_info):
        # Initialize the connection to both Google Sheets and Google Drive
        self.gsheet_service, self.drive_service = self.connect_drive(service_account_info)

    def connect_drive(self, bi_key):
        """Set up Google Sheets and Google Drive connection."""
        # Authenticate using the provided service account credentials
        gauth = GoogleAuth()
        gauth.credentials = service_account.Credentials.from_service_account_info(bi_key, scopes=SCOPES)

        # Google Drive service
        drive_service = GoogleDrive(gauth)
        
        # Google Sheets service
        gsheet_service = gspread.authorize(gauth.credentials)
        
        return gsheet_service, drive_service

    