import os
import sys
import time
import uuid
import json
import hashlib
import logging
import warnings
import asyncio
import pickle
import torch
import numpy as np
from tenacity import *
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from contextlib import contextmanager
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import pandas_gbq
import unicodedata
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import multiprocessing
from google.oauth2 import service_account
from google.cloud import storage, bigquery
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    OptimizersConfigDiff,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    SearchParams,
    MatchText
)
import pytz
from model.AutoRouting import *
from tabulate import tabulate
import requests
import httplib2
from httplib2 import Http
from datetime import *
import aiohttp
import concurrent.futures
import io
from pydantic import BaseModel
from typing import Optional
from math import ceil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Cấu hình pandas
pd.set_option("display.max_rows", 10000)
pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_colwidth", 10000)

# Ignore UserWarnings
warnings.simplefilter("ignore", UserWarning)

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
dotenv_path = ROOT_PATH / 'config/.env'
load_dotenv(dotenv_path)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")