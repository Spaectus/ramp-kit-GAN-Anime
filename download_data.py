import urllib.request
from pathlib import Path
import zipfile

tmp_filename = "archive.zip"
LOCAL_DATA = Path(__file__).parent
LOCAL_DATA.mkdir(exist_ok=True)

urllib.request.urlretrieve(
    "https://drive.rezel.net/s/kfdPRTr8xdBmGkH/download/data.zip", tmp_filename)
with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
    zip_ref.extractall(LOCAL_DATA)
Path(tmp_filename).unlink()
