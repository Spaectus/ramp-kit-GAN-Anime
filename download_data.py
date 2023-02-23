import urllib.request
from pathlib import Path
import zipfile

tmp_filename = "archive.zip"
LOCAL_DATA = Path(__file__).parent

if 0:
    url = "https://drive.rezel.net/s/kfdPRTr8xdBmGkH/download/data.zip"
else:
    url = "https://drive.google.com/uc?export=download&id=1xeRgfKecCg9sPCmTFUhvqe9ouG66ylsb&confirm=t&uuid=84c9041f-3b66-4e2f-97d6-732e5186b5bb&at=ALgDtsw9NtCf4Of4ZlAcVYhOXnVQ:1677075866291"

urllib.request.urlretrieve(url, tmp_filename)
with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
    zip_ref.extractall(LOCAL_DATA)
Path(tmp_filename).unlink()
