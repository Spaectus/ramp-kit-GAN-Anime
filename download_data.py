import urllib.request
from pathlib import Path
import zipfile

tmp_filename = "archive.zip"
LOCAL_DATA = Path(__file__).parent / "data"
LOCAL_DATA.mkdir(exist_ok=True)

urllib.request.urlretrieve(
    "https://drive.google.com/uc?export=download&id=1R-8ieidrUVuqNHgD-Hyw_od5pHbjgbOO&confirm=t&uuid=25b26a1a-8ed1-4eb1-a2a4-343d0e48a3ee&at=ALgDtswyiievmv0tddNky9ePQCt9:1676728294393", tmp_filename)
with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
    zip_ref.extractall(LOCAL_DATA)
Path(tmp_filename).unlink()
