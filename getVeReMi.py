# VeReMi Dataset Downloader
# This script downloads the VeReMi dataset from GitHub and extracts it.
import os, sys, json, time, tarfile, requests
from tqdm import tqdm

OWNER = "VeReMi-dataset"
REPO  = "VeReMi"
TAG   = "v1.0"        # 'latest' 가능
DEST  = os.path.join(os.getcwd(), "veremi_all")
OUT   = os.path.join(DEST, "extracted")
GH_TOKEN = os.environ.get("GITHUB_TOKEN")  # 선택

os.makedirs(DEST, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

sess = requests.Session()
sess.headers.update({"User-Agent": "v2x-downloader", "Accept": "application/vnd.github+json"})
if GH_TOKEN:
    sess.headers.update({"Authorization": f"Bearer {GH_TOKEN}"})

rel_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/" + ("latest" if TAG=="latest" else f"tags/{TAG}")
r = sess.get(rel_url, timeout=30)
r.raise_for_status()
rel = r.json()
assets = rel.get("assets", [])
if not assets:
    print("No assets found. Check tag/repo.")
    sys.exit(1)

def download_with_resume(url, path):
    # 이어받기: 존재 파일 크기 체크
    mode = "wb"
    resume_header = {}
    existing = os.path.exists(path)
    if existing:
        size = os.path.getsize(path)
        # Range 헤더 지원 시 이어받기
        resume_header["Range"] = f"bytes={size}-"
        mode = "ab"

    with sess.get(url, stream=True, headers=resume_header, timeout=60) as resp:
        if resp.status_code in (200, 206):
            total = int(resp.headers.get("Content-Length", 0))
            desc = os.path.basename(path)
            with open(path, mode) as f, tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            # 이어받기 미지원 시 전체 받기
            resp = sess.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            desc = os.path.basename(path)
            with open(path, "wb") as f, tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

# 다운로드
for a in assets:
    name = a["name"]
    url  = a["browser_download_url"]
    out  = os.path.join(DEST, name)
    if os.path.exists(out):
        print(f"Skip exists: {name}")
        continue
    print(f"Downloading: {name}")
    download_with_resume(url, out)

# 압축해제(.tgz)
for name in os.listdir(DEST):
    if name.endswith(".tgz"):
        path = os.path.join(DEST, name)
        print(f"Extracting: {name}")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(OUT)

print("\nDone!")
print("Downloaded to:", DEST)
print("Extracted to :", OUT)
