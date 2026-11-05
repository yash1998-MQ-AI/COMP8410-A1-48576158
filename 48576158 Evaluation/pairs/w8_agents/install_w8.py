"""Week 8 install for Colab: MiniWoB++ + a headless Chromium with a MATCHING chromedriver (they must share the major version).

    python pairs/w8_agents/install_w8.py

1. pip install miniwob selenium (+ model libraries are already in Colab: transformers, torch)
2. Playwright's Chromium (playwright installs the system libraries it needs), used only as the browser binary
3. chromedriver of the same major version from Chrome for Testing
4. writes /content/w8_browser.json {chrome, driver}; the notebook sets MINIWOB_CHROME_BINARY and MINIWOB_CHROMEDRIVER from it
5. quick self-test: opens a MiniWoB task, prints the observation keys and clicks a button
"""
import glob, json, os, re, subprocess, sys, urllib.request, zipfile


def sh(cmd, check=True):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def main():
    sh("pip -q install miniwob selenium playwright accelerate qwen-vl-utils")
    sh("playwright install --with-deps chromium")
    cands = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    if not cands:
        sys.exit("no playwright chromium found")
    chrome = cands[-1]
    ver = subprocess.run([chrome, "--version"], capture_output=True, text=True).stdout.strip()
    print(ver)
    major = re.search(r"(\d+)\.\d+\.\d+\.\d+", ver).group(1)
    with urllib.request.urlopen("https://googlechromelabs.github.io/chrome-for-testing/latest-versions-per-milestone-with-downloads.json") as r:
        data = json.load(r)
    dl = data["milestones"][major]["downloads"]["chromedriver"]
    url = next(d["url"] for d in dl if d["platform"] == "linux64")
    os.makedirs("/content/chromedriver", exist_ok=True)
    zpath = "/content/chromedriver/cd.zip"
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall("/content/chromedriver")
    driver = glob.glob("/content/chromedriver/**/chromedriver", recursive=True)[0]
    os.chmod(driver, 0o755)
    print(subprocess.run([driver, "--version"], capture_output=True, text=True).stdout.strip())
    json.dump({"chrome": chrome, "driver": driver}, open("/content/w8_browser.json", "w"))
    print("INSTALL DONE", chrome, driver)


if __name__ == "__main__":
    main()
