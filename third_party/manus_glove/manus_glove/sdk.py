import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)


_SDK_ZIP_URL = "https://static.manus-meta.com/resources/manus_core_3/sdk/MANUS_Core_3.1.1_SDK.zip"

# ManusSDK ships a separate client per platform. On Windows the single
# ManusSDK.dll exports the whole API (including CoreSdk_InitializeIntegrated,
# which on Linux lives in a separate libManusSDK_Integrated.so).
if sys.platform == "win32":
    _SDK_ZIP_MEMBER = "ManusSDK_v3.1.1/SDKMinimalClient_Windows/ManusSDK/lib/ManusSDK.dll"
    _LIB_FILENAME = "ManusSDK.dll"
    _CLIENT_DIRS = ("SDKMinimalClient_Windows", "SDKClient_Windows")
elif sys.platform.startswith("linux"):
    _SDK_ZIP_MEMBER = "ManusSDK_v3.1.1/SDKMinimalClient_Linux/ManusSDK/lib/libManusSDK_Integrated.so"
    _LIB_FILENAME = "libManusSDK_Integrated.so"
    _CLIENT_DIRS = ("SDKMinimalClient_Linux", "SDKClient_Linux")
else:
    raise RuntimeError(f"Unsupported platform for ManusSDK: {sys.platform}")

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_sdk_root() -> str | None:
    """Locate the vendored MANUS_Core_3.1.1_SDK directory.

    Walks up from the package location (covers a vendored install, e.g. in the
    repo's third_party/) and from the current working directory (covers a
    pip-installed package run from anywhere inside the repo). At each level the
    SDK may sit either directly or under a third_party/ subdirectory.
    """
    for start in {os.path.dirname(os.path.abspath(__file__)), os.getcwd()}:
        directory = start
        while True:
            for base in (directory, os.path.join(directory, "third_party")):
                candidate = os.path.join(base, "MANUS_Core_3.1.1_SDK", "ManusSDK_v3.1.1")
                if os.path.isdir(candidate):
                    return candidate
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent
    return None


def resolve_lib_path() -> str:
    """Resolve the SDK library path.

    Search order:
      1. vendored SDK in MANUS_Core_3.1.1_SDK/ (found by walking up from the
         package location or the current working directory);
      2. repo-local ManusSDK/lib next to the package (legacy convention);
      3. user cache;
      4. auto-download from the Manus installer.
    """
    candidates = []
    sdk_root = _find_sdk_root()
    if sdk_root is not None:
        for client_dir in _CLIENT_DIRS:
            candidates.append(os.path.join(sdk_root, client_dir, "ManusSDK", "lib", _LIB_FILENAME))
    candidates.append(os.path.join(_PACKAGE_PARENT, "ManusSDK", "lib", _LIB_FILENAME))
    cache_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "manus_glove",
        "lib",
        _LIB_FILENAME,
    )
    candidates.append(cache_path)

    for path in candidates:
        if os.path.isfile(path):
            return path

    logger.info("ManusSDK library not found locally; downloading from Manus installer...")
    download_sdk(cache_path)
    return cache_path


def download_sdk(dest: str) -> None:
    """Download the Manus installer zip and extract the SDK library to *dest*."""

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger.info("Downloading %s ...", _SDK_ZIP_URL)
        urllib.request.urlretrieve(_SDK_ZIP_URL, tmp_path)
        with zipfile.ZipFile(tmp_path) as zf:
            with zf.open(_SDK_ZIP_MEMBER) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        logger.info("Saved ManusSDK .so to %s", dest)
    except Exception:
        # Clean up partial file on failure
        if os.path.exists(dest):
            os.remove(dest)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
