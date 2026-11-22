import json
import shutil

import pytest
from pydantic import ValidationError

from mcp_server.server import SEED_DIR, build_server


def test_build_server_invalid_seed_asset_raises(tmp_path):
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    assets = json.loads((tmp_path / "assets.json").read_text())
    assets[0]["status"] = "approved"
    (tmp_path / "assets.json").write_text(json.dumps(assets))

    with pytest.raises(ValidationError):
        build_server(tmp_path)


def test_build_server_invalid_seed_brand_raises(tmp_path):
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    (tmp_path / "brands.json").write_text(json.dumps([{"slug": "loomhouse"}]))

    with pytest.raises(ValidationError):
        build_server(tmp_path)


def test_build_server_seed_asset_with_unknown_brand_raises(tmp_path):
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    assets = json.loads((tmp_path / "assets.json").read_text())
    assets[0]["brand"] = "verdant_skin"
    (tmp_path / "assets.json").write_text(json.dumps(assets))

    with pytest.raises(ValueError, match="verdant_skin"):
        build_server(tmp_path)


def test_build_server_invalid_seed_guidelines_raises(tmp_path):
    shutil.copytree(SEED_DIR, tmp_path, dirs_exist_ok=True)
    brands = json.loads((tmp_path / "brands.json").read_text())
    brands[0]["guidelines"]["palette"][0]["hex"] = "oat"
    (tmp_path / "brands.json").write_text(json.dumps(brands))

    with pytest.raises(ValidationError):
        build_server(tmp_path)
