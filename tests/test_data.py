import io
import json
import zipfile

import numpy as np
import pytest
import responses

from cinematch.config import SIGNALS
from cinematch.data import MANIFEST, ArtifactError, ensure_artifacts, load_artifacts


def test_similarity_matches_dense_cosine(art):
    for s in ("sbert", "keywords", "collab"):
        mat = art.embeddings[s] if s == "sbert" else art.factors[s].toarray()
        dense = mat @ mat.T
        np.testing.assert_allclose(art.similarity(s, 1), dense[1], atol=1e-5)


def test_context_signal_formula(art):
    sim = art.similarity("context", 0)
    assert sim[0] == pytest.approx(1.0)
    pop, year = art.catalog["popularity"].to_numpy(), art.catalog["year"].to_numpy()
    expected = (np.exp(-abs(pop[2] - pop[0]) / 20) + np.exp(-abs(year[2] - year[0]) / 10)) / 2
    assert sim[2] == pytest.approx(expected, rel=1e-5)


def test_similarities_stack_shape(art):
    assert art.similarities(0).shape == (len(SIGNALS), art.n)
    with pytest.raises(KeyError):
        art.similarity("nope", 0)


def test_collab_coverage(art):
    assert not art.has_collab(3)
    assert art.collab_coverage == pytest.approx(7 / 8)


def test_load_roundtrip(artifacts_dir, art):
    loaded = load_artifacts(artifacts_dir)
    assert loaded.n == art.n
    assert loaded.catalog["genres_list"].iloc[2] == ["Horror"]
    np.testing.assert_allclose(loaded.similarity("vibe", 0), art.similarity("vibe", 0), atol=2e-3)


def test_missing_manifest(tmp_path):
    with pytest.raises(ArtifactError, match="manifest"):
        load_artifacts(tmp_path)


def test_corrupted_file_detected(artifacts_dir):
    (artifacts_dir / "fac_genre.npz").write_bytes(b"corrupted")
    with pytest.raises(ArtifactError, match="SHA-256"):
        load_artifacts(artifacts_dir)


def test_missing_file_detected(artifacts_dir):
    (artifacts_dir / "emb_vibe.npy").unlink()
    with pytest.raises(ArtifactError, match="manquant"):
        load_artifacts(artifacts_dir)


def test_shape_mismatch_rejected(art):
    from dataclasses import replace

    with pytest.raises(ArtifactError):
        replace(art, embeddings={**art.embeddings, "sbert": art.embeddings["sbert"][:3]})


def zip_dir(directory):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in directory.iterdir():
            zf.write(f, arcname=f.name)
    return buf.getvalue()


@responses.activate
def test_ensure_artifacts_downloads_once(artifacts_dir, tmp_path):
    url = "https://example.org/artifacts.zip"
    responses.get(url, body=zip_dir(artifacts_dir))
    target = tmp_path / "downloaded"
    ensure_artifacts(target, url)
    assert (target / MANIFEST).exists()
    load_artifacts(target)
    ensure_artifacts(target, url)  # déjà présent : pas de second téléchargement
    assert len(responses.calls) == 1


@responses.activate
def test_ensure_artifacts_rejects_paths_outside_directory(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    responses.get("https://example.org/a.zip", body=buf.getvalue())
    with pytest.raises(ArtifactError):
        ensure_artifacts(tmp_path / "a", "https://example.org/a.zip")
    assert not (tmp_path / "evil.txt").exists()


def test_manifest_lists_hashes(artifacts_dir):
    manifest = json.loads((artifacts_dir / MANIFEST).read_text())
    assert len(manifest["files"]) == 8
    assert all(len(f["sha256"]) == 64 for f in manifest["files"].values())
