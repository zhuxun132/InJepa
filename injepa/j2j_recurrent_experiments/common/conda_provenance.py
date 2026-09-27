"""Bind Habitat's official conda distribution to the actual imported bytes."""
import hashlib
import json
from pathlib import Path
import tarfile

from j2j_iclr_experiments.common.artifacts import file_identity


def verify_habitat_conda_install(install_root, *, module_path, native_paths,
                                 expected_commit):
    root = Path(install_root).resolve()
    records = list((root / "conda-meta").glob("habitat-sim-0.2.4-*.json"))
    if len(records) != 1:
        raise ValueError("official Habitat conda record must be unique")
    record = json.loads(records[0].read_text())
    build = record.get("build", "")
    if (record.get("name") != "habitat-sim" or record.get("version") != "0.2.4"
            or not build.endswith("_" + expected_commit)):
        raise ValueError("Habitat conda build does not bind the admitted commit")
    archive_path = Path(record["package_tarball_full_path"]).resolve()
    expected_name = f"habitat-sim-0.2.4-{build}.tar.bz2"
    url = "https://conda.anaconda.org/aihabitat/linux-64/" + expected_name
    if record.get("url") != url or archive_path.name != expected_name:
        raise ValueError("Habitat conda archive is not the official release URL")
    archive_identity = file_identity(archive_path, name="Habitat conda archive")
    if archive_identity["sha256"] != record.get("sha256"):
        raise ValueError("Habitat conda archive SHA mismatch")
    paths = [Path(module_path).resolve(), *(Path(p).resolve() for p in native_paths)]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Habitat conda needs module and unique native extensions")
    members = []
    with tarfile.open(archive_path, "r:bz2") as archive:
        index = json.load(archive.extractfile("info/index.json"))
        if any(index.get(key) != record.get(key) for key in ("name", "version", "build")):
            raise ValueError("Habitat archive index differs from installed record")
        for path in paths:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("Habitat imported member is outside conda install") from exc
            member = archive.getmember(relative)
            if not member.isfile():
                raise ValueError("Habitat imported archive member must be a regular file")
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            identity = file_identity(path, name="Habitat imported member")
            if identity["sha256"] != digest.hexdigest():
                raise ValueError("Habitat live member differs from official archive")
            members.append(dict(identity, archive_member=relative))
    return {"kind": "official_conda", "commit_id": expected_commit, "url": url,
            "archive": archive_identity,
            "record": file_identity(records[0], name="Habitat conda record"),
            "members": members}
