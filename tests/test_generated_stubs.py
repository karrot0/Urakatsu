"""The gRPC stubs refuse to import on an older grpcio or protobuf than generated them, so the
dependency floors in pyproject.toml must not fall below those versions: an update installs over
an existing environment and keeps any grpcio that still meets the floor."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "urakatsu" / "service" / "gen"


def floor(package: str) -> tuple[int, ...]:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["dependencies"]
    for dependency in dependencies:
        match = re.fullmatch(rf"{re.escape(package)}\s*>=\s*([\d.]+)", dependency.strip())
        if match:
            return tuple(int(part) for part in match.group(1).split("."))
    raise AssertionError(f"pyproject.toml has no {package}>= floor")


def test_grpcio_floor_meets_the_generated_stubs():
    source = (GEN / "urakatsu_pb2_grpc.py").read_text("utf-8")
    generated = re.search(r"GRPC_GENERATED_VERSION = '([\d.]+)'", source).group(1)
    needed = tuple(int(part) for part in generated.split("."))
    assert floor("grpcio") >= needed
    assert floor("grpcio-tools") >= needed


def test_protobuf_floor_meets_the_generated_messages():
    source = (GEN / "urakatsu_pb2.py").read_text("utf-8")
    match = re.search(r"ValidateProtobufRuntimeVersion\(\s*\S+,\s*(\d+),\s*(\d+),\s*(\d+),", source)
    assert floor("protobuf") >= tuple(int(part) for part in match.groups())
