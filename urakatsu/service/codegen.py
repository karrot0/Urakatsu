"""Generates the Python protobuf + gRPC stubs into ``urakatsu/service/gen``.

Run ``python -m urakatsu.service.codegen``. The generated ``*_pb2_grpc.py``
imports its sibling with an absolute ``urakatsu.v1`` path, which this script
rewrites to a package-relative import.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROTO_ROOT = HERE / "proto"
PROTO = PROTO_ROOT / "urakatsu" / "v1" / "urakatsu.proto"
OUT = HERE / "gen"


def main() -> int:
    from grpc_tools import protoc

    OUT.mkdir(parents=True, exist_ok=True)
    args = [
        "protoc",
        f"-I{PROTO_ROOT}",
        f"--python_out={OUT}",
        f"--pyi_out={OUT}",
        f"--grpc_python_out={OUT}",
        str(PROTO),
    ]
    code = protoc.main(args)
    if code != 0:
        print("protoc failed", file=sys.stderr)
        return code
    # protoc nests output under urakatsu/v1/; flatten into gen/ with relative imports.
    nested = OUT / "urakatsu" / "v1"
    for path in nested.glob("urakatsu_pb2*"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^from urakatsu\.v1 import urakatsu_pb2", "from . import urakatsu_pb2", text, flags=re.M)
        text = re.sub(r"^import urakatsu_pb2", "from . import urakatsu_pb2", text, flags=re.M)
        (OUT / path.name).write_text(text, encoding="utf-8")
        path.unlink()
    for directory in (nested, nested.parent):
        try:
            for leftover in directory.iterdir():
                if leftover.name == "__init__.py":
                    leftover.unlink()
            directory.rmdir()
        except OSError:
            pass
    print(f"generated {sorted(p.name for p in OUT.glob('urakatsu_pb2*'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
