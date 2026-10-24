"""``urakatsu`` command line.

    urakatsu serve      workspace UI + gRPC-web + JSON API + native gRPC in one process
    urakatsu desktop    the workspace in its own window (what the Windows app runs)
    urakatsu listen     API mode: JSON API + native gRPC, no UI
    urakatsu run        one-shot pipeline on image files
    urakatsu providers  list providers and availability
    urakatsu devices    show discovered hardware and runtimes
    urakatsu runtimes   list, check or update the native runtimes (always the newest releases)
    urakatsu config     print the active configuration (TOML)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from urakatsu import __version__


def _add_service_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (UI, gRPC-web, JSON API)")
    parser.add_argument("--grpc-port", type=int, default=50051, help="native gRPC port (0 to disable)")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N | dml")
    parser.add_argument("--cors", action="append", default=None, help="allowed CORS origin (repeatable)")
    parser.add_argument("--log-level", default="info")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urakatsu", description="Modular manga translation pipeline")
    parser.add_argument("--version", action="version", version=f"urakatsu {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the workspace: UI + gRPC-web + JSON API + gRPC")
    _add_service_args(serve)
    serve.add_argument("--ui-dir", type=Path, default=None, help="built UI directory (default: web/dist)")
    serve.add_argument("--no-ui", action="store_true", help="do not serve the UI")

    sub.add_parser("desktop", help="open the workspace in its own window (needs the desktop extra)")

    listen = sub.add_parser("listen", help="API mode: expose JSON API + gRPC without the UI")
    _add_service_args(listen)
    listen.set_defaults(host="0.0.0.0")

    run = sub.add_parser("run", help="run the pipeline on image files")
    run.add_argument("images", nargs="+", type=Path)
    run.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    run.add_argument("--operation", default="full", choices=["full", "through", "only", "stages"])
    run.add_argument("--stage", default=None)
    run.add_argument("--stages", nargs="*", default=None)
    run.add_argument("--device", default=None)
    run.add_argument("--target", default=None, help="target language (overrides config)")
    run.add_argument("--keep", action="store_true", help="keep the project in the workspace")

    sub.add_parser("providers", help="list providers")
    sub.add_parser("devices", help="show hardware, device, runtimes, model store")
    runtimes = sub.add_parser("runtimes", help="list, check or update the native runtimes (llama.cpp, stable-diffusion.cpp)")
    runtimes.add_argument("action", nargs="?", default="list", choices=["list", "check", "update"])
    sub.add_parser("config", help="print the active configuration as TOML")
    sub.add_parser("codegen", help="regenerate protobuf stubs")
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_serve(args: argparse.Namespace, serve_ui: bool) -> int:
    from urakatsu.service.app import serve
    from urakatsu.service.workspace import Workspace

    from urakatsu.runtimes.packages import start_updater

    _configure_logging(args.log_level)
    start_updater()  # always the newest llama.cpp and stable-diffusion.cpp (AGENTS.md)
    workspace = Workspace(device=args.device)
    ui_dir = getattr(args, "ui_dir", None)
    try:
        asyncio.run(
            serve(
                workspace,
                host=args.host,
                port=args.port,
                grpc_port=args.grpc_port or None,
                ui_dir=ui_dir,
                serve_ui=serve_ui and not getattr(args, "no_ui", False),
                cors_origins=args.cors,
                log_level=args.log_level,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from urakatsu.core.config import load_config
    from urakatsu.core.stage import Operation, Stage
    from urakatsu.pipeline import Pipeline, PipelineError, ProjectStore, Request
    from urakatsu.runtimes import RuntimeManager

    _configure_logging("info")
    config = load_config()
    if args.device:
        config.device = args.device
    if args.target:
        config.translation.target_language = args.target
    if args.operation == "full":
        operation = Operation.full()
    elif args.operation == "through":
        operation = Operation.through(Stage.parse(args.stage or "translation"))
    elif args.operation == "only":
        operation = Operation.only(Stage.parse(args.stage or "detection"))
    else:
        operation = Operation.select([Stage.parse(s) for s in (args.stages or [])])

    store = ProjectStore()
    project = store.create(f"cli {args.images[0].stem}")
    pages = [store.import_page(project, path.name, path.read_bytes()) for path in args.images]
    session = store.session(project)
    pipeline = Pipeline(config, RuntimeManager(config.device))

    def progress(event):
        print(f"[{event.kind.value:>9}] {event.page or '':>32} {event.stage.value if event.stage else '':>11} {event.model or ''} {event.message}")

    try:
        report = asyncio.run(pipeline.execute(session, Request(operation=operation, progress=progress)))
    except PipelineError as error:
        print(f"failed: {error}", file=sys.stderr)
        store.delete(project)
        return 1
    from urakatsu.renderer import Renderer
    from urakatsu.service.views import page_view

    args.out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(config.typesetting)
    snapshot = session.snapshot()
    for page, path in zip(pages, args.images):
        renderer.render(snapshot, page).composite().save(args.out / f"{path.stem}.render.png")
        (args.out / f"{path.stem}.json").write_text(page_view(snapshot, page).model_dump_json(indent=2), encoding="utf-8")
    if not args.keep:
        store.delete(project)
    print(f"{report.status.value}: {report.completed}/{report.total} stages in {report.elapsed_ms:.0f} ms -> {args.out}")
    return 0


def cmd_providers() -> int:
    from urakatsu.core.registry import registry

    for info in registry.info():
        status = "available" if info.available.ok else f"unavailable: {info.available.reason}"
        print(f"{info.stage.value:<12} {info.id:<28} {status}")
        for runtime in info.runtimes:
            mark = "*" if runtime.name == info.default_runtime else " "
            state = "ok" if runtime.available.ok else runtime.available.reason
            print(f"{'':<12} {mark} runtime {runtime.name:<10} {state}")
        for weight in info.weights:
            print(f"{'':<12}   weights: {weight}")
    return 0


def cmd_devices() -> int:
    from urakatsu.core.config import load_config
    from urakatsu.runtimes import RuntimeManager

    manager = RuntimeManager(load_config().device)
    report = manager.describe()
    print(f"platform : {report.hardware.platform}")
    print(f"selected : {report.device.label()} ({report.device.description})")
    for device in report.hardware.candidates():
        print(f"  - {device.label():<9} {device.description}{f' [{device.target}]' if device.target else ''}")
    for name, entry in report.runtimes.items():
        print(f"runtime  : {name:<6} {'available' if entry.get('available') else 'unavailable: ' + str(entry.get('reason'))}")
    print(f"store    : {report.store_root} ({report.store_bytes / 1e6:.1f} MB)")
    return 0


def cmd_runtimes(action: str) -> int:
    """Urakatsu runs the newest runtime releases (AGENTS.md); this installs and reports them."""
    from urakatsu.runtimes.hardware import discover
    from urakatsu.runtimes.packages import PACKAGES, PackageManager, preferred_variants

    manager = PackageManager()
    devices = discover().candidates()
    status = 0
    for package in PACKAGES.values():
        installed = manager.installed(package)
        current = f"{installed.tag} ({installed.variant})" if installed else "not installed"
        if action == "list":
            print(f"{package.name:<22} {current}")
            continue
        if package.on_demand and installed is None and action == "update":
            print(f"{package.name:<22} not installed (installs when a source system needs it)")
            continue
        if action == "check":
            latest = manager.latest(package, preferred_variants(devices))
            fresh = installed is not None and (installed.tag, installed.variant) == (latest.tag, latest.variant)
            print(f"{package.name:<22} {current:<36} latest {latest.tag} ({latest.variant}){'' if fresh else '  UPDATE AVAILABLE'}")
            status |= 0 if fresh else 1
            continue

        def progress(name: str, done: int, total: int) -> None:
            print(f"\r  {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)

        updated = manager.ensure(package, devices, progress=progress, force=True)
        print(f"\r{package.name:<22} {updated.tag} ({updated.variant})" + " " * 30)
    return status


def cmd_config() -> int:
    import tomli_w

    from urakatsu.core.config import config_path, load_config

    print(f"# {config_path()}")
    print(tomli_w.dumps(load_config().model_dump(mode="json", exclude_none=True)))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; config and OCR text are not ASCII.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "desktop":
        from urakatsu.desktop.app import main as desktop_main

        return desktop_main()
    if args.command == "serve":
        return cmd_serve(args, serve_ui=True)
    if args.command == "listen":
        return cmd_serve(args, serve_ui=False)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "providers":
        return cmd_providers()
    if args.command == "devices":
        return cmd_devices()
    if args.command == "runtimes":
        return cmd_runtimes(args.action)
    if args.command == "config":
        return cmd_config()
    if args.command == "codegen":
        from urakatsu.service.codegen import main as codegen

        return codegen()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
