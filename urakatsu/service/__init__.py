"""Service layer: one workspace exposed over gRPC, gRPC-web, and JSON HTTP.

- ``serve``  : UI + gRPC-web + JSON API on one HTTP port, plus native gRPC on another.
- ``listen`` : API mode (JSON + native gRPC, no UI) for integrations and automation.
"""
