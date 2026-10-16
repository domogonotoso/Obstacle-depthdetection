"""Verify CUDA without loading models; optionally probe the running server."""
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", help="E.g. http://127.0.0.1:8000 (requires no local torch)")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    try:
        if args.server:
            from app.remote import RemoteClient
            client = RemoteClient(args.server, timeout=10)
            try:
                info = client.diagnostics()
            finally:
                client.close()
        else:
            from app.runtime import gpu_diagnostics
            info = gpu_diagnostics()
        print(json.dumps(info, indent=2))
        if args.require_cuda and not (info.get("cuda_available") and info.get("cuda_tensor_test")):
            print("CUDA validation did not pass", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Diagnostics failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
