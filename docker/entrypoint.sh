#!/usr/bin/env bash
set -euo pipefail

readonly BUILD_INFO="/opt/flux2-klein-9b-fft/.container-build-info.json"
readonly REPOSITORY="/opt/flux2-klein-9b-fft"

echo "Klein container identity"
if [[ -f "${BUILD_INFO}" ]]; then
  python - "${BUILD_INFO}" "${REPOSITORY}" <<'PY'
import json
import sys

info_path, repository = sys.argv[1:]
try:
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"build_metadata=unreadable ({type(exc).__name__})")
else:
    for key in ("environment_version", "source_commit", "requirements_lock_sha256"):
        print(f"{key}={info.get(key, 'unknown')}")
print(f"repository={repository}")
print(f"python={sys.version.split()[0]}")
PY
else
  echo "build_metadata=missing"
  echo "repository=${REPOSITORY}"
  echo "python=$(python -c 'import sys; print(sys.version.split()[0])')"
fi

if (( $# )); then
  exec "$@"
fi

# A no-command container is an intentionally inert, inspectable environment.
# It performs no installation, download, authentication, git operation, or
# training. Use docker stop to end it.
exec sleep infinity
