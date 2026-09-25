#!/usr/bin/env bash
# Validate a locally loaded Klein image before a workflow authenticates or publishes.
set -euo pipefail

if (( $# != 4 )); then
  echo "Usage: $0 IMAGE SOURCE_COMMIT ENVIRONMENT_VERSION REQUIREMENTS_LOCK_SHA256" >&2
  exit 2
fi

readonly IMAGE="$1"
readonly EXPECTED_SOURCE_COMMIT="$2"
readonly EXPECTED_ENVIRONMENT_VERSION="$3"
readonly EXPECTED_LOCK_SHA256="$4"
readonly REPOSITORY="/opt/flux2-klein-9b-fft"
readonly BUILD_INFO="${REPOSITORY}/.container-build-info.json"
readonly IDLE_CONTAINER="klein-image-validation-$$"

cleanup() { docker rm -f "${IDLE_CONTAINER}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
fail() { printf 'Container validation failed: %s\n' "$*" >&2; exit 1; }
require_equal() {
  local actual="$1" expected="$2" subject="$3"
  [[ "${actual}" == "${expected}" ]] || fail "${subject}: expected ${expected@Q}, got ${actual@Q}"
}

[[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit must be a lowercase 40-character SHA"
[[ "${EXPECTED_LOCK_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "requirements lock digest must be a lowercase SHA-256"
[[ "${EXPECTED_ENVIRONMENT_VERSION}" != "unknown" && "${EXPECTED_ENVIRONMENT_VERSION}" != "unversioned" ]] || fail "environment version must be a release identity"

printf 'Validating local image %s\n' "${IMAGE}"
docker image inspect "${IMAGE}" >/dev/null
require_equal "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${IMAGE}")" "linux/amd64" "image platform"
require_equal "$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${IMAGE}")" "${EXPECTED_SOURCE_COMMIT}" "OCI source revision"
require_equal "$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' "${IMAGE}")" "${EXPECTED_ENVIRONMENT_VERSION}" "OCI environment version"
require_equal "$(docker image inspect --format '{{ index .Config.Labels "com.thorne-marlowe.klein.requirements-sha256" }}' "${IMAGE}")" "${EXPECTED_LOCK_SHA256}" "OCI requirements lock digest"

docker run --rm --entrypoint python "${IMAGE}" - "${BUILD_INFO}" "${REPOSITORY}" "${EXPECTED_SOURCE_COMMIT}" "${EXPECTED_ENVIRONMENT_VERSION}" "${EXPECTED_LOCK_SHA256}" <<'PY'
import json
import sys
path, repository, commit, version, lock = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    info = json.load(handle)
expected = {"source_commit": commit, "environment_version": version,
            "requirements_lock_sha256": lock, "repository": repository}
if info != expected:
    raise SystemExit(f"unexpected build metadata: {info!r}")
PY

passthrough_output="$(docker run --rm "${IMAGE}" true 2>&1)" || fail "entrypoint command passthrough failed"
printf '%s\n' "${passthrough_output}"
for identity_line in "environment_version=${EXPECTED_ENVIRONMENT_VERSION}" "source_commit=${EXPECTED_SOURCE_COMMIT}" "requirements_lock_sha256=${EXPECTED_LOCK_SHA256}" "repository=${REPOSITORY}"; do
  grep -Fqx "${identity_line}" <<<"${passthrough_output}" >/dev/null || fail "entrypoint identity report omitted ${identity_line@Q}"
done

docker run -d --name "${IDLE_CONTAINER}" "${IMAGE}" >/dev/null
sleep 1
[[ "$(docker inspect --format '{{.State.Running}}' "${IDLE_CONTAINER}")" == "true" ]] || fail "zero-argument entrypoint did not remain running"
idle_output="$(docker logs "${IDLE_CONTAINER}" 2>&1)" || fail "could not read zero-argument entrypoint output"
grep -Fqx "source_commit=${EXPECTED_SOURCE_COMMIT}" <<<"${idle_output}" >/dev/null || fail "zero-argument entrypoint identity report was incomplete"
docker rm -f "${IDLE_CONTAINER}" >/dev/null

docker run --rm --entrypoint python "${IMAGE}" -m pip check
docker run --rm --entrypoint python "${IMAGE}" - <<'PY'
import importlib
import importlib.metadata
from pathlib import Path
from scripts.runpod_preflight import CORE_PACKAGES, _locked_core_versions
locked = _locked_core_versions(Path("requirements-smoke.txt"))
if set(locked) != set(CORE_PACKAGES):
    raise SystemExit(f"unexpected locked core package set: {sorted(locked)}")
for distribution, module in CORE_PACKAGES.items():
    importlib.import_module(module)
    installed = importlib.metadata.version(distribution)
    if installed != locked[distribution]:
        raise SystemExit(f"{distribution}: expected {locked[distribution]}, got {installed}")
torch = importlib.import_module("torch")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected torch CUDA 12.8 build, got {torch.version.cuda!r}")
PY

preflight_json="$(docker run --rm --entrypoint python "${IMAGE}" scripts/runpod_preflight.py --profile system --json)" || fail "system preflight returned a failing exit status"
printf '%s\n' "${preflight_json}"
printf '%s' "${preflight_json}" | python3 -c '
import json
import sys
document = json.load(sys.stdin)
if document.get("exit_code") != 0:
    raise SystemExit("system preflight reported a failing exit code")
failures = [check["identifier"] for check in document.get("checks", []) if check.get("status") == "FAIL"]
if failures:
    raise SystemExit("system preflight failures: " + ", ".join(failures))
'

docker run --rm "${IMAGE}" python scripts/train_klein_standalone.py --help >/dev/null
docker run --rm "${IMAGE}" python scripts/runpod_preflight.py --help >/dev/null
docker run --rm "${IMAGE}" python scripts/run_container_tests.py
rename_test_output="$(docker run --rm "${IMAGE}" python -m unittest -v tests.test_runpod_preflight.RunpodPreflightTests.test_real_linux_checkpoint_rename_probe_publishes_and_cleans_up 2>&1)" || fail "Linux checkpoint-rename test failed"
printf '%s\n' "${rename_test_output}"
if grep -Eiq 'skipped|OK \(skipped=' <<<"${rename_test_output}"; then fail "Linux checkpoint-rename test was skipped"; fi

docker run --rm --entrypoint /bin/sh "${IMAGE}" -ec '
repository=/opt/flux2-klein-9b-fft
test -d "$repository/scripts"
test -d "$repository/tests"
test -d "$repository/docs"
test -f "$repository/requirements-smoke.txt"
test ! -e "$repository/.git"
for path in models model datasets data outputs checkpoints runs; do test ! -e "$repository/$path"; done
runtime_assets="$(find "$repository" -xdev -type f \( -name "*.safetensors" -o -name "*.safetensors.index.json" -o -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.zip" -o -name "*.7z" \) -print)"
if [ -n "$runtime_assets" ]; then
  echo "runtime asset or archive found in image:" >&2
  printf '%s\n' "$runtime_assets" >&2
  exit 1
fi
credential_files="$(find "$repository" -xdev -type f \( -name ".env" -o -name ".env.*" -o -name "*.pem" -o -name "*.key" \) -print)"
if [ -n "$credential_files" ]; then
  echo "credential-like file found in image:" >&2
  printf '%s\n' "$credential_files" >&2
  exit 1
fi
'

printf 'Container validation passed: %s\n' "${IMAGE}"
