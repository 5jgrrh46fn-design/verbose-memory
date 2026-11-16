#!/usr/bin/env bash
# Download the result archives, verify them against SHA256SUMS and extract
# them into results/ and reconstruction/ at the repository root.
# Needs curl, zstd and tar. About 12 GB on disk after extraction.
set -euo pipefail
cd "$(dirname "$0")/.."
# The address is put together from its parts so that it reads the same in
# a mirror of this repository.
HOST=github.com
OWNER=5jgrrh46fn-design
REPO=verbose-memory
BASE=${RELEASE_URL:-https://$HOST/$OWNER/$REPO/releases/download/submission}
mkdir -p downloads
while read -r sum name; do
  [ -f "downloads/$name" ] || curl -L --fail -o "downloads/$name" "$BASE/$name"
done < SHA256SUMS
if command -v sha256sum >/dev/null; then
  (cd downloads && sha256sum -c ../SHA256SUMS)
else
  (cd downloads && shasum -a 256 -c ../SHA256SUMS)
fi
while read -r sum name; do
  zstd -dc --long=27 "downloads/$name" | tar -xf -
done < SHA256SUMS
echo "extracted: results/ and reconstruction/"
