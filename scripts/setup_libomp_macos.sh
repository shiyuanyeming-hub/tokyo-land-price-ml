#!/usr/bin/env bash
#
# macOS で LightGBM / XGBoost を使うための補助スクリプト。
#
# 両ライブラリは OpenMP ランタイム（libomp.dylib）に依存します。
# 通常は次の 1 行で解決します（これが最も確実です）。
#
#   brew install libomp
#
# このスクリプトは、Homebrew を使えない環境（権限がない、社内ルールで入れられない等）で
# Homebrew のボトルから libomp だけを取り出し、`DYLD_FALLBACK_LIBRARY_PATH` で
# 読み込ませるためのファイルを用意します。
#
# 使い方:
#   ./scripts/setup_libomp_macos.sh
#   export DYLD_LIBRARY_PATH="$PWD/.libs"
#
# もっと確実な方法: 既に libomp.dylib を持っているアプリを探して、そのディレクトリを
# DYLD_LIBRARY_PATH に通す。実測で動作した例（R がインストールされている環境）:
#   export DYLD_LIBRARY_PATH=/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib
#   python -m src.pipeline
#
# 注意: 取得した dylib は Homebrew のボトルそのものです。
# 環境によっては dyld が署名を検証できず読み込めないことがあります
# （その場合は brew install libomp を使ってください）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/.libs"
mkdir -p "${DEST}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "macOS 以外では不要です（Linux では libgomp が同梱されます）"
  exit 0
fi

if [ -f "${DEST}/libomp.dylib" ]; then
  echo "既にあります: ${DEST}/libomp.dylib"
  echo "  export DYLD_LIBRARY_PATH=\"${DEST}\""
  exit 0
fi

echo "Homebrew のボトルから libomp を取得します（ネットワークが必要）"
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:homebrew/core/libomp:pull&service=ghcr.io" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
read -r SHA TAG < <(curl -s https://formulae.brew.sh/api/formula/libomp.json | python3 -c "
import json,sys
files=json.load(sys.stdin)['bottle']['stable']['files']
for tag in ('arm64_tahoe','arm64_sequoia','arm64_sonoma','arm64_ventura','x86_64_ventura','arm64_monterey','x86_64_monterey'):
    if tag in files:
        print(files[tag]['sha256'], tag); break
")
echo "  bottle: ${TAG} (${SHA:0:12}…)"
TMP=$(mktemp -d)
curl -sL -H "Authorization: Bearer ${TOKEN}" -o "${TMP}/libomp.tar.gz" \
  "https://ghcr.io/v2/homebrew/core/libomp/blobs/sha256:${SHA}"
tar xzf "${TMP}/libomp.tar.gz" -C "${TMP}"
cp "$(find "${TMP}" -name 'libomp.dylib' | head -1)" "${DEST}/libomp.dylib"
rm -rf "${TMP}"

echo "完了: ${DEST}/libomp.dylib"
echo "実行前に次を通してください:"
echo "  export DYLD_LIBRARY_PATH=\"${DEST}\""
