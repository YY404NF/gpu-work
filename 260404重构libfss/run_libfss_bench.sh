#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/.deps/libfss-src"
BIN="$ROOT/libfss_batch_bench"
TMP_BIN="$ROOT/.libfss_batch_bench.$$"

if [[ ! -f "$SRC/cpp/fss-client.cpp" ]]; then
  echo "missing local libfss source under $SRC" >&2
  exit 1
fi

g++ \
  -O3 \
  -std=c++20 \
  -maes \
  -msse4.2 \
  -I"$SRC/cpp" \
  -DOPENSSL_AES_H \
  -DAESNI \
  -Daesni_set_encrypt_key=AES_set_encrypt_key \
  -Daesni_set_decrypt_key=AES_set_decrypt_key \
  -Daesni_encrypt=AES_encrypt \
  -Daesni_decrypt=AES_decrypt \
  "$ROOT/libfss_batch_bench.cpp" \
  "$SRC/cpp/fss-client.cpp" \
  "$SRC/cpp/fss-server.cpp" \
  "$SRC/cpp/fss-common.cpp" \
  -lcrypto \
  -lgmpxx \
  -lgmp \
  -lpthread \
  -ldl \
  -lssl \
  -o "$TMP_BIN"

mv "$TMP_BIN" "$BIN"

"$BIN" "$@"
