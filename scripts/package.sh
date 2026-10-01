#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
test -x ./yolo_app_26
if [[ -d .git || -f .git ]]; then
    version="$(git rev-parse --short=12 HEAD)"
    if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
        version="${version}-dirty"
    fi
else
    version="$(cat BUILD_COMMIT 2>/dev/null || printf 'unversioned')"
fi
name="yolo-hid-rk3588-${version}"
mkdir -p dist
stage="$(mktemp -d "${TMPDIR:-/tmp}/yolo-hid-package.XXXXXXXX")"
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/$name/src" "$stage/$name/tools" "$stage/$name/scripts" "$stage/$name/tests" "$stage/$name/Models"
install -m 755 yolo_app_26 setup.sh "$stage/$name/"
cp Makefile README.md .gitignore "$stage/$name/"
while IFS= read -r -d '' source; do
    mkdir -p "$stage/$name/$(dirname "$source")"
    cp "$source" "$stage/$name/$source"
done < <(find src tools scripts tests -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.py' -o -name '*.sh' -o -name Makefile \) -print0)
printf '%s\n' "$version" > "$stage/$name/BUILD_COMMIT"
tar -C "$stage" -czf "dist/$name.tar.gz" "$name"
(cd dist && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
printf 'PACKAGE=dist/%s.tar.gz\n' "$name"
printf 'SHA256=dist/%s.tar.gz.sha256\n' "$name"
