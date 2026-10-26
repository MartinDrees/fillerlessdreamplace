#!/usr/bin/env bash
set -euo pipefail

mkdir -p bookshelf

BASE="https://www.ispd.cc/contests/05/ispd05-contest"

benchmarks=(
    adaptec1
    adaptec2
    adaptec3
    adaptec4
    bigblue1
    bigblue2
    bigblue3
    bigblue4
)

for bench in "${benchmarks[@]}"; do
    echo "=== $bench ==="

    archive="${bench}.tar.gz"

    if [[ "$bench" == "adaptec1" || "$bench" == "adaptec3" ]]; then
        url="$BASE/$archive"
    else
        url="$BASE/benchmarks/$archive"
    fi

    tmp=$(mktemp -d)

    echo "Downloading..."
    wget -q --show-progress "$url" -O "$tmp/$archive"

    echo "Extracting outer archive..."
    tar -xzf "$tmp/$archive" -C "$tmp"
    rm "$tmp/$archive"

    # Some benchmark archives contain individually gzipped
    # Bookshelf files. Uncompress those wherever they occur.
    find "$tmp" -type f -name '*.gz' -exec gunzip -f {} +

    # Find the actual directory containing <bench>.aux.
    aux=$(find "$tmp" -type f -name "${bench}.aux" -print -quit)

    if [[ -z "$aux" ]]; then
        echo "ERROR: could not find ${bench}.aux"
        find "$tmp" -maxdepth 3 -type f
        exit 1
    fi

    srcdir=$(dirname "$aux")

    rm -rf "bookshelf/$bench"
    mkdir -p "bookshelf/$bench"

    # Copy only the actual benchmark files, independent of how
    # directories are arranged inside the tarball.
    cp -a "$srcdir"/. "bookshelf/$bench/"

    rm -rf "$tmp"
done

echo
echo "Result:"
find bookshelf -maxdepth 2 -type f | sort
