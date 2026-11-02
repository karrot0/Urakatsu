#!/bin/sh
# Installs Urakatsu on Linux, and updates it when run again:
#
#   curl -fsSL https://urakatsu.com/install.sh | sh
#
# Downloads the release's tarball, checks it, and runs its setup, which installs per user without
# administrator rights (packaging/launcher/src/platform/unix.rs). Options go after `sh -s --`:
#
#   --nightly  the nightly build instead of the latest release
#   --quiet    no setup window: the default data folder, and the app opens when it is done
#
# Everything is inside main, so a download cut short runs nothing.

set -eu

REPO="karrot0/Urakatsu"
RELEASES="https://github.com/$REPO/releases"
API="https://api.github.com/repos/$REPO/releases"

say() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

# The package with WebKitGTK 4.1, which the setup window needs, for this distribution.
webkit_hint() {
    if command -v apt-get > /dev/null 2>&1; then
        say "  sudo apt-get install libwebkit2gtk-4.1-0"
    elif command -v dnf > /dev/null 2>&1; then
        say "  sudo dnf install webkit2gtk4.1"
    elif command -v pacman > /dev/null 2>&1; then
        say "  sudo pacman -S webkit2gtk-4.1"
    elif command -v zypper > /dev/null 2>&1; then
        say "  sudo zypper install libwebkit2gtk-4_1-0"
    else
        say "  WebKitGTK 4.1, from your distribution's packages"
    fi
}

# Reads a release's JSON (GitHub's assets list each one's name first) and prints the Linux
# tarball's download URL, then its digest. Works without jq, on one line or pretty-printed.
pick_asset() {
    tr ',' '\n' | awk '
        function value() { sub(/.*: *"/, ""); sub(/".*/, ""); return $0 }
        /"name": *"/ { if (found) exit; found = /-linux-x86_64\.tar\.gz"/ }
        found && /"browser_download_url"/ { url = value() }
        found && /"digest"/ { digest = value() }
        END { print url; print digest }
    '
}

main() {
    release="$API/latest"
    channel="latest release"
    quiet=""
    for option in "$@"; do
        case "$option" in
            --nightly)
                release="$API/tags/nightly"
                channel="nightly build"
                ;;
            --quiet) quiet="--quiet" ;;
            *) die "unknown option $option (use --nightly or --quiet)" ;;
        esac
    done

    [ "$(uname -s)" = Linux ] || die "this script is for Linux. Other systems: $RELEASES/latest"
    [ "$(uname -m)" = x86_64 ] || die "Urakatsu's Linux build is for x86-64, and this is $(uname -m)"
    getconf GNU_LIBC_VERSION > /dev/null 2>&1 || die "Urakatsu needs a Linux with glibc 2.35 or newer"
    [ "$(id -u)" != 0 ] || die "Urakatsu installs per user: run this as yourself, without sudo"
    [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || die "Urakatsu is a desktop app and there is no display here"
    command -v curl > /dev/null 2>&1 || die "this script needs curl"

    say "Finding the $channel..."
    json="$(curl -fsSL "$release")" || die "could not read $release (GitHub may be limiting requests: try again later)"
    asset="$(printf '%s' "$json" | pick_asset)"
    url="$(printf '%s\n' "$asset" | sed -n 1p)"
    digest="$(printf '%s\n' "$asset" | sed -n 2p)"
    [ -n "$url" ] || die "the $channel has no Linux build yet. See $RELEASES"

    # Not /tmp, which is often mounted noexec.
    cache="${XDG_CACHE_HOME:-$HOME/.cache}"
    mkdir -p "$cache"
    work="$(mktemp -d "$cache/urakatsu-install.XXXXXX")"
    trap 'rm -rf "$work"' EXIT
    trap 'exit 130' INT TERM

    say "Downloading ${url##*/}..."
    curl -fL --progress-bar -o "$work/urakatsu.tar.gz" "$url" || die "the download failed"
    if [ -n "$digest" ]; then
        printf '%s  %s\n' "${digest#sha256:}" "$work/urakatsu.tar.gz" | sha256sum -c --status \
            || die "the download does not match its checksum"
    fi
    tar -xzf "$work/urakatsu.tar.gz" -C "$work"

    missing="$(ldd "$work/Urakatsu" 2>&1 | grep 'not found' || true)"
    case "$missing" in
        *GLIBC*) die "Urakatsu needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36)" ;;
        *libwebkit2gtk*)
            say "Urakatsu needs WebKitGTK 4.1. Install it, then run this again:"
            webkit_hint
            exit 1
            ;;
        ?*) die "Urakatsu needs libraries this system is missing: $missing" ;;
    esac

    if [ -n "$quiet" ]; then
        "$work/Urakatsu" --quiet < /dev/null
        say "Urakatsu is installed: open it from the applications menu."
    else
        say "Opening Urakatsu setup..."
        "$work/Urakatsu" < /dev/null
    fi
}

main "$@"
