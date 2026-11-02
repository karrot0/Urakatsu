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

# The app's window is Qt (pywebview's Qt backend), from wheels that leave these to the system.
# Setup installs it later, so they are checked here, before they can stop the first start.
QT_LIBS="libnss3.so libnspr4.so libXtst.so.6 libasound.so.2"
# Only X11 sessions load Qt's xcb plugin; many desktops lack these, libxcb-cursor above all.
XCB_LIBS="libxcb-cursor.so.0 libxcb-icccm.so.4 libxcb-image.so.0 libxcb-keysyms.so.1
    libxcb-render-util.so.0 libxcb-shape.so.0 libxcb-xkb.so.1 libxkbcommon-x11.so.0"

# The libraries above that this system lacks.
missing_qt_libs() {
    known="$(PATH="$PATH:/sbin:/usr/sbin" ldconfig -p 2> /dev/null)" || return 0
    needed="$QT_LIBS"
    [ -n "${WAYLAND_DISPLAY:-}" ] || needed="$needed $XCB_LIBS"
    for lib in $needed; do
        case "$known" in
            *"$lib "*) ;;
            *) printf '%s ' "$lib" ;;
        esac
    done
}

# Everything the setup window and the app need from this distribution, as one command.
packages_hint() {
    if command -v apt-get > /dev/null 2>&1; then
        alsa=libasound2 # renamed in Ubuntu 24.04 and Debian 13, and apt will not pick for us
        apt-cache show libasound2t64 > /dev/null 2>&1 && alsa=libasound2t64
        say "  sudo apt-get install libwebkit2gtk-4.1-0 libnss3 libxtst6 $alsa libxcb-cursor0" \
            "libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0" \
            "libxcb-xkb1 libxkbcommon-x11-0"
    elif command -v dnf > /dev/null 2>&1; then
        say "  sudo dnf install webkit2gtk4.1 nss libXtst alsa-lib xcb-util-cursor xcb-util-wm" \
            "xcb-util-image xcb-util-keysyms xcb-util-renderutil libxkbcommon-x11"
    elif command -v pacman > /dev/null 2>&1; then
        say "  sudo pacman -S webkit2gtk-4.1 nss libxtst alsa-lib xcb-util-cursor xcb-util-wm" \
            "xcb-util-image xcb-util-keysyms xcb-util-renderutil libxkbcommon-x11"
    elif command -v zypper > /dev/null 2>&1; then
        say "  sudo zypper install libwebkit2gtk-4_1-0 mozilla-nss libXtst6 libasound2 libxcb-cursor0" \
            "libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0" \
            "libxcb-xkb1 libxkbcommon-x11-0"
    else
        say "  the packages with these libraries, from your distribution"
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
    for tool in curl awk tar sha256sum ldd; do
        command -v "$tool" > /dev/null 2>&1 || die "this script needs $tool, which is not installed"
    done

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
    esac
    libs="$(printf '%s\n' "$missing" | awk 'NF { printf "%s ", $1 }')$(missing_qt_libs)"
    if [ -n "$libs" ]; then
        say "This system is missing libraries Urakatsu needs: $libs"
        say "Install them, then run this again:"
        packages_hint
        exit 1
    fi

    if [ -n "$quiet" ]; then
        "$work/Urakatsu" --quiet < /dev/null
        say "Urakatsu is installed and starting. It is also in your applications menu."
    else
        say "Opening Urakatsu setup..."
        "$work/Urakatsu" < /dev/null
    fi
}

main "$@"
