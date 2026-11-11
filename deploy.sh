#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$PROJECT_DIR/stage-crop-aware-clean"
PLUGIN_DEST="/usr/lib/obs-plugins/obs-vision-filter.so"
DATA_DEST="/usr/share/obs/obs-plugins/obs-vision-filter"
BACKUP_DIR="$PROJECT_DIR/backups/$(date +%Y%m%d-%H%M%S)"

BUILD_FIRST=0
DRY_RUN=0
UNINSTALL=0
NO_BACKUP=0

usage() {
    cat <<EOF
Usage: $0 [options]

Deploy the staged OBS Vision Filter plugin for the system OBS installation.

Options:
  --build       Rebuild and refresh the local stage before deploying
  --dry-run     Show actions without changing the system
  --uninstall   Remove the deployed plugin and its data
  --no-backup   Do not back up an existing deployment
  -h, --help    Show this help

Default destinations:
  Plugin: $PLUGIN_DEST
  Data:   $DATA_DEST
EOF
}

log() {
    printf '[obs-vision-filter] %s\n' "$*"
}

run() {
    if (( DRY_RUN )); then
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

require_root_or_sudo() {
    if (( DRY_RUN )); then
        return
    fi
    if (( EUID != 0 )) && ! command -v sudo >/dev/null 2>&1; then
        printf 'Error: this deployment needs root privileges or sudo.\n' >&2
        exit 1
    fi
}

root_run() {
    if (( EUID == 0 )); then
        run "$@"
    else
        run sudo "$@"
    fi
}

while (($#)); do
    case "$1" in
        --build) BUILD_FIRST=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --no-backup) NO_BACKUP=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Error: unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

require_root_or_sudo

if (( UNINSTALL )); then
    log "Removing deployed plugin and data"
    root_run rm -f "$PLUGIN_DEST"
    root_run rm -rf "$DATA_DEST"
    log "Uninstall complete"
    exit 0
fi

if (( BUILD_FIRST )); then
    command -v cmake >/dev/null 2>&1 || {
        printf 'Error: cmake is required for --build.\n' >&2
        exit 1
    }
    log "Building release artifact"
    run cmake -S "$PROJECT_DIR" -B "$PROJECT_DIR/build" -DCMAKE_BUILD_TYPE=Release
    run cmake --build "$PROJECT_DIR/build" -j"$(nproc)"
    run rm -rf "$STAGE_DIR"
    run mkdir -p "$STAGE_DIR/lib/obs-plugins" \
        "$STAGE_DIR/share/obs/obs-plugins/obs-vision-filter"
    run cp "$PROJECT_DIR/build/obs-vision-filter.so" \
        "$STAGE_DIR/lib/obs-plugins/obs-vision-filter.so"
    run cp -a "$PROJECT_DIR/data/." \
        "$STAGE_DIR/share/obs/obs-plugins/obs-vision-filter/"
fi

if [[ ! -f "$STAGE_DIR/lib/obs-plugins/obs-vision-filter.so" ]]; then
    printf 'Error: staged plugin not found: %s\n' \
        "$STAGE_DIR/lib/obs-plugins/obs-vision-filter.so" >&2
    printf 'Build it first or rerun with --build.\n' >&2
    exit 1
fi

if [[ ! -d "$STAGE_DIR/share/obs/obs-plugins/obs-vision-filter" ]]; then
    printf 'Error: staged plugin data not found: %s\n' \
        "$STAGE_DIR/share/obs/obs-plugins/obs-vision-filter" >&2
    exit 1
fi

if command -v pgrep >/dev/null 2>&1 && pgrep -x obs >/dev/null 2>&1; then
    printf 'Error: OBS is running. Close OBS before deployment.\n' >&2
    exit 1
fi

if [[ -e "$PLUGIN_DEST" || -e "$DATA_DEST" ]]; then
    if (( NO_BACKUP )); then
        log "Existing deployment found; backup disabled"
    else
        log "Backing up existing deployment to $BACKUP_DIR"
        root_run mkdir -p "$BACKUP_DIR"
        [[ -e "$PLUGIN_DEST" ]] && root_run cp -a "$PLUGIN_DEST" "$BACKUP_DIR/"
        [[ -e "$DATA_DEST" ]] && root_run cp -a "$DATA_DEST" "$BACKUP_DIR/"
    fi
fi

log "Installing plugin binary"
root_run install -Dm755 \
    "$STAGE_DIR/lib/obs-plugins/obs-vision-filter.so" \
    "$PLUGIN_DEST"

log "Installing plugin data"
root_run mkdir -p /usr/share/obs/obs-plugins
root_run rm -rf "$DATA_DEST"
root_run cp -a \
    "$STAGE_DIR/share/obs/obs-plugins/obs-vision-filter" \
    "$DATA_DEST"

log "Deployment complete"
printf '\nRestart OBS and add "Vision contour filter" from the source Filters dialog.\n'
