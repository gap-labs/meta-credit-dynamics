#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish_release.sh <tag> [options]

Options:
  --target <path>       Target public repository path (used in target-mode=existing)
                        (default: /home/andreas/prj/meta-credit-dynamics)
  --target-mode <mode>  existing | fresh-clone (default: existing)
  --remote <url>        Public remote URL for fresh-clone mode
                        (default: git@github.com:gap-labs/meta-credit-dynamics.git)
  --branch <name>       Branch to clone/push in fresh-clone mode (default: main)
  --workdir <path>      Work directory for fresh-clone mode
                        (default: temporary directory)
  --whitelist <path>    Allowlist file (default: scripts/publish_whitelist.txt)
  --source <path>       Source private repository path (default: repo root of this script)
  --allow-dirty-source  Skip clean-working-tree check in source repo
  --log-file <path>     Log file path (default: <source>/tmp/publish_logs/publish_*.log)
  --public-slug <slug>  Public GitHub slug owner/repo used in generated demo notebook
                        (default: gap-labs/meta-credit-dynamics)
  --no-generate-team-demo
                        Disable auto generation of notebooks/team_demo.ipynb
                        from notebooks/team.ipynb in the tag snapshot
  --no-commit           Do not create a commit in target repo after apply
  --commit-message <m>  Commit message for target repo publish commit
  --public-tag <tag>    Tag name to create in target repo (default: source tag)
  --no-public-tag       Do not create/update a tag in target repo
  --retag               Allow replacing an existing public tag in target repo
  --no-push-prompt      Skip interactive push question
  --yes-push            Push automatically without prompt (implies --apply)
  --apply               Execute copy (default: dry-run only)
  --clean-target        Remove everything in target repo except .git before copying
  -h, --help            Show help

Behavior:
  - Exports a snapshot from the given tag using git archive (no git history transfer).
  - By default auto-generates notebooks/team_demo.ipynb from notebooks/team.ipynb.
  - Can publish into an existing checkout or a fresh clone of public main.
  - Writes a detailed run log to disk.
  - On apply, can commit + public-tag and asks at the end: Push [y|N]: (default N).
  - Copies only allowlisted paths from the snapshot to the target repository.
  - Dry-run by default; use --apply to perform changes.

Example:
  scripts/publish_release.sh v0.7.19 \
    --target /home/andreas/prj/meta-credit-dynamics \
    --whitelist scripts/publish_whitelist.txt \
    --apply --clean-target

  scripts/publish_release.sh v0.7.19 \
    --target-mode fresh-clone \
    --remote git@github.com:gap-labs/meta-credit-dynamics.git \
    --branch main \
    --apply
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TAG=""
SOURCE_REPO="${DEFAULT_SOURCE_REPO}"
TARGET_REPO="/home/andreas/prj/meta-credit-dynamics"
TARGET_MODE="existing"
TARGET_REMOTE="git@github.com:gap-labs/meta-credit-dynamics.git"
TARGET_BRANCH="main"
WORKDIR=""
WHITELIST_FILE="${SCRIPT_DIR}/publish_whitelist.txt"
LOG_FILE=""
PUBLIC_SLUG="gap-labs/meta-credit-dynamics"
GENERATE_TEAM_DEMO=1
REQUIRE_CLEAN_SOURCE=1
AUTO_COMMIT=1
COMMIT_MESSAGE=""
PUBLIC_TAG=""
CREATE_PUBLIC_TAG=1
ALLOW_RETAG=0
PUSH_PROMPT=1
AUTO_PUSH=0
APPLY=0
CLEAN_TARGET=0
TARGET_WORK_REPO=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --target)
      [[ $# -ge 2 ]] || fail "--target requires a value"
      TARGET_REPO="$2"
      shift 2
      ;;
    --target-mode)
      [[ $# -ge 2 ]] || fail "--target-mode requires a value"
      TARGET_MODE="$2"
      shift 2
      ;;
    --remote)
      [[ $# -ge 2 ]] || fail "--remote requires a value"
      TARGET_REMOTE="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || fail "--branch requires a value"
      TARGET_BRANCH="$2"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 ]] || fail "--workdir requires a value"
      WORKDIR="$2"
      shift 2
      ;;
    --whitelist)
      [[ $# -ge 2 ]] || fail "--whitelist requires a value"
      WHITELIST_FILE="$2"
      shift 2
      ;;
    --source)
      [[ $# -ge 2 ]] || fail "--source requires a value"
      SOURCE_REPO="$2"
      shift 2
      ;;
    --allow-dirty-source)
      REQUIRE_CLEAN_SOURCE=0
      shift
      ;;
    --log-file)
      [[ $# -ge 2 ]] || fail "--log-file requires a value"
      LOG_FILE="$2"
      shift 2
      ;;
    --public-slug)
      [[ $# -ge 2 ]] || fail "--public-slug requires a value"
      PUBLIC_SLUG="$2"
      shift 2
      ;;
    --no-generate-team-demo)
      GENERATE_TEAM_DEMO=0
      shift
      ;;
    --no-commit)
      AUTO_COMMIT=0
      shift
      ;;
    --commit-message)
      [[ $# -ge 2 ]] || fail "--commit-message requires a value"
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --public-tag)
      [[ $# -ge 2 ]] || fail "--public-tag requires a value"
      PUBLIC_TAG="$2"
      CREATE_PUBLIC_TAG=1
      shift 2
      ;;
    --no-public-tag)
      CREATE_PUBLIC_TAG=0
      shift
      ;;
    --retag)
      ALLOW_RETAG=1
      shift
      ;;
    --no-push-prompt)
      PUSH_PROMPT=0
      shift
      ;;
    --yes-push)
      AUTO_PUSH=1
      APPLY=1
      PUSH_PROMPT=0
      shift
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --clean-target)
      CLEAN_TARGET=1
      shift
      ;;
    --*)
      fail "Unknown option: $1"
      ;;
    *)
      if [[ -z "$TAG" ]]; then
        TAG="$1"
        shift
      else
        fail "Unexpected positional argument: $1"
      fi
      ;;
  esac
done

[[ -n "$TAG" ]] || {
  usage
  fail "Tag is required"
}

[[ -d "$SOURCE_REPO/.git" ]] || fail "Source repo is not a git repository: $SOURCE_REPO"
[[ -f "$WHITELIST_FILE" ]] || fail "Whitelist file not found: $WHITELIST_FILE"
[[ "$PUBLIC_SLUG" == */* ]] || fail "--public-slug must have format owner/repo"
[[ "$TARGET_MODE" == "existing" || "$TARGET_MODE" == "fresh-clone" ]] || fail "--target-mode must be existing|fresh-clone"

if [[ "$REQUIRE_CLEAN_SOURCE" -eq 1 ]]; then
  if [[ -n "$(git -C "$SOURCE_REPO" status --porcelain)" ]]; then
    fail "Working tree not clean in source repo: $SOURCE_REPO (commit/stash changes or use --allow-dirty-source)"
  fi
fi

git -C "$SOURCE_REPO" rev-parse --verify "${TAG}^{commit}" >/dev/null 2>&1 || fail "Tag not found or not a commit: $TAG"

if [[ -z "$PUBLIC_TAG" ]]; then
  PUBLIC_TAG="$TAG"
fi

if [[ -z "$COMMIT_MESSAGE" ]]; then
  COMMIT_MESSAGE="chore: publish snapshot ${TAG}"
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="${SOURCE_REPO}/tmp/publish_logs/publish_$(date +%Y%m%d_%H%M%S)_${TAG}.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run log: $LOG_FILE"
echo "Source repo: $SOURCE_REPO"
echo "Target mode: $TARGET_MODE"
echo "Tag: $TAG"

declare -a ALLOWLIST=()
while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="${raw%%#*}"
  line="$(trim "$line")"
  [[ -z "$line" ]] && continue

  line="${line%/}"
  [[ -z "$line" ]] && continue

  [[ "$line" = /* ]] && fail "Whitelist path must be relative: $line"
  [[ "$line" == *".."* ]] && fail "Whitelist path must not contain '..': $line"

  ALLOWLIST+=("$line")
done < "$WHITELIST_FILE"

[[ ${#ALLOWLIST[@]} -gt 0 ]] || fail "Whitelist resolved to zero paths"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
snapshot_dir="$tmpdir/snapshot"
mkdir -p "$snapshot_dir"

if [[ "$TARGET_MODE" == "existing" ]]; then
  [[ -d "$TARGET_REPO/.git" ]] || fail "Target repo is not a git repository: $TARGET_REPO"
  TARGET_WORK_REPO="$TARGET_REPO"
else
  if [[ -n "$WORKDIR" ]]; then
    TARGET_WORK_REPO="$WORKDIR"
    [[ ! -e "$TARGET_WORK_REPO" ]] || fail "--workdir already exists, refusing to overwrite: $TARGET_WORK_REPO"
  else
    TARGET_WORK_REPO="$tmpdir/public_repo"
  fi
  echo "[0/6] Cloning fresh target: $TARGET_REMOTE (branch $TARGET_BRANCH)"
  git clone --depth 1 --branch "$TARGET_BRANCH" "$TARGET_REMOTE" "$TARGET_WORK_REPO"
fi

echo "[1/6] Exporting snapshot from tag '$TAG'"
git -C "$SOURCE_REPO" archive --format=tar "$TAG" | tar -xf - -C "$snapshot_dir"

if [[ ! -f "$snapshot_dir/requirements.txt" ]]; then
  echo "WARNING: requirements.txt missing in tag snapshot; Colab setup may fail."
fi

req_in_allowlist=0
for rel in "${ALLOWLIST[@]}"; do
  if [[ "$rel" == "requirements.txt" ]]; then
    req_in_allowlist=1
    break
  fi
done
if [[ "$req_in_allowlist" -eq 0 ]]; then
  echo "WARNING: requirements.txt not in whitelist; Colab setup in published demo may fail."
fi

if [[ "$GENERATE_TEAM_DEMO" -eq 1 ]]; then
  echo "[2/6] Generating notebooks/team_demo.ipynb from notebooks/team.ipynb"
  team_src="$snapshot_dir/notebooks/team.ipynb"
  team_dst="$snapshot_dir/notebooks/team_demo.ipynb"
  [[ -f "$team_src" ]] || fail "Missing source notebook for demo generation: notebooks/team.ipynb"

  python3 - "$team_src" "$team_dst" "$PUBLIC_SLUG" <<'PY'
import json
import sys

src, dst, slug = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src, "r", encoding="utf-8") as f:
  notebook = json.load(f)

colab_team_demo = f"https://colab.research.google.com/github/{slug}/blob/main/notebooks/team_demo.ipynb"
repo_clone_url = f"https://github.com/{slug}.git"

for cell in notebook.get("cells", []):
  if cell.get("cell_type") == "code":
    cell["execution_count"] = None
    cell["outputs"] = []

  source = cell.get("source")
  if isinstance(source, list):
    lines = source
    was_list = True
  elif isinstance(source, str):
    lines = [source]
    was_list = False
  else:
    continue

  updated = []
  for line in lines:
    line = line.replace(
      "https://colab.research.google.com/github/andreas-bille/meta-credit-dynamics/blob/main/notebooks/team.ipynb",
      colab_team_demo,
    )
    line = line.replace(
      "https://colab.research.google.com/github/gap-labs/meta-credit-dynamics/blob/main/notebooks/team.ipynb",
      colab_team_demo,
    )
    line = line.replace(
      "https://colab.research.google.com/github/andreas-bille/meta-credit-dynamics/blob/main/notebooks/team_demo.ipynb",
      colab_team_demo,
    )
    line = line.replace(
      "https://colab.research.google.com/github/gap-labs/meta-credit-dynamics/blob/main/notebooks/team_demo.ipynb",
      colab_team_demo,
    )
    line = line.replace("https://github.com/andreas-bille/meta-credit-dynamics.git", repo_clone_url)
    line = line.replace("https://github.com/gap-labs/meta-credit-dynamics.git", repo_clone_url)
    updated.append(line)

  cell["source"] = updated if was_list else "".join(updated)

with open(dst, "w", encoding="utf-8") as f:
  json.dump(notebook, f, ensure_ascii=False, indent=4)
  f.write("\n")
PY
else
  echo "[2/6] Skipping team_demo generation (--no-generate-team-demo)"
fi

echo "[3/6] Validating allowlist entries (${#ALLOWLIST[@]})"
for rel in "${ALLOWLIST[@]}"; do
  src="$snapshot_dir/$rel"
  [[ -e "$src" ]] || fail "Allowlist path not found in tag snapshot: $rel"
  if [[ -d "$src" ]]; then
    echo "  + dir  $rel"
  else
    echo "  + file $rel"
  fi
done

if [[ "$APPLY" -eq 0 ]]; then
  echo "[4/6] Dry-run mode: no files changed"
  echo "      Target repo was: $TARGET_WORK_REPO"
  echo "      Re-run with --apply to copy into target repo"
  exit 0
fi

if [[ "$CLEAN_TARGET" -eq 1 ]]; then
  echo "[4/6] Cleaning target repository (except .git): $TARGET_WORK_REPO"
  find "$TARGET_WORK_REPO" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
else
  echo "[4/6] Updating allowlisted paths only (no full clean)"
fi

echo "[5/6] Copying allowlisted snapshot paths into target"
for rel in "${ALLOWLIST[@]}"; do
  src="$snapshot_dir/$rel"
  dst="$TARGET_WORK_REPO/$rel"

  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  cp -a "$src" "$dst"
done

echo "Done. Target repo status:"
git -C "$TARGET_WORK_REPO" --no-pager status --short

if [[ "$AUTO_COMMIT" -eq 1 ]]; then
  if [[ -n "$(git -C "$TARGET_WORK_REPO" status --porcelain)" ]]; then
    echo "[6/6] Creating commit in target repo"
    git -C "$TARGET_WORK_REPO" add -A
    git -C "$TARGET_WORK_REPO" commit -m "$COMMIT_MESSAGE"

    if [[ "$CREATE_PUBLIC_TAG" -eq 1 ]]; then
      if git -C "$TARGET_WORK_REPO" rev-parse -q --verify "refs/tags/${PUBLIC_TAG}" >/dev/null 2>&1; then
        if [[ "$ALLOW_RETAG" -eq 1 ]]; then
          git -C "$TARGET_WORK_REPO" tag -fa "$PUBLIC_TAG" -m "public release ${PUBLIC_TAG}"
        else
          fail "Public tag already exists in target repo: ${PUBLIC_TAG} (use --retag or --no-public-tag)"
        fi
      else
        git -C "$TARGET_WORK_REPO" tag -a "$PUBLIC_TAG" -m "public release ${PUBLIC_TAG}"
      fi
    fi

    do_push=0
    if [[ "$AUTO_PUSH" -eq 1 ]]; then
      do_push=1
    elif [[ "$PUSH_PROMPT" -eq 1 ]]; then
      if [[ -t 0 ]]; then
        read -r -p "Push [y|N]: " reply
        if [[ "${reply:-N}" =~ ^[Yy]$ ]]; then
          do_push=1
        fi
      else
        echo "No TTY available for push prompt; skipping push by default."
      fi
    fi

    if [[ "$do_push" -eq 1 ]]; then
      branch_name="$(git -C "$TARGET_WORK_REPO" rev-parse --abbrev-ref HEAD)"
      git -C "$TARGET_WORK_REPO" push origin "$branch_name"
      if [[ "$CREATE_PUBLIC_TAG" -eq 1 ]]; then
        git -C "$TARGET_WORK_REPO" push origin "$PUBLIC_TAG"
      fi
      echo "Push completed."
    else
      echo "Push skipped."
    fi
  else
    echo "No changes after apply; commit/push skipped."
  fi
else
  echo "Commit disabled (--no-commit)."
fi

echo "Target work repository: $TARGET_WORK_REPO"
