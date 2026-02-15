"""Human-in-the-Loop snapshot testing via Telegram.

Compares generated images against approved snapshots in an artifacts directory.
Unchanged images pass automatically. Changed or new images are sent to a
Telegram chat for human approval, blocking until a response is received.

This module is self-contained — no imports from other lto_avoid modules.
"""

import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv, set_key

_APPROVAL_WORDS = {"approve", "approved", "yes", "ok", "y", "lgtm", "\U0001f44d"}


def _load_png_as_array(data: bytes) -> np.ndarray:
    """Decode PNG bytes to a float32 RGBA numpy array in [0, 1]."""
    from matplotlib.image import imread

    return imread(io.BytesIO(data), format="png").astype(np.float32)


def compare_images(image_a: bytes, image_b: bytes) -> bool:
    """Check whether two PNG images are byte-identical.

    Args:
        image_a: PNG bytes for the first image.
        image_b: PNG bytes for the second image.

    Returns:
        True if the images are exactly identical, False otherwise.
    """
    return image_a == image_b


def make_diff_image(image_a: bytes, image_b: bytes) -> bytes:
    """Create a PNG visualization highlighting pixel differences.

    Produces a side-by-side comparison with the diff amplified 5x in the
    center panel.

    Args:
        image_a: PNG bytes for the old (approved) image.
        image_b: PNG bytes for the new (candidate) image.

    Returns:
        PNG bytes of the diff visualization.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr_a = _load_png_as_array(image_a)
    arr_b = _load_png_as_array(image_b)

    # If shapes differ, just show them side by side
    if arr_a.shape != arr_b.shape:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(arr_a)
        ax1.set_title("Approved")
        ax2.imshow(arr_b)
        ax2.set_title("New")
    else:
        diff = np.abs(arr_a - arr_b)
        # Amplify diff for visibility (clamp to [0, 1])
        diff_amplified = np.clip(diff * 5.0, 0.0, 1.0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        ax1.imshow(arr_a)
        ax1.set_title("Approved")
        ax2.imshow(diff_amplified)
        ax2.set_title("Diff (5x amplified)")
        ax3.imshow(arr_b)
        ax3.set_title("New")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _get_bot_token() -> str:
    """Return the Telegram bot token from environment, or raise."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set. "
            "Set it in .env or as an environment variable."
        )
    return token


def _get_chat_id(token: str) -> str:
    """Return the Telegram chat ID, auto-discovering on first run.

    If TELEGRAM_CHAT_ID is not set, prints instructions and polls
    getUpdates until a /start message arrives, then saves the chat_id
    to .env.
    """
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if chat_id:
        return chat_id

    print(
        "\n"
        "TELEGRAM_CHAT_ID not set. To auto-discover:\n"
        "  1. Open your Telegram bot\n"
        "  2. Send /start\n"
        "  Waiting for /start message...\n"
    )

    offset = 0
    while True:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"offset": offset, "timeout": 60},
            timeout=70,
        )
        resp.raise_for_status()
        updates = resp.json().get("result", [])
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            if msg.get("text", "").strip() == "/start":
                discovered_id = str(msg["chat"]["id"])
                os.environ["TELEGRAM_CHAT_ID"] = discovered_id
                # Try to persist to .env
                env_path = Path.cwd() / ".env"
                if env_path.exists():
                    set_key(str(env_path), "TELEGRAM_CHAT_ID", discovered_id)
                print(f"Discovered chat_id: {discovered_id}")
                return discovered_id


def _send_photo(token: str, chat_id: str, photo: bytes, caption: str) -> int:
    """Send a photo to the Telegram chat. Returns the message_id."""
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendPhoto",
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": ("image.png", io.BytesIO(photo), "image/png")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["result"]["message_id"]


def _drain_updates(token: str) -> int:
    """Consume all pending updates and return the next offset.

    This ensures that old messages (like /start or replies to previous
    snapshots) don't get picked up by a subsequent _poll_reply call.
    """
    offset = 0
    while True:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"offset": offset, "timeout": 0},
            timeout=10,
        )
        resp.raise_for_status()
        updates = resp.json().get("result", [])
        if not updates:
            return offset
        offset = updates[-1]["update_id"] + 1


_POLL_TIMEOUT_S = 300  # 5 minutes


def _poll_reply(token: str, chat_id: str) -> str:
    """Poll getUpdates until a text reply arrives in the chat. Returns the text.

    Drains all pending updates first so only fresh messages are considered.
    Times out after _POLL_TIMEOUT_S seconds.
    """
    offset = _drain_updates(token)
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"No Telegram reply received within {_POLL_TIMEOUT_S}s"
            )
        poll_timeout = min(60, int(remaining))
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"offset": offset, "timeout": poll_timeout},
            timeout=poll_timeout + 10,
        )
        resp.raise_for_status()
        updates = resp.json().get("result", [])
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            if str(msg.get("chat", {}).get("id")) == chat_id and msg.get("text"):
                return msg["text"].strip()
        time.sleep(0.5)


def assert_human_in_the_loop(
    name: str,
    image: bytes | Path,
    artifacts_dir: Path,
) -> None:
    """Compare a generated image against the approved snapshot, requesting
    human review via Telegram if the image has changed.

    Args:
        name: Filename for the snapshot (e.g. "scenario_single_circle.png").
        image: Raw PNG bytes or a Path to the generated PNG file.
        artifacts_dir: Directory where approved snapshots are stored.

    Raises:
        AssertionError: If the human rejects the image, with their feedback
            as the message.
    """
    # Read image bytes
    if isinstance(image, Path):
        image_bytes = image.read_bytes()
    else:
        image_bytes = image

    snapshot_path = artifacts_dir / name

    # Compare against existing approved snapshot
    if snapshot_path.exists():
        approved_bytes = snapshot_path.read_bytes()
        if compare_images(approved_bytes, image_bytes):
            print(f"Snapshot '{name}' unchanged, skipping review")
            return

    # Image is new or changed — need human review
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        # No Telegram configured — save directly (non-interactive fallback)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(image_bytes)
        print(f"Snapshot '{name}' saved (no Telegram configured)")
        return

    chat_id = _get_chat_id(token)

    # Determine caption
    is_new = not snapshot_path.exists()
    caption = f"\U0001f195 New: {name}" if is_new else f"\U0001f504 Changed: {name}"

    # Send the new image
    msg_id = _send_photo(token, chat_id, image_bytes, caption)

    # If changed, also send a diff visualization
    if not is_new:
        diff_png = make_diff_image(approved_bytes, image_bytes)
        _send_photo(token, chat_id, diff_png, f"Diff: {name}")

    # Poll for reply
    reply = _poll_reply(token, chat_id)

    if reply.lower() in _APPROVAL_WORDS:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(image_bytes)
        print(f"Snapshot '{name}' approved and saved")
        return

    raise AssertionError(f"Snapshot '{name}' rejected: {reply}")
