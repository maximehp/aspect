"""Measure moving-room latency through Chrome's compositor frame-swap timestamp."""

import argparse
import base64
import io
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from contextlib import closing

import websocket
from PIL import Image


def pose_marker(png):
    # The two complementary rows reject page contents, scaled edges and stale clears.
    with Image.open(io.BytesIO(base64.b64decode(png))) as image:
        image = image.convert("RGB")

        def cell(k, y):
            return image.getpixel(
                (
                    round(image.width * (2 * k + 1) / 64),
                    round(image.height * (2 * y + 1) / 32),
                )
            )[0]

        value = 0
        for k in range(8):
            bit = cell(k, 0)
            if min(bit, 255 - bit) > 3 or abs(bit + cell(k, 1) - 255) > 6:
                return None
            value |= (bit > 127) << k
        return value


class BrowserTab:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.swaps = {}

    def send(self, method, params):
        self.sequence += 1
        self.socket.send(
            json.dumps({"id": self.sequence, "method": method, "params": params})
        )
        return self.sequence

    def call(self, method, **params):
        request = self.send(method, params)
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == request:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]
            if message.get("method") == "Page.screencastFrame":
                frame = message["params"]
                self.send("Page.screencastFrameAck", {"sessionId": frame["sessionId"]})
                pose = pose_marker(frame["data"])
                if pose is not None:
                    self.swaps.setdefault(pose, []).append(
                        1000 * frame["metadata"]["timestamp"]
                    )

    def evaluate(self, expression):
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=True,
            returnByValue=True,
        )
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")


SETUP = """(async (mode) => {
  const { LiveRenderer } = await import('/src/room-display.js');
  window.pose = (await import('/src/motion-benchmark.js')).motionPose;
  const canvas = document.createElement('canvas');
  document.body.replaceChildren(canvas);
  canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;display:block';
  window.room = new LiveRenderer(canvas);
  if (mode === 'resident') {
    const { ResidentContacts } = await import('/src/resident-contacts.js');
    room.view.contactBackend = new ResidentContacts(room);
  } else if (mode === 'gpu') {
    const { GpuContactIntegrals } = await import('/src/gpu-contact.js');
    room.view.contactBackend = new GpuContactIntegrals(room);
  }
})"""

REPLAY = """(async (extra) => {
  const times = [], gl = room.gl;
  for (let tick = 0; tick <= 120; tick++) {
    await new Promise(requestAnimationFrame);
    const start = performance.now();
    const {scene, camera} = pose(tick / 60, extra);
    room.applyScene(scene, await room.prepareScene(scene), start);
    room.draw(camera, 1, [3840, 2160]);
    // Stamp this actual framebuffer, so compositor capture can identify its pose.
    gl.enable(gl.SCISSOR_TEST);
    for (let y = 0; y < 2; y++) for (let k = 0; k < 8; k++) {
      const value = ((tick >> k) & 1) ^ y;
      gl.scissor(120*k, 2160 - 135*(y+1), 120, 135);
      gl.clearColor(value, value, value, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.disable(gl.SCISSOR_TEST);
    room.synchronize();
    times.push([performance.timeOrigin + start, performance.timeOrigin + performance.now()]);
  }
  for (let i = 0; i < 4; i++) await new Promise(requestAnimationFrame);
  return times;
})"""


def percentiles(samples):
    if not samples:
        return None
    return {
        "median": round(statistics.median(samples), 2),
        "p95": round(sorted(samples)[math.ceil(0.95 * len(samples)) - 1], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu", "resident"])
    parser.add_argument("extra", type=int, choices=[0, 4, 8])
    parser.add_argument(
        "--port", type=int, default=9222, help="Chrome remote debugging port"
    )
    parser.add_argument("--room", default="http://127.0.0.1:8736")
    args = parser.parse_args()
    api = f"http://127.0.0.1:{args.port}/json/"
    page = urllib.parse.quote(args.room.rstrip("/") + "/motion.html", safe=":/")
    with urllib.request.urlopen(
        urllib.request.Request(api + "new?" + page, method="PUT")
    ) as response:
        tab = json.load(response)
    try:
        with closing(
            websocket.create_connection(
                tab["webSocketDebuggerUrl"], suppress_origin=True, timeout=90
            )
        ) as ws:
            browser = BrowserTab(ws)
            browser.call("Network.enable")
            browser.call("Network.setCacheDisabled", cacheDisabled=True)
            browser.call("Page.enable")
            browser.call("Page.bringToFront")
            deadline = time.monotonic() + 15
            while not browser.evaluate("!!document.querySelector('#room')"):
                if time.monotonic() >= deadline:
                    raise RuntimeError("The room benchmark page did not load")
                time.sleep(0.05)
            browser.evaluate(SETUP + "(" + json.dumps(args.mode) + ")")
            browser.call(
                "Page.startScreencast",
                format="png",
                maxWidth=256,
                maxHeight=256,
                everyNthFrame=1,
            )
            rows = browser.evaluate(REPLAY + "(" + str(args.extra) + ")")
            browser.call("Page.stopScreencast")
            latencies, tails, missing = [], [], []
            for pose, (start, done) in enumerate(rows[1:], 1):
                stamps = [t for t in browser.swaps.get(pose, []) if t >= start]
                if stamps:
                    swap = min(stamps)
                    latencies.append(swap - start)
                    tails.append(swap - done)
                else:
                    missing.append(pose)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "extra": args.extra,
                        "matched": len(latencies),
                        "missing": missing,
                        "startToSwapMs": percentiles(latencies),
                        "syncToSwapMs": percentiles(tails),
                        "beforeSync": sum(t < -1 for t in tails),
                    }
                )
            )
    finally:
        urllib.request.urlopen(api + "close/" + tab["id"]).close()


if __name__ == "__main__":
    main()
