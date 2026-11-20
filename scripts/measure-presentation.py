"""Measure moving-room latency through Chrome's compositor frame-swap timestamp."""

import argparse
import base64
import io
import json
import math
import sys
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
        self.optional_requests = []

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
            if message.get("method") == "Network.requestWillBeSent":
                url = message["params"]["request"]["url"]
                if "/optical-operations.js" in url:
                    self.optional_requests.append(url)
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


SETUP = """(async () => {
  const { LiveRenderer } = await import('/src/room-display.js');
  window.pose = (await import('/src/motion-benchmark.js')).motionPose;
  const canvas = document.createElement('canvas');
  document.body.replaceChildren(canvas);
  canvas.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;display:block';
  window.room = new LiveRenderer(canvas);
})"""

REPLAY = """(async (extra, slats, ticks, digest) => {
  const times = [], gl = room.gl;
  for (const tick of ticks) {
    await new Promise(requestAnimationFrame);
    const start = performance.now();
    const {scene, camera} = pose(tick / 60, extra, slats);
    const solution=await room.prepareScene(scene,camera,[3840,2160]);
    const prepared=performance.now();
    room.applyScene(scene,solution,start);
    room.draw(camera, 1, [3840, 2160]);
    const submitted=performance.now();
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
    times.push([performance.timeOrigin + start, performance.timeOrigin + performance.now(),room.frameStats.scanMs||0,
      room.countedBytes(), solution.counts, solution.newJobs, solution.timings.workerMs,
      solution.timings.assemblyMs, solution.timings.solve, prepared-start, submitted-prepared,
      performance.now()-submitted, solution.history.at(-1), solution.operator.closure,
      solution.maxMass, solution.timings.operatorMs]);
    if(digest){
      const pixels=new Uint8Array(3840*2160*4);
      gl.readPixels(0,0,3840,2160,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
      const hash=await crypto.subtle.digest('SHA-256',pixels);
      times.at(-1).push(Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join(''));
    }
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
    parser.add_argument("extra", type=int, choices=[0, 4, 8])
    parser.add_argument(
        "--port", type=int, default=9222, help="Chrome remote debugging port"
    )
    parser.add_argument("--room", default="http://127.0.0.1:8736")
    parser.add_argument("--slats", type=int, choices=[0, 1, 2], default=0, help="Single slats in motion, or 0 for both full blinds")
    parser.add_argument("--poses", type=int, default=120, choices=range(1,121), metavar="1..120",
                        help="Evenly spaced moving poses from the same two-second replay")
    parser.add_argument("--exclude-optional-optics", action="store_true", help="Block optional optical operation modules")
    parser.add_argument("--digest-frames", action="store_true", help="Hash every 4K framebuffer in a separate image comparison run")
    args = parser.parse_args()
    ticks = [round(120*i/args.poses) for i in range(args.poses+1)]
    api = f"http://127.0.0.1:{args.port}/json/"
    page = urllib.parse.quote(args.room.rstrip("/") + "/motion.html", safe=":/")
    with urllib.request.urlopen(
        urllib.request.Request(api + "new?" + page, method="PUT")
    ) as response:
        tab = json.load(response)
    try:
        with closing(
            websocket.create_connection(
                tab["webSocketDebuggerUrl"], suppress_origin=True, timeout=600
            )
        ) as ws:
            browser = BrowserTab(ws)
            browser.call("Network.enable")
            browser.call("Network.setCacheDisabled", cacheDisabled=True)
            if args.exclude_optional_optics:
                browser.call("Network.setBlockedURLs", urls=["*/optical-operations.js*"])
            browser.call("Page.enable")
            browser.call("Page.bringToFront")
            deadline = time.monotonic() + 15
            while not browser.evaluate("!!document.querySelector('#room')"):
                if time.monotonic() >= deadline:
                    raise RuntimeError("The room benchmark page did not load")
                time.sleep(0.05)
            browser.evaluate(SETUP + "()")
            browser.call(
                "Page.startScreencast",
                format="png",
                maxWidth=256,
                maxHeight=256,
                everyNthFrame=1,
            )
            rows = []
            for tick in ticks:
                row, = browser.evaluate(REPLAY + "(" + ",".join(map(json.dumps, [args.extra, args.slats, [tick], args.digest_frames])) + ")")
                rows.append(row)
                print(json.dumps({"tick": tick, "gpuMs": row[1]-row[0], "field": row[12]}), file=sys.stderr, flush=True)
            browser.call("Page.stopScreencast")
            latencies, completed, tails, missing = [], [], [], []
            for pose, (start, done, *_) in zip(ticks[1:], rows[1:]):
                stamps = [t for t in browser.swaps.get(pose, []) if t >= start]
                if stamps:
                    swap = min(stamps)
                    latencies.append(swap - start)
                    completed.append(max(swap, done) - start)
                    tails.append(swap - done)
                else:
                    missing.append(pose)
            print(
                json.dumps(
                    {
                        "renderer": "diffuse-field",
                        "extra": args.extra,
                        "slats": args.slats,
                        "matched": len(latencies),
                        "optionalOpticsRequests": browser.optional_requests,
                        **({"frameDigests": [r[-1] for r in rows]} if args.digest_frames else {}),
                        "missing": missing,
                        "startToSwapMs": percentiles(latencies),
                        "throughBothMs": percentiles(completed),
                        "finishedMs": percentiles([r[1]-r[0] for r in rows[1:]]),
                        "cameraMs": percentiles([r[2] for r in rows[1:]]),
                        "peakCountedMiB": max(r[3] for r in rows)/1048576,
                        "meanWorkCounts": [statistics.mean(v) for v in zip(*(r[4] for r in rows[1:]))],
                        "overFrameBudget": {str(b): sum(t > b for t in completed) for b in (1, 1000/60)},
                        "meanGpuApertures": statistics.mean(r[5] for r in rows[1:]),
                        "worstSpatialResidual": max(r[12]["maximum"] for r in rows[1:]),
                        "framesExceedingSpatialTolerance": sum(r[12]["failed"] > 0 for r in rows[1:]),
                        "worstEquilibriumClosure": max(r[13] for r in rows[1:]),
                        "maximumUnitFieldGain": max(r[14] for r in rows[1:]),
                        "workerMs": percentiles([r[6] for r in rows[1:]]),
                        "assemblyMs": percentiles([r[7] for r in rows[1:]]),
                        "solveMs": percentiles([r[8] for r in rows[1:]]),
                        "operatorConstructionMs": percentiles([r[15] for r in rows[1:]]),
                        "preparationMs": percentiles([r[9] for r in rows[1:]]),
                        "applyAndDrawMs": percentiles([r[10] for r in rows[1:]]),
                        "markAndFinishMs": percentiles([r[11] for r in rows[1:]]),
                        "syncToSwapMs": percentiles(tails),
                        "beforeSync": sum(t < -1 for t in tails),
                    }
                )
            )
    finally:
        urllib.request.urlopen(api + "close/" + tab["id"]).close()


if __name__ == "__main__":
    main()
