"""Screenshot card HTML at 1080x1350.

Chromium must be given the proxy explicitly. It does not inherit HTTPS_PROXY
the way curl does, so without this every webfont silently fails and the cards
render in Georgia and Arial instead of the faces they specify. Nothing errors:
the text just quietly looks wrong.
"""
import os, sys
from playwright.sync_api import sync_playwright

files = [f for f in sys.argv[1:] if not f.startswith("--")]
schemes = ("light",) if "--light-only" in sys.argv else ("light", "dark")
proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                           proxy={"server": proxy} if proxy else None,
                           args=["--ignore-certificate-errors"])
    for scheme in schemes:
        ctx = b.new_context(viewport={"width": 1080, "height": 1350}, device_scale_factor=2,
                            color_scheme=scheme, ignore_https_errors=True)
        p = ctx.new_page()
        for f in files:
            p.goto("file://" + f, wait_until="networkidle")
            p.evaluate("document.fonts.ready")
            p.wait_for_timeout(900)
            # Fail loudly rather than ship a fallback-font card.
            n = p.evaluate("document.fonts.size")
            if n == 0:
                print(f"  WARNING no webfonts loaded for {f.split('/')[-1]}", flush=True)
            out = f[:-5] + ("_dark.png" if scheme == "dark" else ".png")
            p.locator(".card").screenshot(path=out)
            print(scheme, out, flush=True)
        ctx.close()
    b.close()
