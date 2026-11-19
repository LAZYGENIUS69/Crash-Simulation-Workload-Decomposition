# Web crash studio (Three.js)

This folder is the interactive visualiser that now lives in the Lovable app.
It is fully self-contained: open `studio/index.html` in a browser, or serve the
folder with `python3 -m http.server` from inside `web/`.

- `studio/index.html`, `studio.css`, `studio.js` - the workspace UI (viewport, tabs, dock)
- `studio/three.min.js`, `studio/OrbitControls.js`, `studio/GLTFLoader.js` - three.js r128, vendored locally
- `studio/models/suv.glb` - CC0 SUV (Kenney Car Kit, see `studio/models/CREDITS.md`)
- `studio/data/*.csv`, `studio/data/hw.json` - the recorded solver runs the studio replays

Every number shown in the studio is read from `data/`, i.e. from the recorded
C++/OpenMP runs in `../results/`. The page does not solve the crash itself; it
replays a recorded run and rescales playback for the speed/mass sliders.

The full Lovable app (React + TanStack Start) that embeds this studio is in
`../lovable-app/`.
