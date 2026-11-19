/* Crash Studio — replay viewer for the recorded C++/OpenMP crash run. Self-contained. */
(() => {
const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const D = {}; // loaded data
const S = { frac: 0, running: false, threads: 8, sched: "static", asm: "Node Gather CSR", mesh: "100k",
  speed: 15.56, mass: 1500, color: "paint", sigMax: 0, Fmax: 1, shading: "lit", stage: 0, peakF: 0, flashed: false };
const STAGES = ["S1 Spring forces", "S2 Force assembly", "S3 Contact", "S4 Integrate", "S5 CFL / dt"];
const STAGE_COL = ["#00c2ff", "#3ddc97", "#f0a020", "#c678dd", "#ff3b3b"];
const THREAD_COL = [0x3ddc97,0x3ddc97,0xf0a020,0xc678dd,0xe5484d,0x2ec4b6,0xffd23f,0xff7eb6,0x9aa5b1,0x7ae582,0xff9f1c,0x5e60ce];

function log(msg) { const c = $("console"); const t = new Date().toLocaleTimeString(); c.textContent += `[${t}] ${msg}\n`; c.scrollTop = c.scrollHeight; }
function csv(text) { const [h, ...rows] = text.trim().split(/\r?\n/); const keys = h.split(",");
  return rows.map(r => { const v = r.split(","); const o = {}; keys.forEach((k, i) => { const n = Number(v[i]); o[k] = v[i] !== "" && !isNaN(n) ? n : v[i]; }); return o; }); }

// ── Scene ──────────────────────────────────────────────
const vp = $("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x8d9094);
scene.fog = new THREE.Fog(0x8d9094, 14, 32);
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.9;
renderer.outputEncoding = THREE.sRGBEncoding;
vp.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = Math.PI / 2 - 0.02;
function resize() { const w = vp.clientWidth, h = vp.clientHeight; if (!w || !h) return; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); }
new ResizeObserver(resize).observe(vp);

scene.add(new THREE.HemisphereLight(0xffffff, 0x6a6c70, 0.45));
const key = new THREE.DirectionalLight(0xffffff, 0.95); key.position.set(5, 9, 6); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); Object.assign(key.shadow.camera, { left: -6, right: 8, top: 6, bottom: -6, near: 0.5, far: 30 }); key.shadow.bias = -0.0004;
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-5, 3, -4); scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.15); rim.position.set(-2, 4, 6); scene.add(rim);

// Studio floor with subtle tile texture
const tile = document.createElement("canvas"); tile.width = tile.height = 256;
{ const g = tile.getContext("2d"); g.fillStyle = "#9c9fa3"; g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 1500; i++) { g.fillStyle = `rgba(0,0,0,${Math.random() * 0.025})`; g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2); }
  g.strokeStyle = "rgba(0,0,0,0.06)"; g.lineWidth = 1.5; g.strokeRect(0, 0, 256, 256); }
const tileTex = new THREE.CanvasTexture(tile); tileTex.wrapS = tileTex.wrapT = THREE.RepeatWrapping; tileTex.repeat.set(20, 20); tileTex.encoding = THREE.sRGBEncoding;
const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ map: tileTex, roughness: 0.9, metalness: 0.0 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const wall = new THREE.Mesh(new THREE.PlaneGeometry(40, 12), new THREE.MeshStandardMaterial({ color: 0x8f9296, roughness: 1 }));
wall.position.set(0, 6, -6); wall.receiveShadow = true; scene.add(wall);

// Rigid barrier
const barrier = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.8, 3.4), new THREE.MeshStandardMaterial({ color: 0x6a6e74, roughness: 0.9 }));
barrier.position.set(-0.4, 0.9, 0); barrier.castShadow = barrier.receiveShadow = true; scene.add(barrier);
const sc = document.createElement("canvas"); sc.width = 512; sc.height = 64;
{ const g = sc.getContext("2d"); g.fillStyle = "#222"; g.fillRect(0, 0, 512, 64); g.fillStyle = "#e8b400";
  for (let x = -64; x < 576; x += 64) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 32, 0); g.lineTo(x - 16, 64); g.lineTo(x - 48, 64); g.fill(); } }
const stripe = new THREE.Mesh(new THREE.PlaneGeometry(3.3, 0.4), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sc) }));
stripe.position.set(0.005, 0.5, 0); stripe.rotation.y = Math.PI / 2; scene.add(stripe);

// ── Vehicle (segmented so every vertex is a "node") ───
const car = new THREE.Group(); scene.add(car);
const parts = []; // {mesh, geo, orig, region, paintArr, points}
const SUBS = ["Bumper / crash box", "Front rails", "Hood / fenders", "A-pillar / windshield", "Cabin / roof", "Body / sills / rear"];
const CAR_START_X = 2.2, CONTACT_X = 0.42, VIS_CRUSH = 0.55, FRONT_X = -0.4, CRUSH_END = 1.4, CAR_LEN = 3.35;
const A_EFF = 0.0024; // m², effective load-path cross-section used for the stress estimate
let BODY = { maxY: 1.6, halfW: 0.9, roofX: 1.1 };
// Split triangles until every edge is shorter than maxLen, so the heatmap and crumple are smooth
function tessellate(geo, maxLen) {
  const pos = geo.attributes.position, uv = geo.attributes.uv, idx = geo.index; // read through accessors: GLB attributes are interleaved
  const cnt = idx ? idx.count : pos.count, at = (i) => idx ? idx.getX(i) : i;
  const P = [], U = []; const m2 = maxLen * maxLen;
  const d2 = (a, b) => (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2;
  const mid = (a, b) => a.map((v, i) => (v + b[i]) / 2);
  function tri(a, b, c, ua, ub, uc, depth) {
    const ab = d2(a, b), bc = d2(b, c), ca = d2(c, a), mx = Math.max(ab, bc, ca);
    if (mx <= m2 || depth > 7) { P.push(...a, ...b, ...c); U.push(...ua, ...ub, ...uc); return; }
    if (mx === ab) { const m = mid(a, b), um = mid(ua, ub); tri(a, m, c, ua, um, uc, depth + 1); tri(m, b, c, um, ub, uc, depth + 1); }
    else if (mx === bc) { const m = mid(b, c), um = mid(ub, uc); tri(a, b, m, ua, ub, um, depth + 1); tri(a, m, c, ua, um, uc, depth + 1); }
    else { const m = mid(c, a), um = mid(uc, ua); tri(a, b, m, ua, ub, um, depth + 1); tri(m, b, c, um, ub, uc, depth + 1); }
  }
  for (let i = 0; i < cnt; i += 3) {
    const v = k => { const j = at(i + k); return [pos.getX(j), pos.getY(j), pos.getZ(j)]; }, u = k => { const j = at(i + k); return uv ? [uv.getX(j), uv.getY(j)] : [0, 0]; };
    tri(v(0), v(1), v(2), u(0), u(1), u(2), 0);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(P, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(U, 2)); return g;
}
function regionOf(x, y, z) {
  const fx = x - FRONT_X, H = BODY.maxY;
  if (fx < 0.28) return 0;
  if (fx < 1.25 && y < 0.38 * H && Math.abs(z) < BODY.halfW * 0.8) return 1;
  if (x < BODY.roofX - 0.35 && y >= 0.38 * H) return 2;
  if (x < BODY.roofX + 0.1 && y >= 0.5 * H) return 3;
  if (y >= 0.62 * H) return 4;
  return 5;
}
// Load-path weight: how strongly each region carries the barrier load (rails and pillars carry most)
const LOAD_W = [1.0, 1.0, 0.55, 0.8, 0.45, 0.6];
// Windshield crack decal
const cc = document.createElement("canvas"); cc.width = cc.height = 256;
{ const g = cc.getContext("2d"); g.strokeStyle = "rgba(255,255,255,.9)"; g.lineWidth = 1.6;
  for (let a = 0; a < Math.PI * 2; a += 0.33) { g.beginPath(); g.moveTo(128, 128); let r = 0, x = 128, y = 128; while (r < 120) { r += 12 + Math.random() * 10; const aa = a + (Math.random() - .5) * .2; x = 128 + Math.cos(aa) * r; y = 128 + Math.sin(aa) * r; g.lineTo(x, y); } g.stroke(); }
  for (let r = 18; r < 110; r += 22) { g.beginPath(); for (let a = 0; a <= Math.PI * 2 + .1; a += .33) g.lineTo(128 + Math.cos(a) * (r + Math.random() * 5), 128 + Math.sin(a) * (r + Math.random() * 5)); g.stroke(); } }
const crackMat = new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(cc), transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide });
const crack = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 0.5), crackMat); car.add(crack);
// Peak stress marker
const peakMark = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 8), new THREE.MeshBasicMaterial({ color: 0xff3b3b, depthTest: false })); peakMark.renderOrder = 5; peakMark.visible = false; car.add(peakMark);
const wheels = [];
// Load the CC0 Kenney SUV, orient it with the front toward the barrier, and bake it into car-local coordinates
function loadCar() {
  return new Promise((resolve, reject) => new THREE.GLTFLoader().load("models/suv.glb", (gltf) => {
    const root = gltf.scene; root.updateMatrixWorld(true);
    const fw = root.getObjectByName("wheel-front-left"); const fz = fw ? new THREE.Vector3().setFromMatrixPosition(fw.matrixWorld).z : 1;
    const box = new THREE.Box3().setFromObject(root); const len = box.max.z - box.min.z; const s = CAR_LEN / len;
    const M = new THREE.Matrix4().makeTranslation(FRONT_X, 0, 0)
      .multiply(new THREE.Matrix4().makeScale(s, s, s))
      .multiply(new THREE.Matrix4().makeRotationY(fz > 0 ? -Math.PI / 2 : Math.PI / 2))
      .multiply(new THREE.Matrix4().makeTranslation(0, -box.min.y, -(fz > 0 ? box.max.z : box.min.z)));
    // after rotation the front (fz side) sits at x = 0, then shifted to FRONT_X
    root.traverse(o => {
      if (!o.isMesh) return; const g = o.geometry.clone().applyMatrix4(new THREE.Matrix4().multiplyMatrices(M, o.matrixWorld));
      if (/wheel/.test(o.name)) {
        g.computeBoundingBox(); const c = g.boundingBox.getCenter(new THREE.Vector3()); g.translate(-c.x, -c.y, -c.z);
        const w = new THREE.Group(); w.position.copy(c); const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x1c1d1f, roughness: 0.9 })); m.castShadow = true; w.add(m); car.add(w);
        wheels.push({ g: w, x: c.x, f: /front/.test(o.name) ? 1 : 0 }); return;
      }
      const flat = tessellate(g, 0.12);
      flat.computeBoundingBox(); const bb = flat.boundingBox; BODY = { maxY: bb.max.y, halfW: bb.max.z, roofX: FRONT_X + 0.42 * CAR_LEN };
      // roof front edge: smallest x among high, central vertices
      const pa = flat.attributes.position; let roofX = 99; for (let i = 0; i < pa.count; i++) if (pa.getY(i) > 0.93 * bb.max.y && Math.abs(pa.getZ(i)) < 0.3) roofX = Math.min(roofX, pa.getX(i)); if (roofX < 99) BODY.roofX = roofX;
      // paint colour per vertex: sample the kit colour map, then grade to a matte studio grey (dark parts stay dark)
      const img = o.material.map && o.material.map.image; let sample = null;
      if (img) { const cv = document.createElement("canvas"); cv.width = img.width; cv.height = img.height; const cx = cv.getContext("2d"); cx.drawImage(img, 0, 0); const d = cx.getImageData(0, 0, img.width, img.height).data;
        sample = (u, v) => { const x = Math.min(img.width - 1, Math.max(0, Math.floor(((u % 1) + 1) % 1 * img.width))), y = Math.min(img.height - 1, Math.max(0, Math.floor(((v % 1) + 1) % 1 * img.height))); const k = (y * img.width + x) * 4; return [d[k] / 255, d[k + 1] / 255, d[k + 2] / 255]; }; }
      const uvA = flat.attributes.uv, n = pa.count; const paint = new Float32Array(n * 3), region = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const [r, gg, b] = sample ? sample(uvA.getX(i), uvA.getY(i)) : [0.6, 0.6, 0.6]; const lum = 0.3 * r + 0.59 * gg + 0.11 * b;
        const yy = pa.getY(i) / bb.max.y; const glass = b > r + 0.08 && b > gg + 0.02 && yy > 0.5 && yy < 0.9; const grey = glass ? 0.05 : lum < 0.22 ? 0.3 + lum * 0.3 : 0.42 + lum * 0.1; tmpC.setRGB(grey, grey, grey).convertSRGBToLinear(); paint.set([tmpC.r, tmpC.g, tmpC.b], i * 3);
        region[i] = regionOf(pa.getX(i), pa.getY(i), pa.getZ(i));
      }
      flat.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3)); flat.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.0, roughness: 0.78, flatShading: true });
      const mesh = new THREE.Mesh(flat, mat); mesh.castShadow = mesh.receiveShadow = true; car.add(mesh);
      const pts = new THREE.Points(flat, new THREE.PointsMaterial({ size: 0.02, vertexColors: true })); pts.visible = false; mesh.add(pts);
      parts.push({ mesh, geo: flat, orig: pa.clone(), region, paintArr: paint, points: pts, plastic: new Float32Array(n) });
    });
    crack.position.set(BODY.roofX - 0.2, 0.84 * BODY.maxY, 0); crack.rotation.set(0, -Math.PI / 2, 0); crack.rotateX(-0.75);
    let nodes = 0; parts.forEach(p => nodes += p.geo.attributes.position.count); $("p-drawn").textContent = nodes.toLocaleString();
    resolve();
  }, undefined, reject));
}
// Debris (deterministic from time so scrubbing works)
const debris = [];
const debMats = [new THREE.MeshStandardMaterial({ color: 0x1a1d22 }), new THREE.MeshStandardMaterial({ color: 0xb33a2c }), new THREE.MeshStandardMaterial({ color: 0x9ea3aa, metalness: .6, roughness: .4 }), new THREE.MeshStandardMaterial({ color: 0xdfe6ee, transparent: true, opacity: .7 })];
let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let i = 0; i < 90; i++) {
  const s = 0.02 + rnd() * 0.07; const m = new THREE.Mesh(new THREE.BoxGeometry(s, s * (0.3 + rnd()), s * (0.5 + rnd())), debMats[i % 4]);
  m.castShadow = true; m.visible = false; scene.add(m);
  debris.push({ m, p0: new THREE.Vector3(0.05, 0.3 + rnd() * 0.7, (rnd() - .5) * 1.6), v: new THREE.Vector3(0.6 + rnd() * 2.2, 0.8 + rnd() * 2.6, (rnd() - .5) * 3.4), w: new THREE.Vector3(rnd() * 12, rnd() * 12, rnd() * 12) });
}

// ── Partition / colouring ─────────────────────────────
function ownerOf(u, p, sched) {
  if (p === 1) return 0;
  if (sched === "static") return Math.min(p - 1, Math.floor(u * p));
  if (sched === "dynamic") { const chunks = p * 4; const c = Math.min(chunks - 1, Math.floor(u * chunks)); return (c * 7 + (c >> 1)) % p; }
  // guided: chunk = remaining / p, round-robin
  let start = 0, rem = 1, k = 0; while (true) { const sz = Math.max(rem / p, 0.02); if (u < start + sz || rem <= 0.02) return k % p; start += sz; rem -= sz; k++; }
}
function isBoundary(u, p, sched) { const e = 0.008; return ownerOf(Math.max(0, u - e), p, sched) !== ownerOf(Math.min(1, u + e), p, sched); }
const tmpC = new THREE.Color();
function strainColor(s, c) { // blue → green → yellow → red
  s = Math.min(1, Math.max(0, s)); if (s < 0.01) return c.setRGB(0.5, 0.5, 0.5);
  const stops = [[0.7, 0.66, 0.45], [0.98, 0.85, 0.15], [0.95, 0.85, 0.2], [0.9, 0.2, 0.15]]; const f = s * 3, i = Math.min(2, Math.floor(f)), t = f - i;
  return c.setRGB(...stops[i].map((v, k) => v + (stops[i + 1][k] - v) * t));
}
function stressColor(s, c) { // FEA ramp: blue → cyan → green → yellow → red
  s = Math.min(1, Math.max(0, s));
  const st = [[0.5, 0.5, 0.5], [0.75, 0.7, 0.4], [0.98, 0.85, 0.15], [0.98, 0.5, 0.08], [0.9, 0.1, 0.08]]; const f = s * 4, i = Math.min(3, Math.floor(f)), t = f - i;
  return c.setRGB(...st[i].map((v, k) => v + (st[i + 1][k] - v) * t));
}
function applyColors(strains, stress) {
  const minX = FRONT_X - 0.2, maxX = FRONT_X + CAR_LEN + 0.1;
  parts.forEach((p, pi) => {
    const col = p.geo.attributes.color;
    for (let i = 0; i < col.count; i++) {
      const lx = p.orig.getX(i);
      if (S.color === "paint") { col.setXYZ(i, p.paintArr[i * 3], p.paintArr[i * 3 + 1], p.paintArr[i * 3 + 2]); continue; }
      if (S.color === "stress") stressColor(stress[pi][i], tmpC);
      else if (S.color === "strain") strainColor(strains[pi][i], tmpC);
      else if (S.color === "subsystem") tmpC.setHex(THREAD_COL[p.region[i]]);
      else { const u = (lx - minX) / (maxX - minX); tmpC.setHex(THREAD_COL[ownerOf(u, S.threads, S.sched)]); if (isBoundary(u, S.threads, S.sched)) tmpC.setRGB(1, 1, 1); }
      tmpC.convertSRGBToLinear(); col.setXYZ(i, tmpC.r, tmpC.g, tmpC.b);
    }
    col.needsUpdate = true;
  });
}
function updateLegend() {
  const L = $("legend");
  if (S.color === "thread") L.innerHTML = `<b>Thread owner · ${S.sched}</b><br>` + Array.from({ length: S.threads }, (_, i) => `<span><i style="background:#${THREAD_COL[i].toString(16).padStart(6, "0")}"></i>T${i}</span>`).join("") + `<span><i style="background:#fff"></i>partition boundary</span>`;
  else if (S.color === "stress") { const ticks = [0, .25, .5, .75, 1].map(f => `<em>${(f * S.sigMax).toFixed(0)}</em>`).join("");
    L.innerHTML = `<b>Von Mises stress (est.) · MPa</b><div class="ramp"></div><div class="ticks">${ticks}</div><div class="peakline">Peak now: <b id="lg-peak">0</b> MPa · <span id="lg-reg">–</span></div><div class="note">Visual stress estimate from recorded wall force, not FEA output</div>`; }
  else if (S.color === "strain") L.innerHTML = `<b>Plastic strain (normalised)</b><div class="ramp"></div><div class="ticks"><em>0</em><em>max</em></div>`;
  else if (S.color === "subsystem") L.innerHTML = `<b>Subsystem</b><br>` + SUBS.map((s, i) => `<span><i style="background:#${THREAD_COL[i].toString(16).padStart(6, "0")}"></i>${s}</span>`).join("");
  L.style.display = S.color === "paint" ? "none" : "block";
}

// ── Recorded-run mapping ──────────────────────────────
let rows = [], contactIdx = 0, i0 = 0, iMax = 1;
function rowAt(frac) { const f = frac * (rows.length - 1); const a = Math.floor(f), b = Math.min(rows.length - 1, a + 1), t = f - a; const A = rows[a], B = rows[b];
  const o = {}; for (const k in A) o[k] = typeof A[k] === "number" ? A[k] + (B[k] - A[k]) * t : A[k]; o.idx = a; return o; }

function update() {
  if (!rows.length || !parts.length) return;
  const r = rowAt(S.frac); const cf = contactIdx / (rows.length - 1);
  const kScale = (S.mass / 1500) * (S.speed / 15.56);
  S.sigMax = (S.Fmax * kScale) / A_EFF / 1e6;
  const strains = parts.map(p => new Float32Array(p.geo.attributes.position.count));
  const stress = parts.map(p => new Float32Array(p.geo.attributes.position.count));
  let crush = 0;
  if (S.frac <= cf) { car.position.x = CAR_START_X - (S.frac / Math.max(cf, 1e-6)) * (CAR_START_X - CONTACT_X); }
  else { crush = Math.min(1, Math.max(0, (r.max_intrusion_m - i0) / (iMax - i0))); car.position.x = CONTACT_X - crush * 0.05; }
  const cd = crush * VIS_CRUSH;
  const dtv = S.frac > cf ? (S.frac - cf) * 3.2 : -1;
  const fNow = Math.max(0, r.wall_force_N) / (S.Fmax || 1);   // normalised recorded wall force
  const reach = dtv > 0 ? Math.min(CAR_LEN + 0.5, dtv * 6) : 0; // stress wave front travelling back through the body
  let pk = 0, pkI = -1, pkP = null;
  parts.forEach((p, pi) => {
    const pos = p.geo.attributes.position;
    if (S.frac <= cf) p.plastic.fill(0);
    for (let i = 0; i < pos.count; i++) {
      const ox = p.orig.getX(i), oy = p.orig.getY(i), oz = p.orig.getZ(i); const lx = ox;
      const xf = Math.max(0, (CRUSH_END - lx) / (CRUSH_END - FRONT_X)); const xf2 = xf * xf;
      const dx = Math.min(cd * xf2 * 1.4, Math.max(0, lx - FRONT_X + cd));
      const buck = crush * xf2 * 0.07; const dy = Math.sin(lx * 22 + oz * 3) * buck, dz = Math.cos(lx * 17 + oy * 9) * buck * 1.2;
      pos.setXYZ(i, ox + dx, oy + dy, oz + dz);
      const st = Math.min(1, (Math.abs(dx) * 1.6 + Math.abs(dy) * 6 + Math.abs(dz) * 6) / Math.max(VIS_CRUSH, 0.01)); strains[pi][i] = st;
      // stress: recorded force × decay with distance from barrier × load-path weight, gated by the wave front
      const d = lx - FRONT_X; let sg = 0;
      if (d < reach) sg = fNow * Math.exp(-d / 1.5) * LOAD_W[p.region[i]] * (1 + 0.15 * Math.sin(lx * 9 + oy * 7));
      p.plastic[i] = Math.max(p.plastic[i], Math.min(1, st * 1.3));
      sg = Math.min(1, Math.max(sg, p.plastic[i] * 0.85));
      stress[pi][i] = sg; if (sg > pk) { pk = sg; pkI = i; pkP = p; }
    }
    pos.needsUpdate = true; p.geo.computeVertexNormals();
  });
  applyColors(strains, stress);
  if (pkP && S.color === "stress" && pk > 0.02) { const pp = pkP.geo.attributes.position; peakMark.position.set(pp.getX(pkI), pp.getY(pkI), pp.getZ(pkI)); peakMark.visible = true; } else peakMark.visible = false;
  if ($("lg-peak")) { document.querySelectorAll("#legend .ticks em").forEach((e, k) => e.textContent = (k / 4 * S.sigMax).toFixed(0)); $("lg-peak").textContent = (pk * S.sigMax).toFixed(0); $("lg-reg").textContent = pkP && pk > 0.02 ? SUBS[pkP.region[pkI]] : "–"; }
  crackMat.opacity = Math.min(0.9, crush * 1.6);
  wheels.forEach(w => { if (w.f) { w.g.position.x = w.x + cd * 0.45; w.g.rotation.y = cd * 0.5; } });
  const dt = dtv;
  debris.forEach(d => { d.m.visible = dt > 0 && dt < 3; if (!d.m.visible) return;
    const p = d.p0.clone().addScaledVector(d.v, dt); p.y += -4.9 * dt * dt; if (p.y < 0.03) p.y = 0.03; d.m.position.copy(p); d.m.rotation.set(d.w.x * dt, d.w.y * dt, d.w.z * dt); });
  if (dt > 0 && !S.flashed) { S.flashed = true; $("flash").style.opacity = 0.35; setTimeout(() => $("flash").style.opacity = 0, 90); log(`Contact with barrier at t = ${(r.time_s * 1000).toFixed(3)} ms`); }
  if (dt <= 0) S.flashed = false;
  // readouts
  const F = r.wall_force_N * kScale; S.peakF = 0; for (let i = 0; i <= r.idx; i++) S.peakF = Math.max(S.peakF, rows[i].wall_force_N * kScale);
  const v = Math.sqrt(Math.max(0, 2 * r.ke_J / 1500)) * (S.speed / 15.56);
  $("r-speed").textContent = `${v.toFixed(2)} m/s (${(v * 3.6).toFixed(1)} km/h)`;
  $("r-impact").textContent = `${S.speed.toFixed(2)} m/s (${(S.speed * 3.6).toFixed(1)} km/h)`;
  $("r-force").textContent = `${fmt(F / 1000, 1)} kN`; $("r-peak").textContent = `${fmt(S.peakF / 1000, 1)} kN`;
  $("r-g").textContent = `${(S.peakF / (S.mass * 9.80665)).toFixed(1)} g`;
  $("r-crush").textContent = `${(r.max_intrusion_m * 1000).toFixed(1)} mm`;
  $("r-res").textContent = `${r.residual_pct.toFixed(4)} %`;
  $("tl-clock").textContent = (r.time_s * 1000).toFixed(3); $("tl-step").textContent = Math.round(r.step).toLocaleString();
  $("tl-range").value = S.frac * 1000;
  $("status").textContent = S.running ? "Simulating crash (recorded replay)…" : S.frac >= 1 ? "Crash complete — vehicle at rest" : S.frac > 0 ? "Paused" : "Ready to crash";
  drawTele(); liveSim(r, kScale, v); drawPulse();
}

// ── UI wiring ─────────────────────────────────────────
function setRunning(on) { S.running = on; if (on && S.frac >= 1) S.frac = 0; const lab = on ? "Pause" : S.frac >= 1 ? "Restart Crash Test" : S.frac > 0 ? "Resume Crash Test" : "Start Crash Test";
  $("b-start").textContent = lab; $("tl-play").textContent = on ? "❚❚" : "▶"; document.querySelector('[data-tool=play]').textContent = on ? "❚❚" : "▶"; log(on ? "Solver replay started" : "Solver replay paused"); update(); }
function reset() { S.running = false; S.frac = 0; S.stage = 0; setRunning(false); log("Reset to t = 0"); }
function stepOnce() { S.running = false; S.stage = (S.stage + 1) % 5; if (S.stage === 0) S.frac = Math.min(1, S.frac + 1 / (rows.length - 1)); log(`Step-once → ${STAGES[S.stage]}`); drawDag(); drawLoad(); update(); }
function camView(v) { const P = { iso: [[6.4, 3.0, 6.6], [1.4, 0.6, 0]], side: [[1.4, 1.0, 6.2], [1.4, 0.7, 0]], front: [[-2.4, 1.3, 2.2], [0.3, 0.6, 0]], top: [[1.2, 6, 0.05], [1.2, 0, 0]] }[v];
  camera.position.set(...P[0]); controls.target.set(...P[1]); controls.update(); $("sel-camera").value = v; }
function setShading(v) { S.shading = v; $("sel-shading").value = v; parts.forEach(p => { p.mesh.material.wireframe = v === "wire"; p.mesh.material.visible = v !== "nodes"; p.points.visible = v === "nodes"; });
  document.querySelector('[data-tool=wire]').classList.toggle("active", v === "wire"); document.querySelector('[data-tool=nodes]').classList.toggle("active", v === "nodes"); }

$("b-start").onclick = () => setRunning(!S.running); $("tl-play").onclick = () => setRunning(!S.running);
$("b-reset").onclick = reset; $("b-step").onclick = stepOnce;
$("b-create").onclick = () => { reset(); camView("iso"); log(`Scene created · mesh ${S.mesh} · ${S.threads} threads · ${S.sched}`); };
$("b-csv").onclick = () => { const keys = Object.keys(rows[0]); const n = Math.max(1, rowAt(S.frac).idx + 1);
  const body = [keys.join(","), ...rows.slice(0, n).map(r => keys.map(k => r[k]).join(","))].join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([body], { type: "text/csv" })); a.download = "crash_telemetry_recorded.csv"; a.click(); log(`Exported ${n} rows to CSV`); };
$("tl-range").oninput = (e) => { S.running = false; S.frac = e.target.value / 1000; setRunning(false); };
$("in-speed").oninput = (e) => { S.speed = +e.target.value; $("v-speed").textContent = `${S.speed.toFixed(2)} m/s (${(S.speed * 3.6).toFixed(1)} km/h)`; update(); };
$("in-mass").oninput = (e) => { S.mass = +e.target.value; $("v-mass").textContent = `${S.mass} kg`; update(); };
$("in-mesh").onchange = (e) => { S.mesh = e.target.value; refreshHPC(); log(`Mesh → ${S.mesh}`); };
$("sel-camera").onchange = (e) => camView(e.target.value);
$("sel-shading").onchange = (e) => setShading(e.target.value);
$("sel-color").onchange = (e) => { S.color = e.target.value; updateLegend(); update(); };
function seg(id, fn) { $(id).querySelectorAll("button").forEach(b => b.onclick = (ev) => { ev.preventDefault(); $(id).querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); fn(b.textContent); }); }
seg("in-threads", (t) => { S.threads = +t; refreshHPC(); updateLegend(); update(); log(`OMP_NUM_THREADS=${t}`); });
seg("in-sched", (t) => { S.sched = t; drawSplit(); updateLegend(); update(); log(`schedule(${t})`); });
$("in-asm").onchange = (e) => { S.asm = e.target.value; drawAsm(); log(`Assembly → ${S.asm}`); };
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { document.querySelectorAll("#tabs button,.tab").forEach(x => x.classList.toggle("active", x.dataset.tab === b.dataset.tab)); requestAnimationFrame(drawAll); });
document.querySelectorAll("#dock-tabs button").forEach(b => b.onclick = () => { document.querySelectorAll("#dock-tabs button,.dock").forEach(x => x.classList.toggle("active", x.dataset.dock === b.dataset.dock)); requestAnimationFrame(drawAll); });
$("viva-toggle").onchange = (e) => document.body.classList.toggle("viva-on", e.target.checked);
document.querySelectorAll(".tool").forEach(b => b.onclick = () => { const t = b.dataset.tool;
  if (t === "orbit" || t === "pan") { document.querySelectorAll('[data-tool=orbit],[data-tool=pan]').forEach(x => x.classList.remove("active")); b.classList.add("active");
    controls.mouseButtons.LEFT = t === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE; }
  if (t === "fit") camView("iso"); if (t === "wire") setShading(S.shading === "wire" ? "lit" : "wire"); if (t === "nodes") setShading(S.shading === "nodes" ? "lit" : "nodes");
  if (t === "play") setRunning(!S.running); if (t === "step") stepOnce(); if (t === "reset") reset(); });
document.querySelectorAll("#menubar [data-menu]").forEach(b => b.onclick = () => { const m = b.dataset.menu;
  if (m === "file") $("b-csv").click(); if (m === "view") camView(["iso", "side", "front", "top"][(["iso", "side", "front", "top"].indexOf($("sel-camera").value) + 1) % 4]);
  if (m === "sim") setRunning(!S.running); if (m === "help") log("Keys: Space play · S step · R reset · W wireframe · N nodes · F frame · 1–4 cameras"); });
addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return; const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); setRunning(!S.running); } if (k === "s") stepOnce(); if (k === "r") reset(); if (k === "w") setShading(S.shading === "wire" ? "lit" : "wire");
  if (k === "n") setShading(S.shading === "nodes" ? "lit" : "nodes"); if (k === "f") camView("iso"); if ("1234".includes(k) && k) camView(["iso", "side", "front", "top"][+k - 1]); });

// ── Canvas helpers ────────────────────────────────────
function ctx2d(c) { const r = c.getBoundingClientRect(); if (!r.width) return null; const d = devicePixelRatio || 1; const h = r.height || +c.getAttribute("height");
  c.width = r.width * d; c.height = h * d; const g = c.getContext("2d"); g.scale(d, d); g.font = "10px JetBrains Mono, monospace"; return [g, r.width, h]; }
function axes(g, w, h, pad, xl, yl) { g.strokeStyle = "#1f2a36"; g.fillStyle = "#7d8a99"; g.beginPath(); g.moveTo(pad.l, pad.t); g.lineTo(pad.l, h - pad.b); g.lineTo(w - pad.r, h - pad.b); g.stroke(); g.fillText(yl, pad.l + 4, pad.t + 8); g.fillText(xl, w - pad.r - g.measureText(xl).width, h - pad.b - 4); }

function drawDag() {
  const svg = $("dag"); const xs = [90, 290, 490, 690, 890];
  let h = `<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#7d8a99"/></marker></defs>`;
  xs.forEach((x, i) => { const on = i === S.stage; const kind = i === 4 ? "reduction(min)" : i === 2 ? "parallel for + spatial hash" : "parallel for (owner slabs)";
    h += `<rect x="${x - 80}" y="20" width="160" height="52" rx="5" fill="${on ? STAGE_COL[i] + "33" : "#141a22"}" stroke="${on ? STAGE_COL[i] : "#1f2a36"}" stroke-width="${on ? 2 : 1}"/>`;
    h += `<text x="${x}" y="42" fill="#d6e2ee" font-size="13" text-anchor="middle" font-family="IBM Plex Sans">${STAGES[i]}</text><text x="${x}" y="60" fill="#7d8a99" font-size="10" text-anchor="middle" font-family="JetBrains Mono">${kind}</text>`;
    if (i < 4) h += `<line x1="${x + 82}" y1="46" x2="${xs[i + 1] - 84}" y2="46" stroke="#7d8a99" marker-end="url(#ar)"/><line x1="${x + 100}" y1="14" x2="${x + 100}" y2="80" stroke="#ff3b3b" stroke-dasharray="3 3"/><text x="${x + 100}" y="94" fill="#ff3b3b" font-size="9" text-anchor="middle" font-family="JetBrains Mono">barrier</text>`; });
  h += `<path d="M890 72 Q890 112 490 112 Q90 112 90 76" fill="none" stroke="#00c2ff" stroke-dasharray="4 4" marker-end="url(#ar)"/><text x="490" y="108" fill="#00c2ff" font-size="10" text-anchor="middle" font-family="JetBrains Mono">next step n+1 uses dt from S5</text>`;
  svg.innerHTML = h; $("tl-stage").textContent = STAGES[S.stage].split(" ")[0]; $("tl-stage").style.background = STAGE_COL[S.stage];
}
function drawLoad() {
  const c = $("c-load"), x = ctx2d(c); if (!x || !D.threads) return; const [g, w, h] = x; const T = D.threads; const keys = ["s1_spring_s", "s2_assembly_s", "s3_contact_s", "s4_integ_s", "s5_cfl_s"];
  const mx = Math.max(...T.map(t => t.total_active_s)); const pad = 70, bh = Math.min(22, (h - 40) / T.length - 6);
  g.fillStyle = "#7d8a99"; g.fillText(`Per-thread, per-stage time · recorded run (${T.length} threads) · grey = barrier wait (idle)`, pad, 12);
  T.forEach((t, i) => { const y = 22 + i * (bh + 6); let xx = pad; g.fillStyle = "#d6e2ee"; g.fillText(`T${t.thread_id} (cpu ${t.cpu_id})`, 4, y + bh * .7);
    keys.forEach((k, s) => { const ww = (t[k] / mx) * (w - pad - 90); g.fillStyle = STAGE_COL[s]; g.globalAlpha = s === S.stage ? 1 : 0.55; g.fillRect(xx, y, ww, bh); xx += ww; });
    g.globalAlpha = 1; const idle = ((mx - t.total_active_s) / mx) * (w - pad - 90); g.fillStyle = "#263240"; g.fillRect(xx, y, idle, bh);
    g.fillStyle = "#7d8a99"; g.fillText(`${(t.total_active_s * 1000).toFixed(2)} ms`, w - 82, y + bh * .7); });
  const imb = mx / (T.reduce((a, t) => a + t.total_active_s, 0) / T.length); g.fillStyle = "#f0a020"; g.fillText(`load imbalance max/mean = ${imb.toFixed(2)}`, pad, h - 4);
  STAGES.forEach((s, i) => { g.fillStyle = STAGE_COL[i]; g.fillRect(pad + 230 + i * 110, h - 12, 8, 8); g.fillStyle = "#7d8a99"; g.fillText(s.split(" ")[0], pad + 242 + i * 110, h - 4); });
}
function drawTele() {
  const c = $("c-tele"); if (!c.offsetParent) return; const x = ctx2d(c); if (!x || !rows.length) return; const [g, w, h] = x; const pad = { l: 40, r: 10, t: 14, b: 16 };
  axes(g, w, h, pad, "time (ms)", ""); const t1 = rows[rows.length - 1].time_s; const px = t => pad.l + (t / t1) * (w - pad.l - pad.r);
  const series = [["wall_force_N", "#ff3b3b", "wall force"], ["ke_J", "#00c2ff", "kinetic E"], ["plastic_e_J", "#f0a020", "plastic E"], ["damp_e_J", "#c678dd", "damping E"], ["total_e_J", "#3ddc97", "total E"]];
  series.forEach(([k, col, lab], si) => { const mx = Math.max(...rows.map(r => Math.abs(r[k]))) || 1; g.strokeStyle = col; g.beginPath();
    rows.forEach((r, i) => { const y = h - pad.b - (r[k] / mx) * (h - pad.t - pad.b); i ? g.lineTo(px(r.time_s), y) : g.moveTo(px(r.time_s), y); }); g.stroke();
    g.fillStyle = col; g.fillText(lab, pad.l + 8 + si * 90, pad.t - 3); });
  const cur = rowAt(S.frac); g.strokeStyle = "#fff"; g.beginPath(); g.moveTo(px(cur.time_s), pad.t); g.lineTo(px(cur.time_s), h - pad.b); g.stroke();
  g.fillStyle = "#7d8a99"; g.fillText(`${(t1 * 1000).toFixed(1)} ms`, w - 50, h - 3); g.fillText("normalised", 2, pad.t + 10);
}
function scaleRows(m) { return (D.scaling || []).filter(r => r.scale === m); }
function refreshHPC() {
  const R = scaleRows(S.mesh); if (!R.length) return; requestAnimationFrame(() => { drawHpcSp(); drawThrStage(); drawSplit(); }); const r = R.find(x => x.threads === S.threads); const kf = R.filter(x => x.threads > 1).map(x => x.karp_flatt).sort((a, b) => a - b);
  const f = kf[Math.floor(kf.length / 2)]; const am = 1 / (f + (1 - f) / S.threads);
  $("m-time").textContent = `${r.time_median_s.toFixed(3)} s`; $("m-sp").textContent = `${r.speedup.toFixed(2)}×`; $("m-eff").textContent = `${(r.efficiency * 100).toFixed(1)} %`;
  $("m-kf").textContent = S.threads === 1 ? "–" : r.karp_flatt.toFixed(3); $("m-am").textContent = `${am.toFixed(2)}×`; $("m-f").textContent = f.toFixed(3);
  $("m-bw").textContent = `${r.bandwidth_gb_s.toFixed(1)} GB/s (${r.stream_pct.toFixed(0)}% STREAM)`;
  $("p-mesh").textContent = S.mesh; $("p-nodes").textContent = r.num_nodes.toLocaleString(); $("p-thr").textContent = S.threads; $("p-msstep").textContent = (r.time_median_s / r.steps * 1000).toFixed(2);
  drawAsm();
}
function drawAsm() {
  const c = $("c-asm"); if (!c.offsetParent || !D.sweep) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x;
  const modes = ["Node Gather CSR", "Private Slabs", "Atomic Updates", "Node Gather CSR (FP32)"];
  const vals = modes.map(m => D.sweep.find(r => r.mesh === S.mesh && r.mode_name === m && r.threads === S.threads));
  const mx = Math.max(...vals.map(v => v ? v.time_median_s : 0)) || 1; g.fillStyle = "#7d8a99"; g.fillText(`median time, ${S.threads} threads, ${S.mesh} (lower is better)`, 0, 10);
  vals.forEach((v, i) => { const y = 20 + i * 30; g.fillStyle = "#d6e2ee"; g.fillText(modes[i], 0, y + 13);
    if (!v) { g.fillStyle = "#7d8a99"; g.fillText("not measured", 150, y + 13); return; }
    const ww = (v.time_median_s / mx) * (w - 230); g.fillStyle = modes[i] === S.asm ? "#3ddc97" : "#263240"; g.fillRect(150, y + 3, ww, 14);
    g.fillStyle = "#d6e2ee"; g.fillText(`${v.time_median_s.toFixed(3)} s`, 155 + ww, y + 14); });
}
function drawSpeed() {
  const c = $("c-speed"); if (!c.offsetParent || !D.scaling) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const pad = { l: 30, r: 10, t: 16, b: 18 };
  axes(g, w, h, pad, "threads p", "speedup S(p)"); const P = [1, 2, 4, 8, 12], mxS = 6; const px = p => pad.l + ((p - 1) / 11) * (w - pad.l - pad.r), py = s => h - pad.b - (s / mxS) * (h - pad.t - pad.b);
  g.setLineDash([4, 4]); g.strokeStyle = "#7d8a99"; g.beginPath(); g.moveTo(px(1), py(1)); g.lineTo(px(6), py(6)); g.stroke(); g.setLineDash([]);
  [["100k", "#00c2ff"], ["500k", "#3ddc97"]].forEach(([m, col], k) => { const R = scaleRows(m); g.strokeStyle = col; g.fillStyle = col; g.beginPath();
    R.forEach((r, i) => { i ? g.lineTo(px(r.threads), py(r.speedup)) : g.moveTo(px(r.threads), py(r.speedup)); }); g.stroke();
    R.forEach(r => { g.beginPath(); g.arc(px(r.threads), py(r.speedup), 3, 0, 7); g.fill(); }); g.fillText(m, w - 90 + k * 40, pad.t + 22);
    const kf = R.filter(x => x.threads > 1).map(x => x.karp_flatt).sort((a, b) => a - b); const f = kf[Math.floor(kf.length / 2)];
    g.globalAlpha = .45; g.setLineDash([2, 3]); g.beginPath(); P.forEach((p, i) => { const s = 1 / (f + (1 - f) / p); i ? g.lineTo(px(p), py(s)) : g.moveTo(px(p), py(s)); }); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; });
  g.fillStyle = "#7d8a99"; g.fillText("dashed = ideal · dotted = Amdahl fit", pad.l + 4, pad.t + 22); P.forEach(p => g.fillText(p, px(p) - 3, h - 5));
}
function drawEff() {
  const c = $("c-eff"); if (!c.offsetParent || !D.scaling) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const pad = { l: 30, r: 10, t: 16, b: 18 };
  axes(g, w, h, pad, "", "efficiency E(p)"); const P = [1, 2, 4, 8, 12]; const bw = (w - pad.l - pad.r) / P.length;
  P.forEach((p, i) => { [["100k", "#00c2ff"], ["500k", "#3ddc97"]].forEach(([m, col], k) => { const r = scaleRows(m).find(x => x.threads === p); if (!r) return;
    const bh = r.efficiency * (h - pad.t - pad.b); g.fillStyle = col; g.fillRect(pad.l + i * bw + 8 + k * (bw / 2 - 6), h - pad.b - bh, bw / 2 - 10, bh); });
    g.fillStyle = "#7d8a99"; g.fillText(`p=${p}`, pad.l + i * bw + bw / 2 - 12, h - 5); });
}
function liveSim(r, k, v) {
  const E = [["Kinetic", r.ke_J, "#9aa5b1"], ["Plastic work", r.plastic_e_J, "#f0a020"], ["Elastic strain", r.strain_e_J, "#ffd23f"], ["Damping", r.damp_e_J, "#c678dd"], ["Wall", r.wall_e_J, "#ff3b3b"]];
  const tot = E.reduce((a, e) => a + Math.abs(e[1]), 0) || 1;
  $("sim-live").innerHTML = `<dt>Time</dt><dd>${(r.time_s * 1000).toFixed(3)} ms</dd><dt>Step</dt><dd>${Math.round(r.step).toLocaleString()}</dd><dt>Phase</dt><dd>${r.wall_force_N > 0 ? (S.frac >= 1 ? "rebound / rest" : "crushing") : "free flight"}</dd><dt>Decel now</dt><dd>${(r.wall_force_N * k / (S.mass * 9.80665)).toFixed(1)} g</dd><dt>Momentum</dt><dd>${fmt(S.mass * v, 0)} kg·m/s</dd><dt>Energy absorbed</dt><dd>${(100 * r.plastic_e_J / (rows[0].total_e_J || 1)).toFixed(1)} %</dd><dt>Total energy</dt><dd>${fmt(r.total_e_J / 1000, 2)} kJ</dd>`;
  $("sim-energy").innerHTML = E.map(([l, val, c]) => `<div class="bar"><span>${l}</span><div style="width:${Math.abs(val) / tot * 170}px;background:${c}"></div><em>${fmt(val / 1000, 2)} kJ</em></div>`).join("");
}
function drawPulse() {
  const c = $("c-pulse"); if (!c || !c.offsetParent || !rows.length) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const pad = { l: 34, r: 8, t: 14, b: 16 };
  axes(g, w, h, pad, "t (ms)", "wall force kN"); const t1 = rows[rows.length - 1].time_s, mx = S.Fmax || 1; const px = t => pad.l + t / t1 * (w - pad.l - pad.r), py = f => h - pad.b - f / mx * (h - pad.t - pad.b);
  const cur = rowAt(S.frac); g.fillStyle = "rgba(255,59,59,.18)"; g.beginPath(); g.moveTo(px(0), py(0)); rows.forEach(r => { if (r.time_s <= cur.time_s) g.lineTo(px(r.time_s), py(r.wall_force_N)); }); g.lineTo(px(cur.time_s), py(0)); g.fill();
  g.strokeStyle = "#ff3b3b"; g.beginPath(); rows.forEach((r, i) => i ? g.lineTo(px(r.time_s), py(r.wall_force_N)) : g.moveTo(px(r.time_s), py(r.wall_force_N))); g.stroke();
  g.strokeStyle = "#f0a020"; g.setLineDash([3, 3]); const im = iMax || 1; g.beginPath(); rows.forEach((r, i) => { const y = h - pad.b - r.max_intrusion_m / im * (h - pad.t - pad.b); i ? g.lineTo(px(r.time_s), y) : g.moveTo(px(r.time_s), y); }); g.stroke(); g.setLineDash([]);
  g.strokeStyle = "#fff"; g.beginPath(); g.moveTo(px(cur.time_s), pad.t); g.lineTo(px(cur.time_s), h - pad.b); g.stroke();
  g.fillStyle = "#7d8a99"; g.fillText(`${(mx / 1000).toFixed(0)}`, 4, pad.t + 4); g.fillText(`${(t1 * 1000).toFixed(1)}`, w - 30, h - 3); g.fillStyle = "#f0a020"; g.fillText("intrusion (dashed)", w - 120, pad.t - 2);
}
function drawRes() {
  const c = $("c-res"); if (!c || !c.offsetParent || !D.crash) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const pad = { l: 40, r: 8, t: 14, b: 16 };
  axes(g, w, h, pad, "t", "|residual| %"); const lim = 0.1; const all = [...D.crash, ...D.energy]; const t1 = Math.max(...all.map(r => r.time_s));
  const py = v => h - pad.b - Math.min(1, Math.abs(v) / (lim * 1.2)) * (h - pad.t - pad.b), px = t => pad.l + t / t1 * (w - pad.l - pad.r);
  g.strokeStyle = "#ff3b3b"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(pad.l, py(lim)); g.lineTo(w - pad.r, py(lim)); g.stroke(); g.setLineDash([]); g.fillStyle = "#ff3b3b"; g.fillText("gate 0.1 %", w - 70, py(lim) - 3);
  [[D.crash, "#3ddc97", "vehicle crash"], [D.energy, "#ffd23f", "balance audit"]].forEach(([R, col, l], k) => { g.strokeStyle = col; g.beginPath(); R.forEach((r, i) => i ? g.lineTo(px(r.time_s), py(r.residual_pct)) : g.moveTo(px(r.time_s), py(r.residual_pct))); g.stroke(); g.fillStyle = col; g.fillText(l, pad.l + 90 + k * 100, pad.t - 3); });
}
function drawHpcSp() {
  const c = $("c-hpc-sp"); if (!c || !c.offsetParent || !D.scaling) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const R = scaleRows(S.mesh); const bw = (w - 20) / R.length; const mx = Math.max(...R.map(r => r.speedup)) * 1.2;
  R.forEach((r, i) => { const bh = r.speedup / mx * (h - 34); g.fillStyle = r.threads === S.threads ? "#3ddc97" : "#263240"; g.fillRect(10 + i * bw + 6, h - 18 - bh, bw - 12, bh);
    g.fillStyle = "#d6e2ee"; g.fillText(`${r.speedup.toFixed(2)}×`, 10 + i * bw + 8, h - 22 - bh); g.fillStyle = "#7d8a99"; g.fillText(`p=${r.threads}`, 10 + i * bw + 10, h - 5); });
  g.fillStyle = "#7d8a99"; g.fillText(`${S.mesh} · measured speedup, highlighted = selected p`, 10, 10);
}
const STK = ["s1_spring_s", "s2_assembly_s", "s3_contact_s", "s4_integ_s", "s5_cfl_s"];
function drawThrShare() {
  const c = $("c-thr-share"); if (!c || !c.offsetParent || !D.threads) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x; const T = D.threads; const bh = (h - 30) / T.length - 6;
  T.forEach((t, i) => { const y = 6 + i * (bh + 6); let xx = 40; g.fillStyle = "#d6e2ee"; g.fillText(`T${t.thread_id}`, 4, y + bh * .7); const tot = STK.reduce((a, k) => a + t[k], 0) || 1;
    STK.forEach((k, s) => { const ww = t[k] / tot * (w - 50); g.fillStyle = STAGE_COL[s]; g.fillRect(xx, y, ww, bh); if (ww > 34) { g.fillStyle = "#06080b"; g.fillText(`${(t[k] / tot * 100).toFixed(0)}%`, xx + 3, y + bh * .7); } xx += ww; }); });
  STAGES.forEach((s, i) => { g.fillStyle = STAGE_COL[i]; g.fillRect(40 + i * 62, h - 12, 8, 8); g.fillStyle = "#7d8a99"; g.fillText(s.split(" ")[0], 52 + i * 62, h - 4); });
}
function drawThrStage() {
  const c = $("c-thr-stage"); if (!c || !c.offsetParent || !D.sweep) return; const x = ctx2d(c); if (!x) return; const [g, w, h] = x;
  const R = D.sweep.filter(r => r.mesh === S.mesh && r.mode_name === "Node Gather CSR").sort((a, b) => a.threads - b.threads); if (!R.length) { g.fillStyle = "#7d8a99"; g.fillText(`no sweep rows for ${S.mesh}`, 4, 14); return; }
  const ks = ["s1_spring_ms", "s2_assembly_ms", "s3_contact_ms", "s4_integrate_ms", "s5_cfl_ms"]; const mx = Math.max(...R.map(r => ks.reduce((a, k) => a + r[k], 0))); const bw = (w - 40) / R.length;
  R.forEach((r, i) => { let y = h - 18; ks.forEach((k, s) => { const bh = r[k] / mx * (h - 40); g.fillStyle = STAGE_COL[s]; g.fillRect(36 + i * bw + 6, y - bh, bw - 12, bh); y -= bh; });
    g.fillStyle = "#d6e2ee"; g.fillText(`${(ks.reduce((a, k) => a + r[k], 0) / 1000).toFixed(2)}s`, 36 + i * bw + 6, y - 3); g.fillStyle = "#7d8a99"; g.fillText(`p=${r.threads}`, 36 + i * bw + 10, h - 5); });
  g.fillStyle = "#7d8a99"; g.fillText(`${S.mesh} · summed stage time per run`, 36, 10);
}
function drawSplit() {
  const el = $("thr-split"); if (!el) return; const N = 120; const cnt = new Array(S.threads).fill(0); let html = "", prev = -1, run = 0;
  const flush = () => { if (prev >= 0) html += `<div style="flex:${run};background:#${THREAD_COL[prev].toString(16).padStart(6, "0")}">${run > 8 ? "T" + prev : ""}</div>`; };
  for (let i = 0; i < N; i++) { const o = ownerOf((i + .5) / N, S.threads, S.sched); cnt[o]++; if (o !== prev) { flush(); prev = o; run = 0; } run++; } flush(); el.innerHTML = html;
  const r = scaleRows(S.mesh).find(x => x.threads === S.threads); const nodes = r ? r.num_nodes : 0;
  $("thr-split-note").textContent = `Vehicle length front → rear, ${S.threads} threads, schedule(${S.sched}). About ${Math.round(nodes / S.threads).toLocaleString()} nodes and ${r ? Math.round(r.num_springs / S.threads).toLocaleString() : "–"} springs per thread.`;
}
function drawAll() { drawLoad(); drawTele(); drawAsm(); drawSpeed(); drawEff(); drawPulse(); drawRes(); drawHpcSp(); drawThrShare(); drawThrStage(); drawSplit(); }
addEventListener("resize", () => requestAnimationFrame(drawAll));

function fillStatic() {
  // Scaling table
  const t = $("t-scale"); let h = "<tr><th>Mesh</th><th>p</th><th>T median (s)</th><th>min–max</th><th>S(p)</th><th>E(p)</th><th>e (K-F)</th><th>GB/s</th></tr>";
  D.scaling.forEach(r => h += `<tr><td>${r.scale}</td><td>${r.threads}</td><td>${r.time_median_s.toFixed(3)}</td><td>${r.time_min_s.toFixed(3)}–${r.time_max_s.toFixed(3)}</td><td>${r.speedup.toFixed(2)}</td><td>${(r.efficiency * 100).toFixed(0)}%</td><td>${r.threads > 1 ? r.karp_flatt.toFixed(3) : "–"}</td><td>${r.bandwidth_gb_s.toFixed(1)}</td></tr>`);
  t.innerHTML = h;
  const maxRes = (a) => Math.max(...a.map(r => Math.abs(r.residual_pct)));
  $("gates").innerHTML = `<dt>Crash energy residual (max)</dt><dd>${maxRes(D.crash).toFixed(4)} %</dd><dt>Balance audit residual (max)</dt><dd>${maxRes(D.energy).toFixed(4)} %</dd><dt>Gate: closure &lt; 0.1 %</dt><dd style="color:#3ddc97">${Math.max(maxRes(D.crash), maxRes(D.energy)) < 0.1 ? "PASS" : "FAIL"}</dd><dt>Python ↔ C++ agreement</dt><dd>8.16e-9 rel.</dd><dt>Elastic drift, 1,000 steps</dt><dd>≤ 0.00092 %</dd>`;
  const b = (m, p) => scaleRows(m).find(r => r.threads === p);
  const best = (m) => scaleRows(m).reduce((a, r) => r.speedup > a.speedup ? r : a);
  $("interp").innerHTML = [
    `Best speedup is ${best("100k").speedup.toFixed(2)}× at ${best("100k").threads} threads (100k) and ${best("500k").speedup.toFixed(2)}× at ${best("500k").threads} threads (500k). It is far below the ideal linear speedup.`,
    `Efficiency falls from ${(b("100k", 2).efficiency * 100).toFixed(0)}% at p=2 to ${(b("100k", 12).efficiency * 100).toFixed(0)}% at p=12. The Karp–Flatt serial fraction stays around 0.3–0.4, so the limit comes from memory bandwidth and hybrid P/E cores, not from serial code.`,
    `Peak bandwidth is ${Math.max(...D.scaling.map(r => r.bandwidth_gb_s)).toFixed(1)} GB/s, about ${Math.max(...D.scaling.map(r => r.stream_pct)).toFixed(0)}% of STREAM Triad (${D.hw.stream_triad_peak_gb_s} GB/s). With an arithmetic intensity of ${D.hw.arithmetic_intensity_flop_per_byte} FLOP/byte, the kernel is memory-bound.`,
    `The 500k mesh drops at p=12 versus p=8: the extra hyper-threads compete for the same L3 cache (${D.hw.l3_cache}) and memory channels.`,
  ].map(s => `<li>${s}</li>`).join("");
  $("hw").innerHTML = ["cpu", "os", "compiler", "l3_cache", "ram"].map(k => `<dt>${k.replace("_", " ")}</dt><dd style="text-align:left">${D.hw[k]}</dd>`).join("") + `<dt>STREAM Triad</dt><dd style="text-align:left">${D.hw.stream_triad_peak_gb_s} GB/s</dd>`;
  $("ai").textContent = D.hw.arithmetic_intensity_flop_per_byte;
  { const big = scaleRows("500k").find(r => r.threads === 1), sm = scaleRows("100k").find(r => r.threads === 1); const last = rows[rows.length - 1];
    const msStep = big.time_median_s / big.steps * 1000; const fullH = msStep * last.step / 3.6e6;
    $("stage-cards").innerHTML = [["Nodes (100k / 500k)", `${sm.num_nodes.toLocaleString()} / ${big.num_nodes.toLocaleString()}`, "mass points"], ["Springs (500k)", big.num_springs.toLocaleString(), "elements"], ["Steps per crash", last.step.toLocaleString(), `${(last.time_s * 1000).toFixed(1)} ms simulated`], ["Serial cost (500k)", `${msStep.toFixed(1)} ms/step`, "1 thread, measured"], ["One crash, 1 core", `${fullH.toFixed(2)} h`, "steps × ms/step"], ["Memory bound", `${D.hw.arithmetic_intensity_flop_per_byte} FLOP/B`, `STREAM ${D.hw.stream_triad_peak_gb_s} GB/s`]].map(([a, b, c]) => `<div class="card"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join("");
    $("stage-why").textContent = `At ${msStep.toFixed(1)} ms per step on one core, one 500k-node crash of ${last.step.toLocaleString()} steps takes about ${fullH.toFixed(2)} h. A design study of 50 variants at production mesh sizes (10–100× more elements) would take weeks, so the step loop has to run in parallel.`; }
  { const t0 = D.crash[0]; $("sim-setup").innerHTML = `<dt>Integrator</dt><dd>explicit central difference</dd><dt>Time step dt</dt><dd>${((D.crash[1].time_s - t0.time_s) / (D.crash[1].step - t0.step) * 1e6).toFixed(2)} µs</dd><dt>Total steps</dt><dd>${rows[rows.length - 1].step.toLocaleString()}</dd><dt>Recorded frames</dt><dd>${rows.length}</dd><dt>Initial KE</dt><dd>${fmt(t0.ke_J / 1000, 1)} kJ</dd><dt>Contact</dt><dd>penalty wall + spatial hash</dd><dt>Material</dt><dd>elastoplastic springs</dd><dt>Precision</dt><dd>FP64</dd>`; }
  { let sh = "<tr><th>Mesh</th><th>Mode</th><th>p</th><th>T (s)</th><th>±σ</th><th>S</th><th>E %</th><th>e</th></tr>";
    D.sweep.forEach(r => sh += `<tr><td>${r.mesh}</td><td>${r.mode_name.replace("Node Gather ", "").replace(" Updates", "")}</td><td>${r.threads}</td><td>${r.time_median_s.toFixed(3)}</td><td>${r.time_std_s.toFixed(3)}</td><td>${r.speedup_vs_gather1.toFixed(2)}</td><td>${r.efficiency_pct.toFixed(0)}</td><td>${r.threads > 1 ? r.karp_flatt_e.toFixed(2) : "–"}</td></tr>`); $("t-sweep").innerHTML = sh; }
  { const T = D.threads, ms = v => (v * 1000).toFixed(3); const mx = Math.max(...T.map(t => t.total_active_s)), mean = T.reduce((a, t) => a + t.total_active_s, 0) / T.length, mn = Math.min(...T.map(t => t.total_active_s));
    const slow = T.find(t => t.total_active_s === mx), fast = T.find(t => t.total_active_s === mn);
    $("thr-cards").innerHTML = [["Threads", T.length, "OpenMP team"], ["Slowest", `T${slow.thread_id} · ${ms(mx)} ms`, `cpu ${slow.cpu_id}`], ["Fastest", `T${fast.thread_id} · ${ms(mn)} ms`, `cpu ${fast.cpu_id}`], ["Imbalance max/mean", (mx / mean).toFixed(3), mx / mean < 1.1 ? "good (< 1.10)" : "needs balancing"], ["Barrier idle (total)", `${ms(T.reduce((a, t) => a + mx - t.total_active_s, 0))} ms`, "time spent waiting"], ["Parallel efficiency", `${(mean / mx * 100).toFixed(1)} %`, "mean / max active"]].map(([a, b, c]) => `<div class="card"><span>${a}</span><b>${b}</b><small>${c}</small></div>`).join("");
    let th = "<tr><th>T</th><th>cpu</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th><th>S5</th><th>active</th><th>idle</th></tr>";
    T.forEach(t => th += `<tr><td>T${t.thread_id}</td><td>${t.cpu_id}</td>${STK.map(k => `<td>${ms(t[k])}</td>`).join("")}<td>${ms(t.total_active_s)}</td><td class="${mx - t.total_active_s > 0.1 * mx ? "wn" : ""}">${((mx - t.total_active_s) / mx * 100).toFixed(1)}%</td></tr>`);
    $("t-thr").innerHTML = th; $("t-load").innerHTML = th;
    $("thr-pin").innerHTML = Array.from({ length: 12 }, (_, c) => { const t = T.find(x => x.cpu_id === c); return `<span class="pill" style="${t ? `color:#06080b;background:#${THREAD_COL[t.thread_id].toString(16).padStart(6, "0")}` : ""}">cpu${c}${t ? " · T" + t.thread_id : ""}</span>`; }).join("");
    const s1 = T.reduce((a, t) => a + t.s1_spring_s, 0) / T.reduce((a, t) => a + t.total_active_s, 0);
    $("thr-diag").innerHTML = [`S1 spring forces take ${(s1 * 100).toFixed(1)}% of each thread's active time, so it is the stage worth optimising.`,
      `T${slow.thread_id} (cpu ${slow.cpu_id}) is ${((mx / mn - 1) * 100).toFixed(1)}% slower than T${fast.thread_id} (cpu ${fast.cpu_id}). The likely cause is a slower E-core or a heavier slab near the crush zone.`,
      `S2 assembly shows ${T.every(t => t.s2_assembly_s === 0) ? "0 ms here because in this run the CSR gather is fused into S1" : "non-zero time"}. No thread writes to another thread's nodes.`,
      `S5 (CFL reduction) is microseconds per thread, so the reduction itself is not a bottleneck. The barrier wait before it is what costs time.`].map(x => `<li>${x}</li>`).join(""); }
  const base = D.sweep.find(r => r.mesh === "100k" && r.mode_name === "Node Gather CSR" && r.threads === 1);
  if (base) { const ks = [["s1_spring_ms", "S1 Spring"], ["s2_assembly_ms", "S2 Assembly"], ["s3_contact_ms", "S3 Contact"], ["s4_integrate_ms", "S4 Integrate"], ["s5_cfl_ms", "S5 CFL"]];
    const tot = ks.reduce((a, [k]) => a + base[k], 0);
    $("bottleneck").innerHTML = ks.map(([k, l], i) => `<div class="bar"><span>${l}</span><div style="width:${(base[k] / tot) * 180}px;background:${STAGE_COL[i]}"></div><em>${(base[k] / tot * 100).toFixed(1)}%</em></div>`).join(""); }
}

// ── Main loop ─────────────────────────────────────────
let last = performance.now(), frames = 0, acc = 0, stageAcc = 0;
function animate(now) {
  requestAnimationFrame(animate); const dt = Math.min(0.05, (now - last) / 1000); last = now;
  frames++; acc += dt; if (acc > 0.5) { $("p-fps").textContent = (frames / acc).toFixed(1); $("p-ft").textContent = (1000 * acc / frames).toFixed(1) + " ms"; frames = 0; acc = 0; }
  if (S.running) { S.frac += dt / (S.frac > contactIdx / (rows.length - 1) ? 40 : 8); stageAcc += dt; if (stageAcc > 0.12) { stageAcc = 0; S.stage = (S.stage + 1) % 5; drawDag(); if ($("c-load").offsetParent) drawLoad(); }
    if (S.frac >= 1) { S.frac = 1; setRunning(false); log("Crash complete — vehicle at rest"); } update(); }
  controls.update(); renderer.render(scene, camera);
}

// ── Boot ──────────────────────────────────────────────
Promise.all(["vehicle_crash_metrics.csv", "energy_balance_term_by_term.csv", "vehicle_scaling_results.csv", "thread_timing_breakdown.csv", "assembly_sweep_raw.csv", "hw.json"]
  .map(f => fetch("data/" + f).then(r => { if (!r.ok) throw new Error(f); return f.endsWith(".json") ? r.json() : r.text().then(csv); })))
  .then(([crash, energy, scaling, threads, sweep, hw]) => {
    Object.assign(D, { crash, energy, scaling, threads, sweep, hw }); rows = crash;
    contactIdx = Math.max(1, rows.findIndex(r => r.wall_force_N > 0)); i0 = rows[contactIdx].max_intrusion_m; iMax = Math.max(...rows.map(r => r.max_intrusion_m));
    $("tl-range").max = 1000; $("tl-range").step = 1; document.querySelector("#timeline .mono").innerHTML = `t = <b id="tl-clock">0</b> / ${(rows[rows.length - 1].time_s * 1000).toFixed(2)} ms`;
    document.querySelectorAll("#timeline .mono")[1].innerHTML = `Step <b id="tl-step">0</b> / ${rows[rows.length - 1].step.toLocaleString()}`;
    S.Fmax = Math.max(...rows.map(r => r.wall_force_N)); fillStatic(); refreshHPC(); drawDag(); updateLegend(); drawAll();
    loadCar().then(() => { update(); setShading(S.shading); vp.dataset.sceneReady = "1"; log(`Vehicle model loaded: Kenney SUV (CC0), ${$("p-drawn").textContent} nodes after tessellation`); }).catch(e => log("Failed to load vehicle model: " + (e.message || e)));
    log(`Loaded recorded run: ${rows.length} frames, ${scaling.length} scaling configs, ${sweep.length} assembly runs`);
    log(`Host: ${hw.cpu}`);
  }).catch(err => log("Failed to load data: " + err.message));
camView("iso"); resize(); setShading("lit"); requestAnimationFrame(animate);
})();

(function(){
  const go=()=>{try{window.top.location.href="/analysis"}catch(e){window.location.href="/analysis"}};
  document.getElementById("open-analysis").onclick=go;document.getElementById("rail-analysis").onclick=go;
  const app=document.getElementById("app"),hd=document.getElementById("dock-resize");let drag=false;
  const saved=+localStorage.getItem("dockH");const set=h=>{h=Math.max(90,Math.min(window.innerHeight-160,h));app.style.gridTemplateRows=`30px 1fr ${h}px`;return h};
  if(saved)set(saved);
  hd.addEventListener("pointerdown",e=>{drag=true;hd.classList.add("on");hd.setPointerCapture(e.pointerId);e.preventDefault()});
  hd.addEventListener("pointermove",e=>{if(!drag)return;set(window.innerHeight-e.clientY);window.dispatchEvent(new Event("resize"));});
  hd.addEventListener("pointerup",e=>{drag=false;hd.classList.remove("on");localStorage.setItem("dockH",set(window.innerHeight-e.clientY));window.dispatchEvent(new Event("resize"));if(typeof drawAll==="function")drawAll()});
})();
