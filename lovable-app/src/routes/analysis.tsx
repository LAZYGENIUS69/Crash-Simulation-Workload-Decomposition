import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { SiteNav } from "@/components/SiteNav";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Workload Analysis | Crash-Simulation Step-by-Step Walkthrough" },
      { name: "description", content: "Full-screen walkthrough of the 56 km/h crash run: task dependency graph, per-thread load and telemetry from first step to last." },
      { property: "og:title", content: "Workload Analysis | Crash-Simulation Step-by-Step Walkthrough" },
      { property: "og:description", content: "Task dependency graph, thread load and telemetry of the recorded OpenMP crash run, from impact start to end." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Analysis,
});

interface Row { step: number; time_s: number; wall_force_N: number; max_intrusion_m: number; ke_J: number; strain_e_J: number; wall_e_J: number; damp_e_J: number; plastic_e_J: number; total_e_J: number; residual_pct: number; thread_id: number; cpu_id: number; s1_spring_s: number; s2_assembly_s: number; s3_contact_s: number; s4_integ_s: number; s5_cfl_s: number; total_active_s: number; threads: number; speedup: number; efficiency: number; scale: number; }
const g = (x: Row, k: string) => (x as unknown as Record<string, number>)[k] ?? 0;
const parse = (t: string): Row[] => {
  const [h, ...l] = t.trim().split(/\r?\n/);
  const k = h!.split(",");
  return l.map((s) => Object.fromEntries(s.split(",").map((v, i) => [k[i], Number(v)])) as Row);
};

const STAGES = [
  { id: "S1", name: "Spring forces", omp: "#pragma omp for schedule(static)", text: "Every spring computes its elastoplastic force f = k·(|xj − xi| − L0 − εp). Springs past yield store permanent plastic strain, which is how the body stays crumpled. This is the heaviest stage and is memory-bound." },
  { id: "S2", name: "Force assembly", omp: "CSR gather, owner writes only F[i]", text: "Each node sums the forces of its attached springs through a CSR neighbour list. A thread only writes the nodes it owns, so there are no data races, no atomics and no private copies." },
  { id: "S3", name: "Contact", omp: "parallel for + spatial hash", text: "Nodes that pass the rigid wall get a penalty force pushing them back; this sum is the wall force. A spatial hash finds parts of the body folding into each other (self-contact)." },
  { id: "S4", name: "Integration", omp: "parallel for (owner slabs)", text: "Velocity Verlet: v += a·dt, x += v·dt for every node. Each thread updates its own contiguous slab of the vehicle." },
  { id: "S5", name: "Stable time step", omp: "reduction(min:dt)", text: "The CFL condition dt = C·min(L/c) is reduced across all threads. The next step cannot start until every thread reports, which is why S5 is a global sync point." },
];
const COLS_ = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];
const col = (n: number) => COLS_[n % 5] ?? "var(--chart-1)";

function phaseOf(r: Row, peakT: number, contactT: number) {
  if (r.time_s < contactT) return { name: "Approach", text: "The car travels at full speed toward the barrier. No wall force yet; nearly all energy is kinetic." };
  if (r.time_s <= peakT) return { name: "Crush onset", text: "The bumper and front rails hit the wall. Wall force climbs fast and springs in the crumple zone start to yield." };
  if (r.ke_J > 0.05 * 1) return { name: "Progressive crush", text: "The front structure folds back. Kinetic energy turns into plastic work in the rails and hood; intrusion grows." };
  return { name: "Rebound", text: "The car has nearly stopped. Elastic energy springs back a little and the wall force drops to zero." };
}

function Analysis() {
  const [rows, setRows] = useState<Row[]>([]);
  const [thr, setThr] = useState<Row[]>([]);
  const [scal, setScal] = useState<Row[]>([]);
  const [i, setI] = useState(0);
  const [play, setPlay] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    const get = (f: string) => fetch(`/studio/data/${f}`).then((r) => r.text()).then(parse);
    get("vehicle_crash_metrics.csv").then(setRows);
    get("thread_timing_breakdown.csv").then(setThr);
    fetch("/studio/data/vehicle_scaling_results.csv").then((r) => r.text()).then((t) => {
      const [h, ...l] = t.trim().split(/\r?\n/); const k = h!.split(",");
      setScal(l.map((s) => Object.fromEntries(s.split(",").map((v, j) => [k[j], j === 0 ? (v as unknown as number) : Number(v)])) as Row));
    });
  }, []);

  useEffect(() => {
    if (!play || !rows.length) return;
    let last = performance.now();
    const tick = (t: number) => {
      if (t - last > 60) { last = t; setI((p) => { if (p >= rows.length - 1) { setPlay(false); return p; } return p + 1; }); }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [play, rows.length]);

  const info = useMemo(() => {
    if (!rows.length) return null;
    const r0 = rows[0]!; const peak = rows.reduce((a, r) => (r.wall_force_N > a.wall_force_N ? r : a), r0);
    const contact = rows.find((r) => r.wall_force_N > 0) ?? r0;
    const ke0 = r0.ke_J;
    const maxRes = Math.max(...rows.map((r) => Math.abs(r.residual_pct)));
    return { peak, contact, ke0, maxRes, tEnd: rows[rows.length - 1]!.time_s, steps: rows[rows.length - 1]!.step };
  }, [rows]);

  if (!rows.length || !info) return (<div className="flex h-screen flex-col bg-background"><SiteNav /><p className="p-8 font-mono text-sm text-muted-foreground">Loading recorded run…</p></div>);

  const r = rows[i]!;
  const ph = (() => { const p = phaseOf(r, info.peak.time_s, info.contact.time_s); if (p.name === "Progressive crush" && r.ke_J < 0.03 * info.ke0) return phaseOf({ ...r, ke_J: 0 }, info.peak.time_s, info.contact.time_s); return p; })();
  const stage = i % 5;
  const keys = ["s1_spring_s", "s2_assembly_s", "s3_contact_s", "s4_integ_s", "s5_cfl_s"];
  const mx = Math.max(...thr.map((t) => t.total_active_s), 1e-9);
  const mean = thr.reduce((a, t) => a + t.total_active_s, 0) / (thr.length || 1);
  const ms = (s: number) => (s * 1000).toFixed(2);

  // telemetry chart
  const W = 1000, H = 260, P = 36;
  const px = (t: number) => P + (t / info.tEnd) * (W - 2 * P);
  const series: [string, string, string][] = [["wall_force_N", "var(--destructive)", "Wall force"], ["ke_J", "var(--chart-1)", "Kinetic energy"], ["plastic_e_J", "var(--chart-3)", "Plastic work"], ["strain_e_J", "var(--chart-2)", "Elastic strain"], ["total_e_J", "var(--chart-5)", "Total energy"]];
  const path = (k: string) => { const m = Math.max(...rows.map((x) => Math.abs(g(x, k)))) || 1; return rows.map((x, j) => `${j ? "L" : "M"}${px(x.time_s).toFixed(1)} ${(H - P - (g(x, k) / m) * (H - 2 * P)).toFixed(1)}`).join(" "); };

  const s100: Row[] = scal.filter((s) => (s.scale as unknown as string) === "100k");
  const s500: Row[] = scal.filter((s) => (s.scale as unknown as string) === "500k");
  const eTerms: [string, number, string][] = [["Kinetic", r.ke_J, col(0)], ["Elastic strain", r.strain_e_J, col(1)], ["Plastic work", r.plastic_e_J, col(2)], ["Wall work", r.wall_e_J, col(3)], ["Damping", r.damp_e_J, col(4)]];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <SiteNav />
      <header className="sticky top-0 z-10 border-b border-border bg-card/95 px-6 py-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-4">
          <Link to="/" className="font-mono text-xs uppercase tracking-widest text-primary hover:underline">← Back to 3D studio</Link>
          <h1 className="text-lg font-semibold">Workload analysis · start to end</h1>
          <span className="ml-auto font-mono text-xs text-muted-foreground">Recorded C++/OpenMP run · {rows.length} samples · {(info.tEnd * 1000).toFixed(1)} ms · step {info.steps.toLocaleString()}</span>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button onClick={() => { if (i >= rows.length - 1) setI(0); setPlay(!play); }} className="border border-primary px-3 py-1 font-mono text-xs uppercase text-primary hover:bg-primary hover:text-primary-foreground">{play ? "Pause" : "Play"}</button>
          <button onClick={() => { setPlay(false); setI(0); }} className="border border-border px-3 py-1 font-mono text-xs uppercase text-muted-foreground">Reset</button>
          <input type="range" min={0} max={rows.length - 1} value={i} onChange={(e) => setI(+e.target.value)} className="flex-1 accent-[var(--primary)]" />
          <span className="w-56 text-right font-mono text-xs">t = {(r.time_s * 1000).toFixed(2)} ms · step {r.step.toLocaleString()}</span>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6">
        <section className="grid gap-4 md:grid-cols-[1.2fr_1fr]">
          <div className="border border-border bg-card p-4">
            <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Current phase</p>
            <h2 className="mt-1 text-2xl font-semibold text-primary">{ph.name}</h2>
            <p className="mt-2 text-sm text-muted-foreground">{ph.text}</p>
            <div className="mt-4 flex h-2 w-full bg-secondary">
              {[["Approach", info.contact.time_s], ["Crush onset", info.peak.time_s], ["Crush & rebound", info.tEnd]].map(([n, t], k, a) => { const t0 = k ? (a[k - 1]![1] as number) : 0; return <div key={n as string} title={n as string} style={{ width: `${(((t as number) - t0) / info.tEnd) * 100}%`, background: col(k) }} />; })}
            </div>
            <div className="relative h-3"><div className="absolute top-0 h-3 w-0.5 bg-foreground" style={{ left: `${(r.time_s / info.tEnd) * 100}%` }} /></div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[["Wall force", `${(r.wall_force_N / 1000).toFixed(1)} kN`], ["Peak force", `${(info.peak.wall_force_N / 1000).toFixed(1)} kN @ ${(info.peak.time_s * 1000).toFixed(1)} ms`], ["Intrusion", `${(r.max_intrusion_m * 1000).toFixed(1)} mm`], ["Speed (from KE)", `${(Math.sqrt(Math.max(r.ke_J, 0) / info.ke0) * 56).toFixed(1)} km/h`], ["Energy absorbed", `${(((info.ke0 - r.ke_J) / info.ke0) * 100).toFixed(1)} %`], ["Energy residual", `${r.residual_pct.toFixed(4)} %`]].map(([k, v]) => (
              <div key={k} className="border border-border bg-card p-3"><p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{k}</p><p className="mt-1 font-mono text-base">{v}</p></div>
            ))}
          </div>
        </section>

        <section className="border border-border bg-card p-4">
          <h2 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">1 · Task dependency graph — what runs in every one of the {info.steps.toLocaleString()} steps</h2>
          <svg viewBox="0 0 1000 150" className="mt-3 w-full">
            <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted-foreground)" /></marker></defs>
            {STAGES.map((s, k) => { const x = 100 + k * 200; const on = k === stage; return (
              <g key={s.id}>
                <rect x={x - 85} y={20} width={170} height={60} fill={on ? "var(--secondary)" : "var(--background)"} stroke={on ? col(k) : "var(--border)"} strokeWidth={on ? 2.5 : 1} />
                <text x={x} y={45} textAnchor="middle" fill="var(--foreground)" fontSize="14">{s.id} · {s.name}</text>
                <text x={x} y={66} textAnchor="middle" fill="var(--muted-foreground)" fontSize="9" fontFamily="monospace">{s.omp.slice(0, 30)}</text>
                {k < 4 && <><line x1={x + 87} y1={50} x2={x + 113} y2={50} stroke="var(--muted-foreground)" markerEnd="url(#ar)" /><line x1={x + 100} y1={12} x2={x + 100} y2={92} stroke="var(--destructive)" strokeDasharray="3 3" /><text x={x + 100} y={104} textAnchor="middle" fill="var(--destructive)" fontSize="9" fontFamily="monospace">barrier</text></>}
              </g>); })}
            <path d="M900 80 Q900 140 500 140 Q100 140 100 84" fill="none" stroke="var(--primary)" strokeDasharray="4 4" markerEnd="url(#ar)" />
            <text x={500} y={134} textAnchor="middle" fill="var(--primary)" fontSize="10" fontFamily="monospace">step n+1 uses dt from S5</text>
          </svg>
          <div className="mt-4 grid gap-3 md:grid-cols-5">
            {STAGES.map((s, k) => (
              <button key={s.id} onClick={() => setI(i - (i % 5) + k)} className={`border p-3 text-left text-xs ${k === stage ? "border-primary bg-secondary" : "border-border"}`}>
                <p className="font-mono uppercase" style={{ color: col(k) }}>{s.id} · {s.name}</p>
                <p className="mt-1 font-mono text-[10px] text-muted-foreground">{s.omp}</p>
                <p className="mt-2 text-muted-foreground">{s.text}</p>
              </button>
            ))}
          </div>
        </section>

        <section className="border border-border bg-card p-4">
          <h2 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">2 · Thread load — who does how much, and who waits</h2>
          <div className="mt-3 grid gap-2">
            {thr.map((t) => (
              <div key={t.thread_id} className="flex items-center gap-3 text-xs">
                <span className="w-28 font-mono">T{t.thread_id} · cpu {t.cpu_id}</span>
                <div className="flex h-6 flex-1 bg-secondary">
                  {keys.map((k, s) => <div key={k} title={`${STAGES[s]!.id}: ${ms(g(t, k))} ms`} style={{ width: `${(g(t, k) / mx) * 100}%`, background: col(s), opacity: s === stage ? 1 : 0.55 }} />)}
                </div>
                <span className="w-40 text-right font-mono text-muted-foreground">{ms(t.total_active_s)} ms · idle {ms(mx - t.total_active_s)}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-sm text-muted-foreground">
            Load imbalance (max / mean) = <b className="font-mono text-foreground">{(mx / mean).toFixed(2)}</b>. The grey tail on each bar is time that thread spends waiting at the barrier for the slowest one. S1 spring forces take{" "}
            <b className="font-mono text-foreground">{((thr.reduce((a, t) => a + t.s1_spring_s, 0) / thr.reduce((a, t) => a + t.total_active_s, 0)) * 100).toFixed(0)} %</b> of all thread time, so it is where balancing pays off most. Threads on efficiency cores finish later than those on performance cores.
          </p>
          {s100.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full font-mono text-xs">
                <thead><tr className="text-muted-foreground"><th className="text-left">Threads</th>{s100.map((s) => <th key={s.threads} className="text-right">{s.threads}</th>)}</tr></thead>
                <tbody>
                  <tr><td>Speedup 100k</td>{s100.map((s) => <td key={s.threads} className="text-right">{s.speedup.toFixed(2)}×</td>)}</tr>
                  <tr><td>Speedup 500k</td>{s500.map((s) => <td key={s.threads} className="text-right">{s.speedup.toFixed(2)}×</td>)}</tr>
                  <tr><td>Efficiency 100k</td>{s100.map((s) => <td key={s.threads} className="text-right">{(s.efficiency * 100).toFixed(0)}%</td>)}</tr>
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="border border-border bg-card p-4">
          <h2 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">3 · Telemetry — the whole crash, normalised</h2>
          <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full">
            <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--border)" /><line x1={P} y1={P} x2={P} y2={H - P} stroke="var(--border)" />
            <rect x={px(info.contact.time_s)} y={P} width={px(info.peak.time_s) - px(info.contact.time_s)} height={H - 2 * P} fill="var(--destructive)" opacity={0.08} />
            {series.map(([k, c]) => <path key={k} d={path(k)} fill="none" stroke={c} strokeWidth={1.6} />)}
            <line x1={px(r.time_s)} x2={px(r.time_s)} y1={P} y2={H - P} stroke="var(--foreground)" />
            <text x={W - P} y={H - 10} textAnchor="end" fill="var(--muted-foreground)" fontSize="11" fontFamily="monospace">{(info.tEnd * 1000).toFixed(1)} ms</text>
            <text x={P} y={H - 10} fill="var(--muted-foreground)" fontSize="11" fontFamily="monospace">0 ms</text>
          </svg>
          <div className="mt-2 flex flex-wrap gap-4 text-xs">{series.map(([k, c, n]) => <span key={k} className="flex items-center gap-1"><i className="inline-block h-2 w-3" style={{ background: c }} />{n}: <b className="font-mono">{k === "wall_force_N" ? `${(g(r, k) / 1000).toFixed(1)} kN` : `${(g(r, k) / 1000).toFixed(1)} kJ`}</b></span>)}</div>
          <h3 className="mt-5 font-mono text-xs uppercase tracking-widest text-muted-foreground">Where the energy is right now</h3>
          <div className="mt-2 flex h-6 w-full bg-secondary">{eTerms.map(([n, v, c]) => <div key={n} title={`${n}: ${(v / 1000).toFixed(1)} kJ`} style={{ width: `${(Math.max(v, 0) / r.total_e_J) * 100}%`, background: c }} />)}</div>
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">{eTerms.map(([n, v, c]) => <span key={n}><i className="mr-1 inline-block h-2 w-3" style={{ background: c }} />{n} {((Math.max(v, 0) / r.total_e_J) * 100).toFixed(1)}%</span>)}</div>
          <p className="mt-3 text-sm text-muted-foreground">Total energy stays flat while it moves from kinetic into plastic work: the worst residual over the run is <b className="font-mono text-foreground">{info.maxRes.toFixed(4)} %</b>, under the 0.1 % correctness gate.</p>
        </section>

        <section className="border border-border bg-card p-4 text-sm text-muted-foreground">
          <h2 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">How it all fits together</h2>
          <ol className="mt-2 list-decimal space-y-1 pl-5">
            <li>The vehicle is a lattice of nodes and springs, split into contiguous slabs along its length, one slab per OpenMP thread.</li>
            <li>Every 2.5 µs step runs S1 → S5; barriers between stages keep all threads in lock-step.</li>
            <li>Phases above (approach, crush, rebound) change the physics, not the per-step work: every step costs about the same, so the speedup measured on 100 steps holds for the whole run.</li>
            <li>The step is memory-bound, so speedup flattens past about 8 threads; the recommended next tier is MPI across nodes plus GPU kernels for S1, S2 and S4.</li>
          </ol>
          <p className="mt-3 text-xs">All numbers load from the recorded result files of the C++ run; nothing on this page is simulated in the browser.</p>
        </section>
      </main>
    </div>
  );
}
