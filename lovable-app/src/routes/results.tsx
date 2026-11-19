import { createFileRoute } from "@tanstack/react-router";
import { SiteNav } from "@/components/SiteNav";
import scaling from "@/assets/figures/scaling_curves.png.asset.json";
import amdahl from "@/assets/figures/amdahl_fit.png.asset.json";
import vScaling from "@/assets/figures/vehicle_scaling_curves.png.asset.json";
import stage from "@/assets/figures/stage_breakdown.png.asset.json";
import pulse from "@/assets/figures/vehicle_crash_pulse.png.asset.json";
import timelapse from "@/assets/figures/vehicle_timelapse_strip.png.asset.json";
import partitions from "@/assets/figures/vehicle_thread_partitions.png.asset.json";
import strain from "@/assets/figures/vehicle_plastic_strain_hud.png.asset.json";
import dep from "@/assets/figures/dep_graph.png.asset.json";
import partView from "@/assets/figures/partition_view.png.asset.json";
import assembly from "@/assets/figures/assembly_scaling_comparison.png.asset.json";
import energy from "@/assets/figures/energy_closure.png.asset.json";

const metrics = [
  { label: "Elastic energy drift", value: "≤ 0.00092%", note: "1,000 steps, E₀ = 32,844.94 J" },
  { label: "Full crash energy closure", value: "≤ 0.010%", note: "Wall impact, damping & buckling" },
  { label: "CSR gather speedup (100k)", value: "2.17×", note: "Private slabs spill 12 MiB L3" },
  { label: "CSR gather speedup (500k)", value: "2.11× / 2.37×", note: "FP64 / FP32" },
  { label: "STREAM Triad peak", value: "35.32 GB/s", note: "4 OpenMP threads" },
  { label: "Python ↔ C++ agreement", value: "8.16e-9", note: "Relative error" },
];

const figures = [
  { src: assembly.url, title: "Force assembly strategies", caption: "Private slabs vs atomics vs node-centric CSR gather." },
  { src: vScaling.url, title: "Vehicle mesh scaling", caption: "Speedup and efficiency across OpenMP thread counts." },
  { src: energy.url, title: "Energy closure audit", caption: "Term-by-term balance over the crash event." },
  { src: pulse.url, title: "Crash pulse", caption: "Wall force and intrusion at 56 km/h." },
  { src: timelapse.url, title: "Deformation timelapse", caption: "Frontal crumple sequence." },
  { src: strain.url, title: "Plastic strain", caption: "Elastoplastic yield distribution." },
  { src: partitions.url, title: "Thread partitions", caption: "Spatial workload decomposition per thread." },
  { src: stage.url, title: "Stage breakdown", caption: "Spring, assembly, contact, integrate, CFL." },
  { src: scaling.url, title: "Scaling curves", caption: "Strong scaling on the lattice mesh." },
  { src: amdahl.url, title: "Amdahl fit", caption: "Serial fraction estimate." },
  { src: dep.url, title: "Dependency graph", caption: "Per-step task dependencies." },
  { src: partView.url, title: "Partition view", caption: "Mesh partition layout." },
];

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Results | Crash-Simulation Workload Decomposition" },
      { name: "description", content: "Parallel scaling, assembly trade-offs, and energy conservation results for the OpenMP crash solver." },
      { property: "og:title", content: "Results | Crash-Simulation Workload Decomposition" },
      { property: "og:description", content: "Parallel scaling, assembly trade-offs, and energy conservation results for the OpenMP crash solver." },
      { property: "og:type", content: "article" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Results,
});

function Results() {
  return (
    <div className="min-h-screen bg-background">
      <SiteNav />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="text-3xl font-bold tracking-tight">Crash-Simulation Workload Decomposition</h1>
        <p className="mt-2 max-w-3xl text-muted-foreground">
          Explicit 3D vehicle crash solver in C++17 + OpenMP. Spring-mass lattice, elastoplastic buckling,
          penalty barrier and spatial-hash self-contact, profiled on a 12-thread Intel i5-1334U.
        </p>
        <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {metrics.map((m) => (
            <div key={m.label} className="rounded-lg border border-border bg-card p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">{m.label}</div>
              <div className="mt-1 font-mono text-2xl font-semibold text-accent">{m.value}</div>
              <div className="mt-1 text-sm text-muted-foreground">{m.note}</div>
            </div>
          ))}
        </section>
        <section className="mt-10 grid gap-6 md:grid-cols-2">
          {figures.map((f) => (
            <figure key={f.title} className="overflow-hidden rounded-lg border border-border bg-card">
              <img src={f.src} alt={f.title} loading="lazy" className="w-full bg-foreground" />
              <figcaption className="p-4">
                <div className="font-semibold">{f.title}</div>
                <div className="text-sm text-muted-foreground">{f.caption}</div>
              </figcaption>
            </figure>
          ))}
        </section>
      </main>
    </div>
  );
}
