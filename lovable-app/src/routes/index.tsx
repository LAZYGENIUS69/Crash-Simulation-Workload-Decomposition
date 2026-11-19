import { createFileRoute } from "@tanstack/react-router";
import { SiteNav } from "@/components/SiteNav";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "3D Crash Studio | Crash-Simulation Workload Decomposition" },
      { name: "description", content: "Interactive WebGL replay of a 56 km/h vehicle crash from an explicit C++17/OpenMP solver." },
      { property: "og:title", content: "3D Crash Studio | Crash-Simulation Workload Decomposition" },
      { property: "og:description", content: "Interactive WebGL replay of a 56 km/h vehicle crash from an explicit C++17/OpenMP solver." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="flex h-screen flex-col bg-background">
      <SiteNav />
      <iframe src="/studio/index.html?v=3" title="3D crash simulator" className="w-full flex-1 border-0" />
    </div>
  );
}
