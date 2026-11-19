import { Link } from "@tanstack/react-router";

export function SiteNav() {
  const cls = "px-3 py-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground transition-colors";
  return (
    <nav className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-card px-4">
      <span className="font-mono text-xs font-semibold text-primary">CRASH-SIM · 24ISE503</span>
      <div className="flex gap-1">
        <Link to="/" className={cls} activeProps={{ className: "bg-secondary text-foreground" }} activeOptions={{ exact: true }}>3D Studio</Link>
        <Link to="/analysis" className={cls} activeProps={{ className: "bg-secondary text-foreground" }}>Analysis</Link>
        <Link to="/results" className={cls} activeProps={{ className: "bg-secondary text-foreground" }}>Results</Link>
      </div>
    </nav>
  );
}
