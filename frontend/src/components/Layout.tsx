import { NavLink } from "react-router-dom";
import type { PropsWithChildren } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "../api/client";

const navItems = [
  { to: "/search", label: "Search" },
  { to: "/ingest", label: "Ingest" },
  { to: "/manage", label: "Manage" },
];

export default function Layout({ children }: PropsWithChildren) {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 5000,
  });
  const h = health?.data;

  return (
    <div className="flex min-h-screen bg-forge-bg text-slate-200">
      <aside className="w-56 bg-forge-panel border-r border-forge-edge flex flex-col">
        <div className="px-5 py-4 border-b border-forge-edge">
          <div className="text-lg font-bold text-forge-accent">ForgeRAG</div>
          <div className="text-xs text-slate-400">
            Engineering knowledge graph
          </div>
        </div>
        <nav className="py-4">
          {navItems.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                `block px-5 py-2 text-sm hover:bg-forge-edge transition ${
                  isActive ? "bg-forge-edge text-forge-accent" : ""
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto px-5 py-4 border-t border-forge-edge text-xs space-y-1">
          <StatusDot label="Neo4j" ok={!!h?.neo4j_connected} />
          <StatusDot label="GPU" ok={!!h?.gpu_available} />
          {h?.details?.vram_free_gb != null && (
            <div className="text-slate-500">
              VRAM {h.details.vram_free_gb}/{h.details.vram_total_gb} GB free
            </div>
          )}
          <div className="text-slate-500">
            {h?.document_count ?? 0} docs · {h?.page_count ?? 0} pages
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}

function StatusDot({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-block h-2 w-2 rounded-full ${
          ok ? "bg-emerald-500" : "bg-rose-500"
        }`}
      />
      <span>{label}</span>
    </div>
  );
}
