import { useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { pageImageUrl } from "../api/client";

export default function Viewer() {
  const { hash, page } = useParams<{ hash: string; page: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const pageNum = parseInt(page || "1", 10);
  const returnTo = searchParams.get("from") || "/search";

  const [zoom, setZoom] = useState<"fit" | "full">("fit");

  if (!hash) {
    return <div className="p-6 text-forge-muted">Missing document hash.</div>;
  }

  const imgSrc = pageImageUrl(hash, pageNum);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="bg-forge-panel border-b border-forge-edge px-4 py-2 flex items-center gap-3 shrink-0">
        <button
          onClick={() => navigate(returnTo)}
          className="text-sm border border-forge-edge rounded px-3 py-1 hover:bg-forge-edge"
        >
          ← Back to search
        </button>

        <div className="flex items-center gap-1 ml-4">
          <button
            onClick={() => pageNum > 1 && navigate(`/view/${hash}/${pageNum - 1}?from=${encodeURIComponent(returnTo)}`)}
            disabled={pageNum <= 1}
            className="text-sm border border-forge-edge rounded px-3 py-1 hover:bg-forge-edge disabled:opacity-30"
          >
            ← Prev
          </button>
          <span className="font-mono text-sm px-3">Page {pageNum}</span>
          <button
            onClick={() => navigate(`/view/${hash}/${pageNum + 1}?from=${encodeURIComponent(returnTo)}`)}
            className="text-sm border border-forge-edge rounded px-3 py-1 hover:bg-forge-edge"
          >
            Next →
          </button>
        </div>

        <div className="flex items-center gap-1 ml-4">
          <button
            onClick={() => setZoom("fit")}
            className={`text-xs border rounded px-2 py-1 ${
              zoom === "fit"
                ? "border-forge-primary text-forge-primary"
                : "border-forge-edge hover:bg-forge-edge"
            }`}
          >
            Fit
          </button>
          <button
            onClick={() => setZoom("full")}
            className={`text-xs border rounded px-2 py-1 ${
              zoom === "full"
                ? "border-forge-primary text-forge-primary"
                : "border-forge-edge hover:bg-forge-edge"
            }`}
          >
            Full size
          </button>
        </div>

        <a
          href={imgSrc}
          download={`page_${pageNum}.png`}
          className="text-xs text-forge-secondary hover:underline ml-auto"
        >
          Download PNG
        </a>
      </div>

      {/* Image viewport */}
      <div className="flex-1 overflow-auto bg-forge-bg p-4">
        <img
          src={imgSrc}
          alt={`Page ${pageNum}`}
          className={`mx-auto border border-forge-edge bg-white ${
            zoom === "fit" ? "max-w-full max-h-[calc(100vh-80px)]" : ""
          }`}
          onError={(e) => {
            (e.target as HTMLImageElement).alt = `Page ${pageNum} not found`;
          }}
        />
      </div>
    </div>
  );
}
