import { useState, useEffect } from "react";
import { Link } from "react-router";
import { parse, isValid } from "date-fns";
import { Search, ArrowDown, ArrowUp, Loader2, AlertTriangle, RefreshCw } from "lucide-react";
import {
  getInteractionDetail,
  getInteractions,
  reprocessInteraction,
  type InteractionSummary,
} from "../../services/api";

function getFailurePreview(row: InteractionSummary): string {
  const failures = row.processingFailures || [];
  if (!failures.length) {
    return String(row.status).toLowerCase() === "failed"
      ? "Processing failed (no error text stored)."
      : "";
  }

  const grouped = new Map<string, { message: string; stages: string[] }>();
  for (const failure of failures) {
    const stage = (failure.stage || "unknown").trim() || "unknown";
    const message = (failure.errorMessage || "Unknown error").trim() || "Unknown error";
    const key = message.toLowerCase();
    const current = grouped.get(key);
    if (current) {
      current.stages.push(stage);
      continue;
    }
    grouped.set(key, { message, stages: [stage] });
  }

  return Array.from(grouped.values())
    .map((group) =>
      group.stages.length > 1
        ? `${group.stages.join(", ")}: ${group.message}`
        : `${group.stages[0]}: ${group.message}`,
    )
    .join(" | ");
}

function isFailedStatus(row: InteractionSummary): boolean {
  return String(row.status || "").toLowerCase() === "failed";
}

/** Backend sends `date` as YYYY-MM-DD and `time` as 12h (e.g. strftime "%I:%M %p"). */
function interactionDateMs(row: InteractionSummary): number {
  const datePart = (row.date || "").trim();
  const timePart = (row.time || "").trim();
  if (!datePart) return 0;

  if (timePart) {
    const combined = `${datePart} ${timePart}`;
    const withPaddedHour = parse(combined, "yyyy-MM-dd hh:mm a", new Date());
    if (isValid(withPaddedHour)) return withPaddedHour.getTime();
    const withHour = parse(combined, "yyyy-MM-dd h:mm a", new Date());
    if (isValid(withHour)) return withHour.getTime();
  }

  const dateOnly = parse(datePart, "yyyy-MM-dd", new Date());
  return isValid(dateOnly) ? dateOnly.getTime() : 0;
}

export function SessionInspector() {
  const [interactions, setInteractions] = useState<InteractionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reprocessingIds, setReprocessingIds] = useState<Set<string>>(new Set());

  const [searchQuery, setSearchQuery] = useState("");

  const [sortField, setSortField] = useState<"score" | "date" | "duration">("score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  useEffect(() => {
    getInteractions()
      .then(setInteractions)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSort = (field: "score" | "date" | "duration") => {
    if (sortField === field) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
    setCurrentPage(1);
  };

  const handleReprocess = async (interactionId: string) => {
    setActionError(null);
    setReprocessingIds((prev) => new Set(prev).add(interactionId));
    try {
      await reprocessInteraction(interactionId);
      const refreshed = await getInteractions();
      setInteractions(refreshed);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reprocess interaction";
      if (message.includes("409")) {
        try {
          await reprocessInteraction(interactionId, { force: true });
          const refreshed = await getInteractions();
          setInteractions(refreshed);
          return;
        } catch {
          setActionError("This interaction is already processing. Please wait and try again.");
        }
      } else {
        setActionError(message);
      }
    } finally {
      setReprocessingIds((prev) => {
        const next = new Set(prev);
        next.delete(interactionId);
        return next;
      });
    }
  };

  const filteredInteractions = interactions.filter((interaction) => {
    const searchLower = searchQuery.toLowerCase();
    return (
      interaction.agentName.toLowerCase().includes(searchLower) ||
      interaction.id.toLowerCase().includes(searchLower) ||
      interaction.date.toLowerCase().includes(searchLower)
    );
  });

  const sortedInteractions = [...filteredInteractions].sort((a, b) => {
    let comparison = 0;
    if (sortField === "score") {
      comparison = a.overallScore - b.overallScore;
    } else if (sortField === "date") {
      const dateA = interactionDateMs(a);
      const dateB = interactionDateMs(b);
      comparison = dateA - dateB;
      if (comparison === 0) {
        comparison = a.id.localeCompare(b.id);
      }
    } else if (sortField === "duration") {
      const [mA, sA] = a.duration.split(":").map(Number);
      const [mB, sB] = b.duration.split(":").map(Number);
      const durA = (mA || 0) * 60 + (sA || 0);
      const durB = (mB || 0) * 60 + (sB || 0);
      comparison = durA - durB;
    }
    return sortOrder === "asc" ? -comparison : comparison;
  });

  const totalItems = sortedInteractions.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedInteractions = sortedInteractions.slice(startIndex, startIndex + itemsPerPage);

  const sortIndicator = (field: "score" | "date" | "duration") => {
    if (sortField !== field) return null;
    return sortOrder === "asc" ? (
      <ArrowUp className="ml-1 h-3.5 w-3.5" />
    ) : (
      <ArrowDown className="ml-1 h-3.5 w-3.5" />
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading interactions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-destructive mx-auto mb-3" />
          <p className="text-foreground text-sm">Failed to load interactions</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 min-w-0 max-w-full">
      {actionError && (
        <div className="mb-4 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12px] font-medium text-destructive">
          {actionError}
        </div>
      )}
      {/* Top Controls */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-[20px] font-bold text-foreground mb-2">Session Inspector</h2>
          <p className="text-[13px] text-muted-foreground">
            {totalItems} interaction{totalItems !== 1 ? "s" : ""} · sorted by {sortField} (
            {sortOrder === "asc" ? "descending" : "ascending"})
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search agent, date, ID…"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="w-[200px] h-10 pl-9 pr-3 bg-muted/20 border border-border rounded-[10px] text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
          </div>

          <div className="flex items-center border border-border rounded-[10px] overflow-hidden bg-card">
            <button
              type="button"
              onClick={() => handleSort("score")}
              className={`flex items-center px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "score" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Score
              {sortIndicator("score")}
            </button>
            <button
              type="button"
              onClick={() => handleSort("date")}
              className={`flex items-center px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "date" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Date
              {sortIndicator("date")}
            </button>
            <button
              type="button"
              onClick={() => handleSort("duration")}
              className={`flex items-center px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "duration" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Duration
              {sortIndicator("duration")}
            </button>
          </div>
        </div>
      </div>

      {/* Table: horizontal scroll + sticky Actions so Inspect stays visible with a narrow main column */}
      <div className="bg-card rounded-[14px] border border-border">
        <div className="overflow-x-auto rounded-t-[14px]">
          <table className="w-full min-w-[56rem] border-collapse">
          <thead>
            <tr className="bg-muted/10 border-b border-border">
              <th className="px-4 py-4 text-left text-label">Agent</th>
              <th className="px-4 py-4 text-left text-label">Date & Time</th>
              <th className="px-4 py-4 text-left text-label">Duration</th>
              <th className="px-4 py-4 text-left text-label">Score</th>
              <th className="px-4 py-4 text-left text-label">Empathy</th>
              <th className="px-4 py-4 text-left text-label">Policy</th>
              <th className="px-4 py-4 text-left text-label">Resolution</th>
              <th className="px-4 py-4 text-center text-label">Status</th>
              <th className="px-4 py-4 text-left text-label w-48 max-w-[12rem]">Error</th>
              <th className="px-3 py-4 text-center text-label sticky right-0 z-30 w-28 min-w-[7rem] border-l border-border bg-muted/10 shadow-[-6px_0_10px_-6px_rgba(0,0,0,0.12)]">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {paginatedInteractions.map((row) => (
              <tr
                key={row.id}
                className={`group hover:bg-muted/5 transition-colors ${isFailedStatus(row) ? "bg-destructive/[0.04]" : ""}`}
              >
                <td className="px-4 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    <span className="text-[14px] font-semibold text-foreground">{row.agentName}</span>
                    {row.hasViolation && (
                      <span className="px-2 py-0.5 bg-destructive/10 text-destructive rounded-full text-[11px] font-medium">
                        ⚠ Violation
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-[13px] text-muted-foreground">
                  {row.date} · {row.time}
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-[13px] text-muted-foreground">
                  {row.duration}
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <div
                    className="text-[18px] font-bold"
                    style={{
                      fontFamily: "var(--font-serif)",
                      color: row.overallScore >= 85 ? "var(--success)" : row.overallScore >= 75 ? "var(--primary)" : "var(--destructive)",
                    }}
                  >
                    {row.overallScore}%
                  </div>
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-[13px] text-foreground">{row.empathyScore}</td>
                <td className="px-4 py-4 whitespace-nowrap text-[13px] text-foreground">{row.policyScore}</td>
                <td className="px-4 py-4 whitespace-nowrap text-[13px] text-foreground">{row.resolutionScore}</td>
                <td className="px-4 py-4 whitespace-nowrap text-center align-middle">
                  <div className="flex justify-center">
                    <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${row.resolved ? "bg-success/5 text-success border-success/20" : "bg-destructive/5 text-destructive border-destructive/20"}`}>
                      {(row.status || "").toLowerCase() === "failed"
                        ? "⚠ Failed"
                        : (row.status || "").toLowerCase() === "processing" || (row.status || "").toLowerCase() === "pending"
                          ? "⟳ Processing"
                          : row.resolved
                            ? "✓ Resolved"
                            : "✗ Unresolved"}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-4 text-[12px] text-muted-foreground max-w-[12rem] w-[12rem] align-top">
                  <span className="break-words line-clamp-4" title={getFailurePreview(row) || undefined}>
                    {isFailedStatus(row) || (row.processingFailures && row.processingFailures.length > 0)
                      ? getFailurePreview(row) || "—"
                      : "—"}
                  </span>
                </td>
                <td
                  className={`sticky right-0 z-20 whitespace-nowrap border-l border-border/80 px-3 py-4 text-center align-middle shadow-[-6px_0_10px_-6px_rgba(0,0,0,0.1)] ${
                    isFailedStatus(row)
                      ? "bg-destructive/[0.04] group-hover:bg-muted/5"
                      : "bg-card group-hover:bg-muted/5"
                  }`}
                >
                  <div className="flex items-center justify-center">
                    {isFailedStatus(row) ? (
                      <button
                        type="button"
                        onClick={() => void handleReprocess(row.id)}
                        disabled={reprocessingIds.has(row.id)}
                        className="inline-flex h-8 items-center gap-1 rounded-lg border border-border px-2.5 text-[11px] font-semibold text-foreground hover:bg-muted disabled:opacity-50"
                      >
                        <RefreshCw className={`h-3.5 w-3.5 ${reprocessingIds.has(row.id) ? "animate-spin" : ""}`} />
                        {reprocessingIds.has(row.id) ? "Reprocessing" : "Reprocess"}
                      </button>
                    ) : (
                      <Link
                        to={`/manager/inspector/${row.id}`}
                        onMouseEnter={() => {
                          void getInteractionDetail(row.id).catch(() => undefined);
                        }}
                        className="inline-flex h-8 items-center justify-center rounded-lg px-2.5 text-primary hover:text-primary/80 font-semibold text-[13px] transition-colors"
                      >
                        Inspect
                      </Link>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>

        {/* Pagination */}
        <div className="px-6 py-4 bg-muted/5 border-t border-border flex items-center justify-between rounded-b-[14px]">
          <div className="text-[13px] text-muted-foreground font-medium">
            Showing {startIndex + 1}–{Math.min(startIndex + itemsPerPage, totalItems)} of {totalItems}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="h-9 px-4 rounded-xl border border-border bg-background text-[13px] font-semibold text-foreground hover:bg-muted disabled:opacity-40 transition-all flex items-center gap-2"
            >
              ← Prev
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || totalItems === 0}
              className="h-9 px-4 rounded-xl border border-border bg-background text-[13px] font-semibold text-foreground hover:bg-muted disabled:opacity-40 transition-all flex items-center gap-2"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
