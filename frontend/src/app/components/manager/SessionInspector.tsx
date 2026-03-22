import { useState, useEffect } from "react";
import { Link } from "react-router";
import { Search, ChevronDown, Loader2, AlertTriangle } from "lucide-react";
import { getInteractions, type InteractionSummary } from "../../services/api";

export function SessionInspector() {
  const [interactions, setInteractions] = useState<InteractionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<"score" | "date" | "duration">("score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
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
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
    setCurrentPage(1);
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
      const dateA = new Date(`${a.date}T${a.time}`).getTime();
      const dateB = new Date(`${b.date}T${b.time}`).getTime();
      comparison = dateA - dateB;
    } else if (sortField === "duration") {
      const [mA, sA] = a.duration.split(":").map(Number);
      const [mB, sB] = b.duration.split(":").map(Number);
      const durA = (mA || 0) * 60 + (sA || 0);
      const durB = (mB || 0) * 60 + (sB || 0);
      comparison = durA - durB;
    }
    return sortOrder === "asc" ? comparison : -comparison;
  });

  const totalItems = sortedInteractions.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedInteractions = sortedInteractions.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="p-6">
      {/* Top Controls */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-[20px] font-bold text-foreground mb-2">Session Inspector</h2>
          <p className="text-[13px] text-muted-foreground">
            {totalItems} interaction{totalItems !== 1 ? "s" : ""} · sorted by score
          </p>
        </div>

        <div className="flex items-center gap-3">
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

          <button className="flex items-center gap-2 h-10 px-4 bg-card border border-border rounded-[10px] text-[13px] hover:bg-muted transition-colors">
            All Agents
            <ChevronDown className="w-4 h-4" />
          </button>

          <div className="flex items-center border border-border rounded-[10px] overflow-hidden bg-card">
            <button
              onClick={() => handleSort("score")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "score" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Score ↓
            </button>
            <button
              onClick={() => handleSort("date")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "date" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Date
            </button>
            <button
              onClick={() => handleSort("duration")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${sortField === "duration" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted"}`}
            >
              Duration
            </button>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-card rounded-[14px] border border-border overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-muted/10 border-b border-border">
              <th className="px-6 py-4 text-left text-label">Agent</th>
              <th className="px-6 py-4 text-left text-label">Date & Time</th>
              <th className="px-6 py-4 text-left text-label">Duration</th>
              <th className="px-6 py-4 text-left text-label">Score</th>
              <th className="px-6 py-4 text-left text-label">Empathy</th>
              <th className="px-6 py-4 text-left text-label">Policy</th>
              <th className="px-6 py-4 text-left text-label">Resolution</th>
              <th className="px-6 py-4 text-left text-label">Status</th>
              <th className="px-6 py-4 text-left text-label">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {paginatedInteractions.map((row) => (
              <tr key={row.id} className="hover:bg-muted/5 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    <span className="text-[14px] font-semibold text-foreground">{row.agentName}</span>
                    {row.hasViolation && (
                      <span className="px-2 py-0.5 bg-destructive/10 text-destructive rounded-full text-[11px] font-medium">
                        ⚠ Violation
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] text-muted-foreground">
                  {row.date} · {row.time}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] text-muted-foreground">
                  {row.duration}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
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
                <td className="px-6 py-4 whitespace-nowrap text-[13px] text-foreground">{row.empathyScore}</td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] text-foreground">{row.policyScore}</td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] text-foreground">{row.resolutionScore}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${row.resolved ? "bg-success/5 text-success border-success/20" : "bg-destructive/5 text-destructive border-destructive/20"}`}>
                    {row.resolved ? "✓ Resolved" : "✗ Unresolved"}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <Link
                    to={`/manager/inspector/${row.id}`}
                    className="text-primary hover:text-primary/80 font-semibold text-[13px] transition-colors"
                  >
                    Inspect →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination */}
        <div className="px-6 py-4 bg-muted/5 border-t border-border flex items-center justify-between">
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
