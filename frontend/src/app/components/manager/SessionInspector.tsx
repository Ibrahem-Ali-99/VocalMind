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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-[#3B82F6] animate-spin" />
        <span className="ml-3 text-[#6B7280] text-sm">Loading interactions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-[#6B7280] text-sm">Failed to load interactions</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const handleSort = (field: "score" | "date" | "duration") => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
    setCurrentPage(1);
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
            <div className="text-label mb-2">
              SESSION INSPECTOR
            </div>
          <p className="text-[13px] text-[#6B7280]">
            {totalItems} interaction{totalItems !== 1 ? "s" : ""} · sorted by {sortField}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-label-foreground" />
            <input
              type="text"
              placeholder="Search agent, date, ID…"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1); // Reset page on search
              }}
              className="w-[200px] h-10 pl-9 pr-3 bg-input border border-border rounded-[10px] text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all shadow-inner"
            />
          </div>

          <button className="flex items-center gap-2 h-10 px-4 bg-card border border-border rounded-[10px] text-[13px] text-foreground hover:bg-muted transition-colors">
            All Agents
            <ChevronDown className="w-4 h-4" />
          </button>

          <div className="flex items-center border border-border rounded-[10px] overflow-hidden bg-card shadow-inner">
            <button
              onClick={() => handleSort("score")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "score"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Score {sortField === "score" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
            <button
              onClick={() => handleSort("date")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "date"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Date {sortField === "date" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
            <button
              onClick={() => handleSort("duration")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "duration"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted"
              }`}
            >
              Duration {sortField === "duration" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
          </div>
        </div>
      </div>

      {/* Interaction Table */}
      <div className="bg-card rounded-[14px] border border-border transition-all overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-12 gap-4 px-5 py-3 border-b border-border bg-background/50">
          <div className="col-span-2 text-label">
            Agent
          </div>
          <div className="col-span-2 text-label">
            Date & Time
          </div>
          <div className="col-span-1 text-label">
            Duration
          </div>
          <div className="col-span-1 text-label">
            Score
          </div>
          <div className="col-span-1 text-label">
            Empathy
          </div>
          <div className="col-span-1 text-label">
            Policy
          </div>
          <div className="col-span-1 text-label">
            Resolution
          </div>
          <div className="col-span-2 text-label">
            Status
          </div>
          <div className="col-span-1 text-label">
            Actions
          </div>
        </div>

        {/* Rows */}
        <div className="divide-y divide-border">
          {paginatedInteractions.length === 0 ? (
            <div className="px-5 py-8 text-center text-[#6B7280] text-[13px]">
              No interactions found matching your criteria.
            </div>
          ) : (
            paginatedInteractions.map((interaction) => (
              <div
                key={interaction.id}
                className="grid grid-cols-12 gap-4 px-5 py-4 hover:bg-muted/40 transition-colors border-b last:border-0 border-border"
              >
              {/* Agent */}
              <div className="col-span-2 flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-primary text-xs font-semibold flex-shrink-0">
                  {interaction.agentName.split(" ").map((n) => n[0]).join("")}
                </div>
                <span className="text-[13px] font-semibold text-foreground">
                  {interaction.agentName}
                </span>
              </div>

              {/* Date & Time */}
              <div className="col-span-2 flex items-center text-[13px] text-foreground">
                {interaction.date} · {interaction.time}
              </div>

              {/* Duration */}
              <div className="col-span-1 flex items-center text-[13px] text-[#6B7280]">
                {interaction.duration}
              </div>

              {/* Score */}
              <div className="col-span-1 flex items-center">
                <span
                  className={`px-2.5 py-1 rounded-full text-[13px] font-semibold ${
                    interaction.overallScore >= 85
                      ? "bg-success/10 text-success"
                      : interaction.overallScore >= 75
                      ? "bg-primary/10 text-primary"
                      : "bg-destructive/10 text-destructive"
                  }`}
                >
                  {interaction.overallScore}%
                </span>
              </div>

              {/* Empathy */}
              <div className="col-span-1 flex items-center text-[12px] text-[#374151]">
                {interaction.empathyScore}
              </div>

              {/* Policy */}
              <div className="col-span-1 flex items-center text-[12px] text-[#374151]">
                {interaction.policyScore}
              </div>

              {/* Resolution */}
              <div className="col-span-1 flex items-center text-[12px] text-muted-foreground">
                {interaction.resolutionScore}
              </div>

              {/* Status */}
              <div className="col-span-2 flex items-center gap-2">
                <span
                  className={`text-[12px] font-semibold ${
                    interaction.resolved ? "text-success" : "text-destructive"
                  }`}
                >
                  {interaction.resolved ? "✓ Resolved" : "✗ Unresolved"}
                </span>
                {interaction.hasViolation && (
                  <span className="px-2 py-0.5 bg-[#FEF3C7] text-[#92400E] rounded-full text-[11px] font-medium">
                    ⚠ Violation
                  </span>
                )}
              </div>

              {/* Actions */}
              <div className="col-span-1 flex items-center">
                <Link
                  to={`/manager/inspector/${interaction.id}`}
                  className="text-[12px] text-primary font-bold hover:underline"
                >
                  Inspect →
                </Link>
              </div>
            </div>
          )))}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-border flex items-center justify-between bg-muted/20">
          <span className="text-[12px] text-muted-foreground">
            Showing {totalItems === 0 ? 0 : startIndex + 1}–{Math.min(startIndex + itemsPerPage, totalItems)} of {totalItems}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 h-8 text-[12px] text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors font-medium"
            >
              ← Prev
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || totalItems === 0}
              className="px-3 h-8 text-[12px] text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors font-medium"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
