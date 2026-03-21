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
          <p className="text-[#9CA3AF] text-xs mt-1">{error}</p>
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
          <h2 className="text-[22px] font-bold text-[#111827] mb-1">
            Session Inspector
          </h2>
          <p className="text-[13px] text-[#6B7280]">
            {totalItems} interaction{totalItems !== 1 ? "s" : ""} · sorted by {sortField}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9CA3AF]" />
            <input
              type="text"
              placeholder="Search agent, date, ID…"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1); // Reset page on search
              }}
              className="w-[200px] h-10 pl-9 pr-3 bg-white border border-[#E5E7EB] rounded-[10px] text-[13px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]"
            />
          </div>

          <button className="flex items-center gap-2 h-10 px-4 bg-white border border-[#E5E7EB] rounded-[10px] text-[13px] text-[#374151] hover:bg-[#F9FAFB] transition-colors">
            All Agents
            <ChevronDown className="w-4 h-4" />
          </button>

          <div className="flex items-center border border-[#E5E7EB] rounded-[10px] overflow-hidden bg-white">
            <button
              onClick={() => handleSort("score")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "score"
                  ? "bg-[#3B82F6] text-white"
                  : "text-[#6B7280] hover:bg-[#F9FAFB]"
              }`}
            >
              Score {sortField === "score" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
            <button
              onClick={() => handleSort("date")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "date"
                  ? "bg-[#3B82F6] text-white"
                  : "text-[#6B7280] hover:bg-[#F9FAFB]"
              }`}
            >
              Date {sortField === "date" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
            <button
              onClick={() => handleSort("duration")}
              className={`px-3 h-10 text-[11px] font-semibold transition-colors ${
                sortField === "duration"
                  ? "bg-[#3B82F6] text-white"
                  : "text-[#6B7280] hover:bg-[#F9FAFB]"
              }`}
            >
              Duration {sortField === "duration" ? (sortOrder === "asc" ? "↑" : "↓") : ""}
            </button>
          </div>
        </div>
      </div>

      {/* Interaction Table */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] shadow-sm overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-12 gap-4 px-5 py-3 border-b border-[#E5E7EB] bg-[#F9FAFB]">
          <div className="col-span-2 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Agent
          </div>
          <div className="col-span-2 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Date & Time
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Duration
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Score
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Empathy
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Policy
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Resolution
          </div>
          <div className="col-span-2 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Status
          </div>
          <div className="col-span-1 text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
            Actions
          </div>
        </div>

        {/* Rows */}
        <div className="divide-y divide-[#E5E7EB]">
          {paginatedInteractions.length === 0 ? (
            <div className="px-5 py-8 text-center text-[#6B7280] text-[13px]">
              No interactions found matching your criteria.
            </div>
          ) : (
            paginatedInteractions.map((interaction) => (
              <div
                key={interaction.id}
                className="grid grid-cols-12 gap-4 px-5 py-4 hover:bg-[#F9FAFB] transition-colors"
              >
              {/* Agent */}
              <div className="col-span-2 flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-[#3B82F6] flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
                  {interaction.agentName.split(" ").map((n) => n[0]).join("")}
                </div>
                <span className="text-[13px] font-semibold text-[#111827]">
                  {interaction.agentName}
                </span>
              </div>

              {/* Date & Time */}
              <div className="col-span-2 flex items-center text-[13px] text-[#374151]">
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
                      ? "bg-[#ECFDF5] text-[#10B981]"
                      : interaction.overallScore >= 75
                      ? "bg-[#EFF6FF] text-[#3B82F6]"
                      : "bg-[#FFFBEB] text-[#F59E0B]"
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
              <div className="col-span-1 flex items-center text-[12px] text-[#374151]">
                {interaction.resolutionScore}
              </div>

              {/* Status */}
              <div className="col-span-2 flex items-center gap-2">
                <span
                  className={`text-[12px] font-semibold ${
                    interaction.resolved ? "text-[#10B981]" : "text-[#EF4444]"
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
                  className="text-[12px] text-[#3B82F6] font-medium hover:underline"
                >
                  Inspect →
                </Link>
              </div>
            </div>
          )))}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[#E5E7EB] flex items-center justify-between">
          <span className="text-[12px] text-[#6B7280]">
            Showing {totalItems === 0 ? 0 : startIndex + 1}–{Math.min(startIndex + itemsPerPage, totalItems)} of {totalItems}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 h-8 text-[12px] text-[#6B7280] hover:text-[#111827] disabled:opacity-40"
            >
              ← Prev
            </button>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || totalItems === 0}
              className="px-3 h-8 text-[12px] text-[#6B7280] hover:text-[#111827] disabled:opacity-40"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
