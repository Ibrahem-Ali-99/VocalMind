import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router";
import { Search, ChevronDown, Loader2, AlertTriangle } from "lucide-react";
import {
  createInteraction,
  getAgents,
  getInteractions,
  reprocessInteraction,
  type AgentSummary,
  type InteractionSummary,
} from "../../services/api";

export function SessionInspector() {
  const navigate = useNavigate();
  const [interactions, setInteractions] = useState<InteractionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reprocessingIds, setReprocessingIds] = useState<Set<string>>(new Set());
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedAudioFile, setSelectedAudioFile] = useState<File | null>(null);
  const [selectedUploadAgentId, setSelectedUploadAgentId] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("All Agents");
  const [sortField, setSortField] = useState<"score" | "date" | "duration">("score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const loadInteractions = () => {
    setLoading(true);
    setLoadError(null);
    getInteractions()
      .then(setInteractions)
      .catch((err) => setLoadError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadInteractions();
    getAgents().then(setAgents).catch(() => setAgents([]));
  }, []);

  const handleUploadInteraction = async () => {
    if (!selectedAudioFile) {
      setActionError("Please select an audio file (.wav or .mp3)");
      return;
    }

    setUploading(true);
    setActionError(null);
    try {
      const result = await createInteraction(selectedAudioFile, selectedUploadAgentId || undefined);
      setShowUploadModal(false);
      setSelectedAudioFile(null);
      setSelectedUploadAgentId("");
      await loadInteractions();
      navigate(`/manager/inspector/${result.interactionId}`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to upload interaction");
    } finally {
      setUploading(false);
    }
  };

  const handleReprocess = async (interactionId: string) => {
    setActionError(null);
    setReprocessingIds((prev) => new Set(prev).add(interactionId));
    try {
      await reprocessInteraction(interactionId);
      await loadInteractions();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reprocess interaction";
      if (message.includes("409")) {
        try {
          await reprocessInteraction(interactionId, { force: true });
          await loadInteractions();
          return;
        } catch {
          setActionError("This interaction is already being processed. Wait a few seconds and try again.");
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

  if (loadError) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-destructive mx-auto mb-3" />
          <p className="text-foreground text-sm">Failed to load interactions</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{loadError}</p>
        </div>
      </div>
    );
  }

  const uniqueAgents = Array.from(new Set(interactions.map(i => i.agentName))).sort();

  const filteredInteractions = interactions.filter((interaction) => {
    const searchLower = searchQuery.toLowerCase();
    const matchesSearch = interaction.agentName.toLowerCase().includes(searchLower) ||
                          interaction.id.toLowerCase().includes(searchLower) ||
                          interaction.date.toLowerCase().includes(searchLower);
    const matchesAgent = selectedAgent === "All Agents" || interaction.agentName === selectedAgent;
    return matchesSearch && matchesAgent;
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
    <div className="p-4 md:p-8 max-w-[1600px] mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-[26px] font-extrabold text-foreground tracking-tight">Session Inspector</h2>
            <p className="text-[13px] text-muted-foreground mt-1">
              {totalItems} interaction{totalItems !== 1 ? "s" : ""} found
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowUploadModal(true)}
            className="h-11 px-6 rounded-2xl bg-primary text-primary-foreground text-[13px] font-bold hover:bg-primary/90 transition-all shadow-[0_4px_12px_rgba(59,130,246,0.3)] hover:shadow-[0_6px_20px_rgba(59,130,246,0.4)]"
          >
            + Upload Audio
          </button>
        </div>

        {/* Filters Row */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 min-w-[180px] max-w-[280px]">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search agent, date, ID…"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full h-10 pl-10 pr-3 bg-card border border-border rounded-xl text-[13px] focus:outline-none focus:ring-2 focus:ring-primary/30 transition-all"
            />
          </div>

          <div className="relative">
            <select
              value={selectedAgent}
              onChange={(e) => {
                setSelectedAgent(e.target.value);
                setCurrentPage(1);
              }}
              className="appearance-none h-10 pl-4 pr-10 bg-card border border-border rounded-xl text-[13px] hover:bg-muted/30 transition-colors focus:outline-none focus:ring-2 focus:ring-primary/30 cursor-pointer"
            >
              <option value="All Agents">All Agents</option>
              {uniqueAgents.map(agent => (
                <option key={agent} value={agent}>{agent}</option>
              ))}
            </select>
            <ChevronDown className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-muted-foreground" />
          </div>

          <div className="flex items-center border border-border rounded-xl overflow-hidden bg-card shadow-sm">
            {(["score", "date", "duration"] as const).map((field) => (
              <button
                key={field}
                onClick={() => handleSort(field)}
                className={`px-4 h-10 text-[11px] font-bold uppercase tracking-wider transition-all ${sortField === field ? "bg-primary text-primary-foreground shadow-inner" : "text-muted-foreground hover:bg-muted/40"}`}
              >
                {field} {sortField === field ? (sortOrder === "desc" ? "↓" : "↑") : ""}
              </button>
            ))}
          </div>
        </div>
      </div>

      {actionError && (
        <div className="mb-4 rounded-[10px] border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12px] font-medium text-destructive">
          {actionError}
        </div>
      )}

      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-[1px] flex items-center justify-center p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-border bg-card p-6 shadow-2xl">
            <h3 className="text-[18px] font-bold text-foreground mb-2">Upload Real Audio Interaction</h3>
            <p className="text-[13px] text-muted-foreground mb-6">Select an agent and upload one call at a time to run the full pipeline.</p>

            <div className="space-y-6">
              {/* Agent Selection */}
              <div>
                <label className="block text-[12px] font-semibold text-foreground mb-3">Select Agent</label>
                {agents.length === 0 ? (
                  <div className="p-4 bg-muted/20 rounded-[10px] text-[13px] text-muted-foreground text-center">
                    No agents available. Create agents first in the organization settings.
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => setSelectedUploadAgentId("")}
                      className={`p-3 rounded-[12px] border-2 transition-all text-left ${
                        selectedUploadAgentId === ""
                          ? "border-primary bg-primary/5"
                          : "border-border bg-muted/20 hover:border-primary/50"
                      }`}
                    >
                      <div className="text-[13px] font-semibold text-foreground">Auto-Select</div>
                      <div className="text-[11px] text-muted-foreground mt-1">Use first active agent</div>
                    </button>
                    {agents.map((agent) => (
                      <button
                        key={agent.id}
                        type="button"
                        onClick={() => setSelectedUploadAgentId(agent.id)}
                        className={`p-3 rounded-[12px] border-2 transition-all text-left ${
                          selectedUploadAgentId === agent.id
                            ? "border-primary bg-primary/5"
                            : "border-border bg-muted/20 hover:border-primary/50"
                        }`}
                      >
                        <div className="text-[13px] font-semibold text-foreground">{agent.name}</div>
                        <div className="text-[11px] text-muted-foreground mt-1">{agent.role || "Agent"}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Audio File */}
              <div>
                <label className="block text-[12px] font-semibold text-foreground mb-2">Audio File (.wav or .mp3)</label>
                <input
                  type="file"
                  accept=".wav,.mp3,audio/wav,audio/mpeg"
                  onChange={(e) => setSelectedAudioFile(e.target.files?.[0] ?? null)}
                  className="w-full text-[13px] file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-[13px] file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90"
                />
                {selectedAudioFile && (
                  <p className="text-[11px] text-muted-foreground mt-2">
                    ✓ Selected: {selectedAudioFile.name}
                  </p>
                )}
              </div>
            </div>

            <div className="mt-6 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  if (!uploading) {
                    setShowUploadModal(false);
                    setSelectedAudioFile(null);
                    setSelectedUploadAgentId("");
                  }
                }}
                className="h-10 px-4 rounded-[10px] border border-border text-[13px] font-semibold hover:bg-muted transition-colors"
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleUploadInteraction()}
                className="h-10 px-4 rounded-[10px] bg-primary text-primary-foreground text-[13px] font-semibold disabled:opacity-60 hover:bg-primary/90 transition-colors"
                disabled={uploading || !selectedAudioFile}
              >
                {uploading ? "Uploading..." : "Upload & Process"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Table Card */}
      <div className="bg-card rounded-2xl border border-border overflow-hidden shadow-sm">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-muted/10 border-b border-border">
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Agent</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Date & Time</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Duration</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Score</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Empathy</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Policy</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Resolution</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Status</th>
              <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-widest text-muted-foreground">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {paginatedInteractions.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-6 py-16 text-center text-muted-foreground">
                  <div className="flex flex-col items-center justify-center">
                    <Search className="w-10 h-10 opacity-20 mb-3" />
                    <p className="text-[14px] font-medium text-foreground">No interactions found</p>
                    <p className="text-[12px] opacity-70 mt-1">Try adjusting your filters or search query.</p>
                  </div>
                </td>
              </tr>
            ) : paginatedInteractions.map((row) => {
              const scoreColor = row.overallScore >= 85 ? "var(--success)" : row.overallScore >= 75 ? "var(--primary)" : "var(--destructive)";
              const rowStatus = (row.status || "").toLowerCase();
              const isRowProcessing = ["pending", "processing"].includes(rowStatus);
              const statusBadge = rowStatus === "failed"
                ? { className: "bg-destructive/5 text-destructive border-destructive/20", label: "⚠ Failed" }
                : isRowProcessing
                  ? { className: "bg-amber-50 text-amber-700 border-amber-200", label: "⟳ Processing" }
                  : row.resolved
                    ? { className: "bg-success/5 text-success border-success/20", label: "✓ Resolved" }
                    : { className: "bg-destructive/5 text-destructive border-destructive/20", label: "✗ Unresolved" };
              return (
              <tr key={row.id} className="group hover:bg-primary/[0.03] transition-colors">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-[11px] font-extrabold text-primary shrink-0">
                      {row.agentName.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <span className="text-[13px] font-bold text-foreground block">{row.agentName}</span>
                      {row.hasViolation && (
                        <span className="px-1.5 py-0.5 bg-destructive/10 text-destructive rounded text-[9px] font-bold">
                          ⚠ Violation
                        </span>
                      )}
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-[12px] text-muted-foreground font-medium">
                  <div>{row.date}</div>
                  <div className="text-[11px] opacity-70">{row.time}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] font-bold text-foreground/70">
                  {row.duration}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div
                    className="text-[20px] font-extrabold tabular-nums"
                    style={{ color: scoreColor }}
                  >
                    {row.overallScore}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] font-bold text-foreground">{row.empathyScore}</td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] font-bold text-foreground">{row.policyScore}</td>
                <td className="px-6 py-4 whitespace-nowrap text-[13px] font-bold text-foreground">{row.resolutionScore}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold border ${statusBadge.className}`}>
                    {statusBadge.label}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <div className="flex items-center justify-end gap-3">
                    <button
                      type="button"
                      disabled={reprocessingIds.has(row.id) || isRowProcessing}
                      onClick={() => void handleReprocess(row.id)}
                      className="text-[12px] font-bold text-foreground/60 hover:text-foreground disabled:opacity-40 transition-colors"
                    >
                      {isRowProcessing ? "Processing..." : reprocessingIds.has(row.id) ? "Reprocessing..." : "Reprocess"}
                    </button>
                    <Link
                      to={`/manager/inspector/${row.id}`}
                      className="px-3 py-1.5 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 font-bold text-[12px] transition-all"
                    >
                      Inspect →
                    </Link>
                  </div>
                </td>
              </tr>
            );
            })}
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
              className="h-9 px-4 rounded-xl border border-border bg-background text-[13px] font-bold text-foreground hover:bg-muted disabled:opacity-40 transition-all"
            >
              ← Prev
            </button>
            <span className="px-3 text-[12px] font-bold text-muted-foreground tabular-nums">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || totalItems === 0}
              className="h-9 px-4 rounded-xl border border-border bg-background text-[13px] font-bold text-foreground hover:bg-muted disabled:opacity-40 transition-all"
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
