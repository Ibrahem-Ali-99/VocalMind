import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import {
  AlertTriangle,
  ChevronDown,
  Clock3,
  Loader2,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react";
import {
  createInteraction,
  getAgents,
  getInteractions,
  reprocessInteraction,
  type AgentSummary,
  type InteractionSummary,
} from "../../services/api";

type SortField = "score" | "date" | "duration";

function parseDuration(duration: string): number {
  const [minutes, seconds] = duration.split(":").map(Number);
  return ((minutes || 0) * 60) + (seconds || 0);
}

function getStatusMeta(status: string, resolved: boolean) {
  const normalized = (status || "").toLowerCase();
  if (normalized === "failed") return { label: "Failed", tone: "border-red-500/20 bg-red-500/10 text-red-200" };
  if (normalized === "pending" || normalized === "processing") {
    return { label: "Processing", tone: "border-amber-400/20 bg-amber-400/10 text-amber-100" };
  }
  if (resolved) return { label: "Resolved", tone: "border-emerald-400/20 bg-emerald-400/10 text-emerald-100" };
  return { label: "Needs review", tone: "border-red-500/20 bg-red-500/10 text-red-200" };
}

function getScoreMeta(score: number) {
  if (score >= 85) return { text: "text-emerald-300", bar: "bg-emerald-400", card: "from-emerald-500/20 to-transparent" };
  if (score >= 75) return { text: "text-blue-300", bar: "bg-blue-400", card: "from-blue-500/20 to-transparent" };
  return { text: "text-red-300", bar: "bg-red-400", card: "from-red-500/20 to-transparent" };
}

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
  const [selectedUploadAgentId, setSelectedUploadAgentId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("All Agents");
  const [sortField, setSortField] = useState<SortField>("score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const loadInteractions = async () => {
    setLoading(true);
    setLoadError(null);
    try {
      setInteractions(await getInteractions());
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load interactions");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadInteractions();
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

  const uniqueAgents = useMemo(
    () => Array.from(new Set(interactions.map((interaction) => interaction.agentName))).sort(),
    [interactions],
  );

  const filteredInteractions = useMemo(() => {
    return interactions.filter((interaction) => {
      const searchLower = searchQuery.toLowerCase();
      const matchesSearch =
        interaction.agentName.toLowerCase().includes(searchLower) ||
        interaction.id.toLowerCase().includes(searchLower) ||
        interaction.date.toLowerCase().includes(searchLower);
      const matchesAgent = selectedAgent === "All Agents" || interaction.agentName === selectedAgent;
      return matchesSearch && matchesAgent;
    });
  }, [interactions, searchQuery, selectedAgent]);

  const sortedInteractions = useMemo(() => {
    return [...filteredInteractions].sort((a, b) => {
      let comparison = 0;
      if (sortField === "score") comparison = a.overallScore - b.overallScore;
      if (sortField === "date") comparison = new Date(`${a.date}T${a.time}`).getTime() - new Date(`${b.date}T${b.time}`).getTime();
      if (sortField === "duration") comparison = parseDuration(a.duration) - parseDuration(b.duration);
      return sortOrder === "asc" ? comparison : -comparison;
    });
  }, [filteredInteractions, sortField, sortOrder]);

  const totalItems = sortedInteractions.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedInteractions = sortedInteractions.slice(startIndex, startIndex + itemsPerPage);

  const summary = useMemo(() => {
    const processing = interactions.filter((item) => ["pending", "processing"].includes((item.status || "").toLowerCase())).length;
    const flagged = interactions.filter((item) => item.hasViolation || !item.resolved || (item.status || "").toLowerCase() === "failed").length;
    const avgScore = interactions.length
      ? Math.round(interactions.reduce((total, item) => total + item.overallScore, 0) / interactions.length)
      : 0;
    const completed = interactions.filter((item) => (item.status || "").toLowerCase() === "completed").length;
    return {
      total: interactions.length,
      processing,
      flagged,
      avgScore,
      completionRate: interactions.length ? Math.round((completed / interactions.length) * 100) : 0,
    };
  }, [interactions]);

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-3 text-sm text-muted-foreground">Loading interactions...</span>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="mx-auto mb-3 h-10 w-10 text-destructive" />
          <p className="text-sm text-foreground">Failed to load interactions</p>
          <p className="mt-1 text-xs text-muted-foreground/80">{loadError}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1600px] p-4 md:p-8">
      <div className="mb-8 space-y-6">
        <div className="overflow-hidden rounded-[28px] border border-slate-800 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.24),transparent_32%),linear-gradient(135deg,#0f172a_0%,#111827_55%,#0b1220_100%)] px-6 py-6 shadow-[0_24px_60px_rgba(2,6,23,0.34)] md:px-8">
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl">
                <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-400/10 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.22em] text-blue-100">
                  <Sparkles className="h-3.5 w-3.5" />
                  Manager overview
                </div>
                <h2 className="text-[28px] font-extrabold tracking-tight text-white md:text-[34px]">Session Inspector</h2>
                <p className="mt-2 max-w-2xl text-[14px] leading-6 text-slate-300">
                  Review live processing, surface risky calls, and jump into explainable evidence faster.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                {summary.processing > 0 && (
                  <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-[12px] font-semibold text-amber-100">
                    <Clock3 className="h-4 w-4" />
                    {summary.processing} active pipeline run{summary.processing !== 1 ? "s" : ""}
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => setShowUploadModal(true)}
                  className="inline-flex h-11 items-center gap-2 rounded-2xl bg-primary px-5 text-[13px] font-bold text-primary-foreground shadow-[0_8px_24px_rgba(59,130,246,0.34)] transition-all hover:-translate-y-0.5 hover:bg-primary/90"
                >
                  <Upload className="h-4 w-4" />
                  Upload Audio
                </button>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {[
                { label: "Visible sessions", value: `${summary.total}`, helper: `${summary.completionRate}% completed` },
                { label: "Average score", value: `${summary.avgScore}%`, helper: "Across loaded calls" },
                { label: "Flagged sessions", value: `${summary.flagged}`, helper: "Violation, unresolved, or failed" },
                { label: "Filtered results", value: `${totalItems}`, helper: searchQuery || selectedAgent !== "All Agents" ? "Filters applied" : "No filters active" },
              ].map((item) => (
                <div key={item.label} className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                  <div className="text-[11px] font-bold uppercase tracking-[0.22em] text-slate-400">{item.label}</div>
                  <div className="mt-3 text-[28px] font-extrabold text-white">{item.value}</div>
                  <div className="mt-1 text-[12px] text-slate-400">{item.helper}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-[24px] border border-slate-800/80 bg-slate-950/60 p-4 shadow-[0_16px_36px_rgba(2,6,23,0.2)]">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex flex-1 flex-col gap-3 md:flex-row md:flex-wrap md:items-center">
              <div className="relative min-w-[220px] flex-1 md:max-w-[340px]">
                <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search agent, date, ID..."
                  value={searchQuery}
                  onChange={(event) => {
                    setSearchQuery(event.target.value);
                    setCurrentPage(1);
                  }}
                  className="h-11 w-full rounded-xl border border-slate-800 bg-slate-900/80 pl-10 pr-3 text-[13px] text-slate-100 outline-none transition-all placeholder:text-slate-500 focus:border-primary/50 focus:ring-2 focus:ring-primary/20"
                />
              </div>

              <div className="relative">
                <select
                  value={selectedAgent}
                  onChange={(event) => {
                    setSelectedAgent(event.target.value);
                    setCurrentPage(1);
                  }}
                  className="h-11 appearance-none rounded-xl border border-slate-800 bg-slate-900/80 pl-4 pr-10 text-[13px] text-slate-100 outline-none transition-colors hover:border-slate-700 focus:border-primary/50 focus:ring-2 focus:ring-primary/20"
                >
                  <option value="All Agents">All Agents</option>
                  {uniqueAgents.map((agent) => (
                    <option key={agent} value={agent}>
                      {agent}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              </div>

              <div className="flex items-center overflow-hidden rounded-xl border border-slate-800 bg-slate-900/80">
                {(["score", "date", "duration"] as const).map((field) => (
                  <button
                    key={field}
                    type="button"
                    onClick={() => {
                      if (sortField === field) {
                        setSortOrder(sortOrder === "asc" ? "desc" : "asc");
                      } else {
                        setSortField(field);
                        setSortOrder("desc");
                      }
                      setCurrentPage(1);
                    }}
                    className={`h-11 px-4 text-[11px] font-bold uppercase tracking-[0.2em] transition-all ${
                      sortField === field ? "bg-primary text-primary-foreground" : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"
                    }`}
                  >
                    {field}
                    {sortField === field ? ` ${sortOrder === "desc" ? "v" : "^"}` : ""}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => void loadInteractions()}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-4 text-[12px] font-bold text-slate-200 transition-colors hover:border-slate-700 hover:bg-slate-800"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {actionError && <div className="mb-4 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12px] font-medium text-destructive">{actionError}</div>}

      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-[2px]">
          <div className="w-full max-w-2xl rounded-[26px] border border-slate-800 bg-slate-950 p-6 shadow-[0_30px_80px_rgba(0,0,0,0.45)]">
            <h3 className="text-[18px] font-bold text-slate-50">Upload Real Audio Interaction</h3>
            <p className="mb-6 mt-2 text-[13px] text-slate-400">Select an agent and upload one call at a time to run the full evaluation pipeline.</p>
            <div className="space-y-6">
              <div>
                <label className="mb-3 block text-[12px] font-semibold text-slate-200">Select Agent</label>
                {agents.length === 0 ? (
                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-center text-[13px] text-slate-400">No agents available. Create agents first in the organization settings.</div>
                ) : (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <button type="button" onClick={() => setSelectedUploadAgentId("")} className={`rounded-2xl border p-4 text-left transition-all ${selectedUploadAgentId === "" ? "border-primary bg-primary/10" : "border-slate-800 bg-slate-900/60 hover:border-primary/40"}`}>
                      <div className="text-[13px] font-semibold text-slate-100">Auto-Select</div>
                      <div className="mt-1 text-[11px] text-slate-400">Use the first active agent</div>
                    </button>
                    {agents.map((agent) => (
                      <button key={agent.id} type="button" onClick={() => setSelectedUploadAgentId(agent.id)} className={`rounded-2xl border p-4 text-left transition-all ${selectedUploadAgentId === agent.id ? "border-primary bg-primary/10" : "border-slate-800 bg-slate-900/60 hover:border-primary/40"}`}>
                        <div className="text-[13px] font-semibold text-slate-100">{agent.name}</div>
                        <div className="mt-1 text-[11px] text-slate-400">{agent.role || "Agent"}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div>
                <label className="mb-2 block text-[12px] font-semibold text-slate-200">Audio File (.wav or .mp3)</label>
                <input type="file" accept=".wav,.mp3,audio/wav,audio/mpeg" onChange={(event) => setSelectedAudioFile(event.target.files?.[0] ?? null)} className="w-full text-[13px] text-slate-200 file:mr-4 file:rounded-xl file:border-0 file:bg-primary file:px-4 file:py-2.5 file:text-[13px] file:font-semibold file:text-primary-foreground hover:file:bg-primary/90" />
                {selectedAudioFile && <p className="mt-2 text-[11px] text-slate-400">Selected: {selectedAudioFile.name}</p>}
              </div>
            </div>
            <div className="mt-6 flex items-center justify-end gap-2">
              <button type="button" onClick={() => { if (!uploading) { setShowUploadModal(false); setSelectedAudioFile(null); setSelectedUploadAgentId(""); } }} className="h-10 rounded-xl border border-slate-800 px-4 text-[13px] font-semibold text-slate-200 transition-colors hover:bg-slate-900" disabled={uploading}>Cancel</button>
              <button type="button" onClick={() => void handleUploadInteraction()} className="h-10 rounded-xl bg-primary px-4 text-[13px] font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-60" disabled={uploading || !selectedAudioFile}>{uploading ? "Uploading..." : "Upload and Process"}</button>
            </div>
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded-[28px] border border-slate-800 bg-[linear-gradient(180deg,rgba(15,23,42,0.98),rgba(10,15,27,0.98))] shadow-[0_24px_60px_rgba(2,6,23,0.25)]">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-white/[0.03]">
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Agent</th>
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Date and time</th>
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Duration</th>
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Score</th>
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Signals</th>
                <th className="px-6 py-4 text-left text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Status</th>
                <th className="px-6 py-4 text-right text-[10px] font-extrabold uppercase tracking-[0.22em] text-slate-500">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {paginatedInteractions.length === 0 ? null : paginatedInteractions.map((row) => {
                const status = getStatusMeta(row.status, row.resolved);
                const score = getScoreMeta(row.overallScore);
                const isRowProcessing = ["pending", "processing"].includes((row.status || "").toLowerCase());
                const isReprocessing = reprocessingIds.has(row.id);
                return (
                  <tr key={row.id} className="group transition-colors hover:bg-white/[0.02]">
                    <td className="px-6 py-5 align-top">
                      <div className="flex items-start gap-3">
                        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-blue-400/20 bg-blue-400/10 text-[13px] font-extrabold text-blue-200">{row.agentName.charAt(0).toUpperCase()}</div>
                        <div className="space-y-1">
                          <div className="text-[14px] font-bold text-slate-100">{row.agentName}</div>
                          <div className="text-[12px] text-slate-500">{row.id}</div>
                          <div className="flex flex-wrap gap-2 pt-1">
                            {row.hasViolation && <span className="inline-flex items-center gap-1 rounded-full border border-red-500/20 bg-red-500/10 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-red-200"><ShieldAlert className="h-3 w-3" />Violation</span>}
                            {row.language && <span className="rounded-full border border-slate-700 bg-slate-900/80 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">{row.language}</span>}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-5 align-top">
                      <div className="space-y-1">
                        <div className="text-[13px] font-semibold text-slate-200">{row.date}</div>
                        <div className="text-[12px] text-slate-500">{row.time}</div>
                      </div>
                    </td>
                    <td className="px-6 py-5 align-top"><div className="rounded-2xl border border-slate-800 bg-slate-900/60 px-3 py-3"><div className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-500">Call length</div><div className="mt-1 text-[18px] font-extrabold text-slate-100">{row.duration}</div></div></td>
                    <td className="px-6 py-5 align-top">
                      <div className={`rounded-2xl border border-white/10 bg-gradient-to-br ${score.card} px-4 py-3`}>
                        <div className={`text-[24px] font-extrabold tabular-nums ${score.text}`}>{row.overallScore}%</div>
                        <div className="text-[11px] font-medium text-slate-400">Overall evaluation</div>
                        <div className="mt-3 h-1.5 rounded-full bg-slate-900/80"><div className={`h-1.5 rounded-full ${score.bar}`} style={{ width: `${Math.max(6, Math.min(100, row.overallScore))}%` }} /></div>
                      </div>
                    </td>
                    <td className="px-6 py-5 align-top">
                      <div className="grid min-w-[230px] grid-cols-3 gap-2">
                        {[{ label: "Empathy", value: row.empathyScore }, { label: "Policy", value: row.policyScore }, { label: "Resolution", value: row.resolutionScore }].map((metric) => (
                          <div key={metric.label} className="rounded-2xl border border-slate-800 bg-slate-900/60 px-3 py-3">
                            <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">{metric.label}</div>
                            <div className="mt-1 text-[18px] font-extrabold text-slate-100">{metric.value}</div>
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-5 align-top"><span className={`inline-flex items-center rounded-full border px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.16em] ${status.tone}`}>{status.label}</span></td>
                    <td className="px-6 py-5 text-right align-top">
                      <div className="flex flex-col items-end gap-2">
                        <button type="button" disabled={isReprocessing || isRowProcessing} onClick={() => void handleReprocess(row.id)} className="inline-flex h-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/80 px-4 text-[12px] font-bold text-slate-200 transition-colors hover:border-slate-700 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40">{isRowProcessing ? "Processing..." : isReprocessing ? "Reprocessing..." : "Reprocess"}</button>
                        <Link to={`/manager/inspector/${row.id}`} className="inline-flex h-9 items-center justify-center rounded-xl bg-primary px-4 text-[12px] font-bold text-primary-foreground transition-all hover:-translate-y-0.5 hover:bg-primary/90">Inspect</Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {paginatedInteractions.length === 0 && (
          <div className="px-6 py-20">
            <div className="mx-auto flex max-w-md flex-col items-center text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/80"><Search className="h-6 w-6 text-slate-500" /></div>
              <p className="mt-5 text-[16px] font-bold text-slate-100">No interactions found</p>
              <p className="mt-2 text-[13px] leading-6 text-slate-400">Try adjusting your filters, clearing the search, or upload a fresh call to populate the queue.</p>
              <button type="button" onClick={() => setShowUploadModal(true)} className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-[13px] font-bold text-primary-foreground transition-colors hover:bg-primary/90"><Upload className="h-4 w-4" />Upload Audio</button>
            </div>
          </div>
        )}

        <div className="flex flex-col gap-3 border-t border-slate-800 bg-white/[0.03] px-6 py-4 md:flex-row md:items-center md:justify-between">
          <div className="text-[13px] font-medium text-slate-400">Showing {totalItems === 0 ? 0 : startIndex + 1} to {Math.min(startIndex + itemsPerPage, totalItems)} of {totalItems}</div>
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setCurrentPage((page) => Math.max(1, page - 1))} disabled={currentPage === 1} className="h-9 rounded-xl border border-slate-800 bg-slate-900/80 px-4 text-[13px] font-bold text-slate-200 transition-colors hover:border-slate-700 hover:bg-slate-800 disabled:opacity-40">Prev</button>
            <span className="px-3 text-[12px] font-bold tabular-nums text-slate-400">{currentPage} / {totalPages}</span>
            <button type="button" onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))} disabled={currentPage === totalPages || totalItems === 0} className="h-9 rounded-xl border border-slate-800 bg-slate-900/80 px-4 text-[13px] font-bold text-slate-200 transition-colors hover:border-slate-700 hover:bg-slate-800 disabled:opacity-40">Next</button>
          </div>
        </div>
      </div>
    </div>
  );
}
