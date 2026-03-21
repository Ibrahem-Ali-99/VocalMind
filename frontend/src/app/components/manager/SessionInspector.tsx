import { useState, useEffect } from "react";
import { Link } from "react-router";
import { Search, ChevronDown, Loader2, AlertTriangle, Calendar, Clock, PhoneCall, TrendingUp } from "lucide-react";
import { getInteractions, type InteractionSummary } from "../../services/api";

export function SessionInspector() {
  const [interactions, setInteractions] = useState<InteractionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "completed" | "pending">("all");
  const [resolutionFilter, setResolutionFilter] = useState<"all" | "resolved" | "unresolved">("all");
  const [violationFilter, setViolationFilter] = useState<"all" | "violation" | "clean">("all");

  useEffect(() => {
    getInteractions()
      .then(setInteractions)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#F8FAFC]">
        <Loader2 className="w-10 h-10 text-[#3B82F6] animate-spin" />
        <span className="ml-3 text-[#64748B] font-medium border-l border-[#E2E8F0] pl-3 py-1">Loading interactions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#F8FAFC]">
        <div className="text-center bg-white p-8 rounded-2xl shadow-sm border border-[#E2E8F0]">
          <AlertTriangle className="w-12 h-12 text-[#EF4444] mx-auto mb-4" />
          <p className="text-[#0F172A] font-bold text-lg">Failed to Load Sessions</p>
          <p className="text-[#64748B] text-sm mt-2 max-w-sm">{error}</p>
        </div>
      </div>
    );
  }

  const filteredInteractions = interactions.filter(i => {
    const matchSearch = i.agentName.toLowerCase().includes(searchQuery.toLowerCase()) || i.id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchStatus = statusFilter === "all" || i.status === statusFilter;
    const matchRes = resolutionFilter === "all" || (resolutionFilter === "resolved" ? i.resolved : !i.resolved);
    const matchViol = violationFilter === "all" || (violationFilter === "violation" ? i.hasViolation : !i.hasViolation);
    return matchSearch && matchStatus && matchRes && matchViol;
  });

  const sortedInteractions = [...filteredInteractions].sort((a, b) => b.overallScore - a.overallScore);

  return (
    <div className="p-4 md:p-8 bg-[#F8FAFC] min-h-screen space-y-8">
      
      {/* Dynamic Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 max-w-[1400px] mx-auto px-2">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-[#0F172A] mb-2 flex items-center gap-3">
            <PhoneCall className="w-8 h-8 text-[#3B82F6]" /> Session Inspector
          </h2>
          <p className="text-sm text-[#64748B]">
            Analyze {interactions.length} automated LLM agent evaluations
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#94A3B8] group-focus-within:text-[#3B82F6] transition-colors" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search agent, date, ID..."
              className="w-[280px] h-11 pl-10 pr-4 bg-white border border-[#E2E8F0] rounded-xl text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-[#3B82F6]/50 focus:border-[#3B82F6] transition-all"
            />
          </div>
          
          <select 
            value={resolutionFilter}
            onChange={(e) => setResolutionFilter(e.target.value as any)}
            className="h-11 px-3 bg-white border border-[#E2E8F0] rounded-xl shadow-sm text-sm font-semibold text-[#334155] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]/50 focus:border-[#3B82F6] transition-colors cursor-pointer appearance-none"
            style={{ backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2394A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>')`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', paddingRight: '36px' }}
          >
            <option value="all">Any Resolution</option>
            <option value="resolved">Resolved</option>
            <option value="unresolved">Unresolved</option>
          </select>

          <select 
            value={violationFilter}
            onChange={(e) => setViolationFilter(e.target.value as any)}
            className="h-11 px-3 bg-white border border-[#E2E8F0] rounded-xl shadow-sm text-sm font-semibold text-[#334155] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]/50 focus:border-[#3B82F6] transition-colors cursor-pointer appearance-none"
            style={{ backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%2394A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>')`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', paddingRight: '36px' }}
          >
            <option value="all">Any Policies</option>
            <option value="violation">Has Violations</option>
            <option value="clean">Clean Records</option>
          </select>
        </div>
      </div>

      <div className="max-w-[1400px] mx-auto grid grid-cols-1 gap-6">
        {/* Interaction Table Container */}
        <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm overflow-hidden">
          
          {/* Table Header */}
          <div className="grid grid-cols-12 gap-4 px-6 py-4 border-b border-[#E2E8F0] bg-[#F8FAFC]/50">
            <div className="col-span-3 text-xs font-bold uppercase tracking-wider text-[#64748B]">Agent Profile</div>
            <div className="col-span-2 text-xs font-bold uppercase tracking-wider text-[#64748B]">Session Info</div>
            <div className="col-span-1 text-xs font-bold uppercase tracking-wider text-[#64748B] text-center w-full">Score</div>
            <div className="col-span-3 text-xs font-bold uppercase tracking-wider text-[#64748B]">Automated Metrics (Emp / Pol / Res)</div>
            <div className="col-span-2 text-xs font-bold uppercase tracking-wider text-[#64748B]">Status Summary</div>
            <div className="col-span-1 text-xs font-bold uppercase tracking-wider text-[#64748B] text-right">Actions</div>
          </div>

          {/* Table Rows */}
          <div className="divide-y divide-[#E2E8F0]">
            {sortedInteractions.map((interaction) => (
              <div
                key={interaction.id}
                className="grid grid-cols-12 gap-4 px-6 py-5 hover:bg-[#F8FAFC] transition-colors items-center group"
              >
                {/* Agent Column */}
                <div className="col-span-3 flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#1D4ED8] flex items-center justify-center text-white text-sm font-bold shadow-sm flex-shrink-0 group-hover:scale-105 transition-transform">
                    {interaction.agentName.split(" ").map((n) => n[0]).join("")}
                  </div>
                  <div>
                    <span className="block text-sm font-bold text-[#0F172A]">
                      {interaction.agentName}
                    </span>
                    <span className="text-xs text-[#64748B] font-mono">
                      {interaction.id.split("-")[1] || interaction.id}
                    </span>
                  </div>
                </div>

                {/* Session Info */}
                <div className="col-span-2 flex flex-col justify-center text-sm">
                  <span className="text-[#334155] font-medium flex items-center gap-1.5"><Calendar className="w-3.5 h-3.5 text-[#94A3B8]"/> {interaction.date}</span>
                  <span className="text-[#64748B] text-xs flex items-center gap-1.5 mt-0.5"><Clock className="w-3.5 h-3.5 text-[#94A3B8]"/> {interaction.time} • {interaction.duration}</span>
                </div>

                {/* Overall Score */}
                <div className="col-span-1 flex items-center justify-center">
                  <span
                    className={`px-3 py-1.5 rounded-lg text-sm font-bold w-16 text-center shadow-sm ${
                      interaction.overallScore >= 85
                        ? "bg-[#D1FAE5] text-[#047857]"
                        : interaction.overallScore >= 75
                        ? "bg-[#DBEAFE] text-[#1D4ED8]"
                        : "bg-[#FEF3C7] text-[#B45309]"
                    }`}
                  >
                    {interaction.overallScore}%
                  </span>
                </div>

                {/* Metrics */}
                <div className="col-span-3 flex items-center gap-4 text-xs font-semibold">
                  <div className="flex flex-col items-center">
                    <span className={`text-sm ${interaction.empathyScore > 80 ? "text-[#3B82F6]" : "text-[#64748B]"}`}>{interaction.empathyScore}</span>
                  </div>
                  <div className="w-px h-6 bg-[#E2E8F0]" />
                  <div className="flex flex-col items-center">
                    <span className={`text-sm ${interaction.policyScore > 80 ? "text-[#10B981]" : "text-[#64748B]"}`}>{interaction.policyScore}</span>
                  </div>
                  <div className="w-px h-6 bg-[#E2E8F0]" />
                  <div className="flex flex-col items-center">
                    <span className={`text-sm ${interaction.resolutionScore > 80 ? "text-[#8B5CF6]" : "text-[#64748B]"}`}>{interaction.resolutionScore}</span>
                  </div>
                </div>

                {/* Status Summary */}
                <div className="col-span-2 flex flex-col items-start justify-center gap-1.5">
                  <span
                    className={`text-xs font-bold px-2.5 py-1 rounded-md ${
                      interaction.resolved ? "bg-[#ECFDF5] text-[#059669] border border-[#A7F3D0]" : "bg-[#FEF2F2] text-[#DC2626] border border-[#FECACA]"
                    }`}
                  >
                    {interaction.resolved ? "✓ Resolved" : "✗ Unresolved"}
                  </span>
                  {interaction.hasViolation && (
                    <span className="px-2 py-0.5 bg-[#FFFBEB] text-[#D97706] border border-[#FDE68A] rounded-md text-[10px] font-bold uppercase tracking-wider flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" /> Violation
                    </span>
                  )}
                </div>

                {/* Actions */}
                <div className="col-span-1 flex justify-end">
                  <Link
                    to={`/manager/inspector/${interaction.id}`}
                    className="flex items-center justify-center bg-[#F8FAFC] border border-[#E2E8F0]  text-[#3B82F6] hover:bg-[#3B82F6] hover:border-[#3B82F6] hover:text-white px-4 py-2 rounded-lg text-sm font-semibold transition-all shadow-sm"
                  >
                    Inspect
                  </Link>
                </div>
              </div>
            ))}
          </div>

          {/* Table Footer */}
          <div className="bg-[#F8FAFC] px-6 py-4 border-t border-[#E2E8F0] flex items-center justify-between">
            <div className="text-sm font-medium text-[#64748B] flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-[#94A3B8]" />
              Showing 1–{sortedInteractions.length} of {filteredInteractions.length} evaluations
            </div>
            <div className="flex items-center gap-2">
              <button className="px-4 py-2 bg-white border border-[#E2E8F0] rounded-lg text-sm font-semibold text-[#94A3B8] disabled:opacity-50 shadow-sm" disabled>
                Previous
              </button>
              <button className="px-4 py-2 bg-white border border-[#E2E8F0] rounded-lg text-sm font-semibold text-[#334155] hover:bg-[#F8FAFC] shadow-sm transition-colors">
                Next page
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
