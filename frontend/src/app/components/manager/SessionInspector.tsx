import { Link } from "react-router";
import { Search, ChevronDown } from "lucide-react";
import { mockInteractions } from "../../data/mockData";

export function SessionInspector() {
  const sortedInteractions = [...mockInteractions].sort((a, b) => a.overallScore - b.overallScore);

  return (
    <div className="p-6">
      {/* Top Controls */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-[22px] font-bold text-[#111827] mb-1">
            Session Inspector
          </h2>
          <p className="text-[13px] text-[#6B7280]">
            All interactions · sorted by score
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9CA3AF]" />
            <input
              type="text"
              placeholder="Search agent, date, ID…"
              className="w-[200px] h-10 pl-9 pr-3 bg-white border border-[#E5E7EB] rounded-[10px] text-[13px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]"
            />
          </div>

          <button className="flex items-center gap-2 h-10 px-4 bg-white border border-[#E5E7EB] rounded-[10px] text-[13px] text-[#374151] hover:bg-[#F9FAFB] transition-colors">
            All Agents
            <ChevronDown className="w-4 h-4" />
          </button>

          <div className="flex items-center border border-[#E5E7EB] rounded-[10px] overflow-hidden bg-white">
            <button className="px-3 h-10 bg-[#3B82F6] text-white text-[11px] font-semibold">
              Score ↑
            </button>
            <button className="px-3 h-10 text-[#6B7280] text-[11px] font-semibold hover:bg-[#F9FAFB]">
              Date ↓
            </button>
            <button className="px-3 h-10 text-[#6B7280] text-[11px] font-semibold hover:bg-[#F9FAFB]">
              Duration
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
          {sortedInteractions.map((interaction) => (
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
          ))}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[#E5E7EB] flex items-center justify-between">
          <span className="text-[12px] text-[#6B7280]">
            Showing 1–{sortedInteractions.length} of 342
          </span>
          <div className="flex items-center gap-2">
            <button className="px-3 h-8 text-[12px] text-[#6B7280] hover:text-[#111827] disabled:opacity-40" disabled>
              ← Prev
            </button>
            <button className="px-3 h-8 text-[12px] text-[#6B7280] hover:text-[#111827]">
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
