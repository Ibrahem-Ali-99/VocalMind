import { useState, useEffect } from "react";
import { Link, useParams } from "react-router";
import { Star, Phone, Target, Zap, Loader2, AlertTriangle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getAgentProfile, getAgents, type AgentProfile } from "../../services/api";

export function AgentDashboard() {
  const { agentId: routeAgentId } = useParams();
  const [data, setData] = useState<AgentProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAgent = async () => {
      try {
        let targetId = routeAgentId;
        if (!targetId) {
          // No auth — auto-select the first agent
          const agents = await getAgents();
          if (agents.length === 0) {
            setError("No agents found in the database");
            return;
          }
          targetId = agents[0].id;
        }
        const profile = await getAgentProfile(targetId);
        setData(profile);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    };
    loadAgent();
  }, [routeAgentId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-[#10B981] animate-spin" />
        <span className="ml-3 text-[#6B7280] text-sm">Loading your dashboard...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-[#6B7280] text-sm">Failed to load agent data</p>
          <p className="text-[#9CA3AF] text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Hero Card */}
      <div
        className="rounded-2xl p-7 text-white"
        style={{ background: "linear-gradient(135deg, #065F46 0%, #0D9488 100%)" }}
      >
        <div className="flex items-start justify-between mb-6">
          {/* Left */}
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-white/60 mb-2">
              MY PERFORMANCE
            </div>
            <h2 className="text-[28px] leading-none mb-2" style={{ fontFamily: 'var(--font-serif)' }}>
              {data.name}
            </h2>
            <p className="text-[13px] text-white/60">
              Agent · VocalMind Corp
            </p>
          </div>

          {/* Right */}
          <div className="text-right">
            <div className="text-[56px] leading-none mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
              {data.overallScore}%
            </div>
            <div className="text-[11px] text-white/50">
              Overall Score
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-white/20 mb-5" />

        {/* Stats Row */}
        <div className="grid grid-cols-3 gap-6">
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
              {data.callsThisWeek}
            </div>
            <div className="text-[11px] text-white/50">
              Calls This Week
            </div>
          </div>
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
              #{data.teamRank}
            </div>
            <div className="text-[11px] text-white/50">
              Team Rank
            </div>
          </div>
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
              {data.resolutionRate}%
            </div>
            <div className="text-[11px] text-white/50">
              Resolution Rate
            </div>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-4">
        {/* Avg Score */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Avg Score
            </span>
            <div className="w-9 h-9 bg-[#ECFDF5] rounded-xl flex items-center justify-center">
              <Star className="w-[18px] h-[18px] text-[#10B981]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#10B981] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.overallScore}%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            from all calls
          </div>
        </div>

        {/* Calls Today */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Calls Today
            </span>
            <div className="w-9 h-9 bg-[#ECFDF5] rounded-xl flex items-center justify-center">
              <Phone className="w-[18px] h-[18px] text-[#10B981]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#10B981] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            8
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            processed calls
          </div>
        </div>

        {/* Resolution */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Resolution
            </span>
            <div className="w-9 h-9 bg-[#F5F3FF] rounded-xl flex items-center justify-center">
              <Target className="w-[18px] h-[18px] text-[#8B5CF6]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#8B5CF6] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.resolutionRate}%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            issues resolved
          </div>
        </div>

        {/* Avg Response */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Avg Response
            </span>
            <div className="w-9 h-9 bg-[#FFFBEB] rounded-xl flex items-center justify-center">
              <Zap className="w-[18px] h-[18px] text-[#F59E0B]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#F59E0B] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.avgResponseTime}s
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            response time
          </div>
        </div>
      </div>

      {/* Two Column Row */}
      <div className="grid grid-cols-2 gap-6">
        {/* My Score Breakdown */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
            My Score Breakdown
          </h3>
          <p className="text-[11px] italic text-[#9CA3AF] mb-5">
            interaction_scores averaged for my calls — empathy, policy, resolution
          </p>

          <div className="space-y-4">
            {/* Empathy Score */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-[#374151]">Empathy Score</span>
                <span className="text-[13px] font-semibold text-[#3B82F6]">{data.empathyScore}%</span>
              </div>
              <div className="h-2.5 bg-[#F3F4F6] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#3B82F6] rounded-full"
                  style={{ width: `${data.empathyScore}%` }}
                />
              </div>
            </div>

            {/* Policy Adherence */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-[#374151]">Policy Adherence</span>
                <span className="text-[13px] font-semibold text-[#10B981]">{data.policyScore}%</span>
              </div>
              <div className="h-2.5 bg-[#F3F4F6] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#10B981] rounded-full"
                  style={{ width: `${data.policyScore}%` }}
                />
              </div>
            </div>

            {/* Resolution */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-[#374151]">Resolution</span>
                <span className="text-[13px] font-semibold text-[#8B5CF6]">{data.resolutionScore}%</span>
              </div>
              <div className="h-2.5 bg-[#F3F4F6] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#8B5CF6] rounded-full"
                  style={{ width: `${data.resolutionScore}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* My Weekly Trend */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
            My Weekly Trend
          </h3>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            interaction_scores for my calls this week — overall score trend
          </p>
          <ResponsiveContainer width="100%" height={190}>
            <LineChart data={data.weeklyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
              <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#6B7280' }} />
              <YAxis domain={[70, 100]} tick={{ fontSize: 12, fill: '#6B7280' }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#10B981"
                strokeWidth={3}
                dot={{ fill: '#FFFFFF', stroke: '#10B981', strokeWidth: 2, r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* My Recent Calls */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          My Recent Calls
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          interactions WHERE agent_id = [me] — personal calls only, sorted by date desc
        </p>

        <div className="space-y-3">
          {data.recentCalls.map((call) => (
            <Link
              key={call.id}
              to={`/agent/calls/${call.id}`}
              className={`block border rounded-[10px] p-3.5 transition-all hover:shadow-md ${
                call.hasReview
                  ? "bg-[#FFFBEB] border-[#FDE68A]"
                  : "bg-white border-[#E5E7EB] hover:border-[#10B981]"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-semibold text-[#111827]">
                    {call.time}
                  </span>
                  {call.hasReview && (
                    <span className="px-2 py-0.5 bg-[#FEF3C7] text-[#92400E] rounded-full text-[11px] font-medium">
                      Review needed
                    </span>
                  )}
                </div>
                <div className="text-right">
                  <div
                    className="text-[22px] leading-none mb-1"
                    style={{
                      fontFamily: 'var(--font-serif)',
                      color: call.score >= 85 ? "#10B981" : call.score >= 75 ? "#3B82F6" : "#F59E0B",
                    }}
                  >
                    {call.score}%
                  </div>
                  <div className={`text-[11px] font-medium ${call.resolved ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                    {call.resolved ? "✓ Resolved" : "✗ Unresolved"}
                  </div>
                </div>
              </div>
              <div className="text-[12px] text-[#9CA3AF]">
                {call.duration} · {call.language}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
