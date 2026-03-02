import { Link } from "react-router";
import {
  BarChart2,
  Phone,
  CheckCircle,
  AlertTriangle,
  Star,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from "recharts";
import {
  mockWeeklyTrend,
  mockEmotionDistribution,
  mockPolicyCompliance,
  mockAgentPerformance,
  mockInteractions,
} from "../../data/mockData";

export function ManagerDashboard() {
  const sortedInteractions = [...mockInteractions].sort((a, b) => a.overallScore - b.overallScore);
  const leaderboard = [...mockAgentPerformance].sort((a, b) => b.overallScore - a.overallScore);

  return (
    <div className="p-6 space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-4 gap-4">
        {/* Average Score */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Average Score
            </span>
            <div className="w-9 h-9 bg-[#EFF6FF] rounded-xl flex items-center justify-center">
              <BarChart2 className="w-[18px] h-[18px] text-[#3B82F6]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#2563EB] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            84.2%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            ↑ 2.3% from last week
          </div>
        </div>

        {/* Calls Processed */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Calls Processed
            </span>
            <div className="w-9 h-9 bg-[#ECFDF5] rounded-xl flex items-center justify-center">
              <Phone className="w-[18px] h-[18px] text-[#10B981]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#059669] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            342
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            this week
          </div>
        </div>

        {/* Resolution Rate */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Resolution Rate
            </span>
            <div className="w-9 h-9 bg-[#ECFDF5] rounded-xl flex items-center justify-center">
              <CheckCircle className="w-[18px] h-[18px] text-[#10B981]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#059669] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            88%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            of completed calls
          </div>
        </div>

        {/* Policy Violations */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Policy Violations
            </span>
            <div className="w-9 h-9 bg-[#FEF2F2] rounded-xl flex items-center justify-center">
              <AlertTriangle className="w-[18px] h-[18px] text-[#EF4444]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-[#DC2626] mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            12
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            interactions flagged
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Weekly Score Trends */}
        <div className="col-span-2 bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
            Weekly Score Trends
          </h3>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            interaction_scores.overall_score avg, grouped by interaction_date
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={mockWeeklyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
              <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#6B7280' }} />
              <YAxis tick={{ fontSize: 12, fill: '#6B7280' }} domain={[70, 95]} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="score" 
                stroke="#3B82F6" 
                strokeWidth={2.5}
                dot={{ fill: '#3B82F6', r: 4 }}
                name="Avg Score"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Emotion Distribution */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
            Emotion Distribution
          </h3>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            utterances.emotion — distribution
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={mockEmotionDistribution}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={2}
                dataKey="value"
              >
                {mockEmotionDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-3">
            {mockEmotionDistribution.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-[12px] text-[#6B7280]">
                  {item.name} {item.value}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Policy Compliance */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Policy Compliance by Category
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          policy_compliance JOIN company_policies — compliance rate per policy_category
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={mockPolicyCompliance} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#6B7280' }} />
            <YAxis dataKey="category" type="category" width={150} tick={{ fontSize: 12, fill: '#6B7280' }} />
            <Tooltip />
            <Bar dataKey="rate" radius={[0, 6, 6, 0]}>
              {mockPolicyCompliance.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Agent Performance */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Agent Performance Breakdown
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          interaction_scores: empathy_score · policy_score · resolution_score per agent
        </p>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={mockAgentPerformance}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#6B7280' }} />
            <YAxis domain={[60, 100]} tick={{ fontSize: 12, fill: '#6B7280' }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Bar dataKey="empathy" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Empathy" />
            <Bar dataKey="policy" fill="#10B981" radius={[4, 4, 0, 0]} name="Policy" />
            <Bar dataKey="resolution" fill="#8B5CF6" radius={[4, 4, 0, 0]} name="Resolution" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-7 gap-4">
        {/* Agent Leaderboard */}
        <div className="col-span-2 bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <Star className="w-4 h-4 text-[#F59E0B]" />
            <h3 className="text-[16px] font-semibold text-[#111827]">
              Agent Leaderboard
            </h3>
          </div>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            agent_performance_snapshots — avg_overall_score
          </p>
          <div className="space-y-3">
            {leaderboard.map((agent, index) => (
              <div key={agent.name} className="flex items-center gap-3">
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-[13px] font-bold flex-shrink-0 ${
                    index === 0
                      ? "bg-[#FEF3C7] text-[#D97706]"
                      : index === 1
                      ? "bg-[#F3F4F6] text-[#6B7280]"
                      : "bg-[#F9FAFB] text-[#9CA3AF]"
                  }`}
                >
                  {index + 1}
                </div>
                <span className="text-[13px] font-semibold text-[#111827]">
                  {agent.name}
                </span>
                <div className="flex-1 h-1.5 bg-[#F3F4F6] rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      agent.overallScore >= 85
                        ? "bg-[#10B981]"
                        : agent.overallScore >= 75
                        ? "bg-[#3B82F6]"
                        : "bg-[#F59E0B]"
                    }`}
                    style={{ width: `${agent.overallScore}%` }}
                  />
                </div>
                <span
                  className={`text-[14px] font-black ${
                    agent.overallScore >= 85
                      ? "text-[#10B981]"
                      : agent.overallScore >= 75
                      ? "text-[#3B82F6]"
                      : "text-[#F59E0B]"
                  }`}
                >
                  {agent.overallScore}%
                </span>
                {agent.trend === "up" ? (
                  <TrendingUp className="w-4 h-4 text-[#10B981]" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-[#F59E0B]" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Interactions */}
        <div className="col-span-5 bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
            Recent Interactions
          </h3>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            interactions JOIN users JOIN interaction_scores — sorted by overall_score asc
          </p>
          <div className="space-y-3 max-h-[360px] overflow-y-auto">
            {sortedInteractions.slice(0, 4).map((interaction) => (
              <Link
                key={interaction.id}
                to={`/manager/inspector/${interaction.id}`}
                className={`block border rounded-[10px] p-3.5 transition-all hover:shadow-md ${
                  interaction.hasViolation
                    ? "bg-[#FFF5F5] border-[#FECACA]"
                    : "bg-white border-[#E5E7EB] hover:border-[#3B82F6]"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-semibold text-[#111827]">
                      {interaction.agentName}
                    </span>
                    {interaction.hasViolation && (
                      <span className="px-2 py-0.5 bg-[#FEE2E2] text-[#DC2626] rounded-full text-[11px] font-medium">
                        Violation
                      </span>
                    )}
                  </div>
                  <div className="text-right">
                    <div
                      className="text-[22px] leading-none mb-1"
                      style={{
                        fontFamily: 'var(--font-serif)',
                        color:
                          interaction.overallScore >= 85
                            ? "#10B981"
                            : interaction.overallScore >= 75
                            ? "#3B82F6"
                            : "#F59E0B",
                      }}
                    >
                      {interaction.overallScore}%
                    </div>
                    <div
                      className={`text-[11px] font-medium ${
                        interaction.resolved ? "text-[#10B981]" : "text-[#EF4444]"
                      }`}
                    >
                      {interaction.resolved ? "✓ Resolved" : "✗ Unresolved"}
                    </div>
                  </div>
                </div>
                <div className="text-[12px] text-[#9CA3AF] mb-2">
                  {interaction.date} · {interaction.time} · {interaction.duration} · {interaction.language}
                </div>
                <div className="flex gap-2">
                  <span className="px-2 py-1 bg-[#F3F4F6] text-[#6B7280] rounded text-[11px]">
                    Empathy
                  </span>
                  <span className="px-2 py-1 bg-[#F3F4F6] text-[#6B7280] rounded text-[11px]">
                    Policy
                  </span>
                  <span className="px-2 py-1 bg-[#F3F4F6] text-[#6B7280] rounded text-[11px]">
                    Resolution
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
