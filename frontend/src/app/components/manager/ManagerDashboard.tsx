import { useState, useEffect } from "react";
import { Link } from "react-router";
import {
  BarChart2,
  Phone,
  CheckCircle,
  AlertTriangle,
  Star,
  TrendingUp,
  TrendingDown,
  Loader2,
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
import { getDashboardStats, type DashboardStats } from "../../services/api";

const MinimalTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/90 backdrop-blur-sm border border-border px-2 py-1 rounded-lg shadow-xl -mt-8">
        <p className="text-[12px] font-bold text-primary">{payload[0].value}%</p>
      </div>
    );
  }
  return null;
};

export function ManagerDashboard() {
  const [data, setData] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDashboardStats()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading dashboard...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-warning mx-auto mb-3" />
          <p className="text-muted-foreground text-sm">Failed to load dashboard data</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const sortedInteractions = [...data.interactions].sort((a, b) => a.overallScore - b.overallScore);
  const leaderboard = [...data.agentPerformance].sort((a, b) => b.overallScore - a.overallScore);

  return (
    <div className="p-6 space-y-6">
      {/* KPI Cards Row */}
      <div className="grid grid-cols-4 gap-4">
        {/* Average Score */}
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <span className="text-label">
              Average Score
            </span>
            <div className="w-9 h-9 bg-primary/10 rounded-xl flex items-center justify-center">
              <BarChart2 className="w-[18px] h-[18px] text-primary" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-primary mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.kpis.avgScore}%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            overall average
          </div>
        </div>

        {/* Calls Processed */}
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Calls Processed
            </span>
            <div className="w-9 h-9 bg-success/10 rounded-xl flex items-center justify-center">
              <Phone className="w-[18px] h-[18px] text-success" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-success mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.kpis.totalCalls}
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            completed calls
          </div>
        </div>

        {/* Resolution Rate */}
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Resolution Rate
            </span>
            <div className="w-9 h-9 bg-success/10 rounded-xl flex items-center justify-center">
              <CheckCircle className="w-[18px] h-[18px] text-success" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-success mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.kpis.resolutionRate}%
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            of completed calls
          </div>
        </div>

        {/* Policy Violations */}
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Policy Violations
            </span>
            <div className="w-9 h-9 bg-destructive/10 rounded-xl flex items-center justify-center">
              <AlertTriangle className="w-[18px] h-[18px] text-destructive" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-destructive mb-1" style={{ fontFamily: 'var(--font-serif)' }}>
            {data.kpis.violationCount}
          </div>
          <div className="text-[12px] text-[#9CA3AF]">
            interactions flagged
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Weekly Score Trends */}
        <div className="col-span-2 bg-card rounded-[14px] border border-border p-5 transition-all">
          <h3 className="text-[16px] font-semibold text-foreground mb-1">
            Weekly Score Trends
          </h3>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            interaction_scores.overall_score avg, grouped by interaction_date
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data.weeklyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} opacity={0.5} />
              <XAxis dataKey="day" tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} axisLine={{ stroke: 'var(--border)' }} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} domain={[70, 95]} axisLine={{ stroke: 'var(--border)' }} />
              <Tooltip content={<MinimalTooltip />} cursor={{ stroke: 'var(--primary)', strokeWidth: 1 }} />
              <Line 
                type="monotone" 
                dataKey="score" 
                stroke="var(--primary)" 
                strokeWidth={3}
                dot={{ fill: 'var(--primary)', r: 4, strokeWidth: 2, stroke: 'var(--card)' }}
                activeDot={{ r: 6, strokeWidth: 0 }}
                name="Avg Score"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Emotion Distribution */}
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <h3 className="text-[16px] font-semibold text-foreground mb-1">
            Emotion Distribution
          </h3>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            utterances.emotion — distribution
          </p>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie
                data={data.emotionDistribution}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={2}
                dataKey="value"
                isAnimationActive={true}
              >
                {data.emotionDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                ))}
              </Pie>
              <Tooltip content={<MinimalTooltip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-3">
            {data.emotionDistribution.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-[12px] text-muted-foreground font-medium">
                  {item.name} {item.value}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Policy Compliance */}
      <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Policy Compliance by Category
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          policy_compliance JOIN company_policies — compliance rate per policy_category
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data.policyCompliance} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} axisLine={{ stroke: 'var(--border)' }} />
            <YAxis dataKey="category" type="category" width={150} tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} axisLine={{ stroke: 'var(--border)' }} />
            <Tooltip content={<MinimalTooltip />} cursor={{ fill: 'var(--primary)', opacity: 0.05 }} />
            <Bar dataKey="rate" radius={[0, 6, 6, 0]} activeBar={{ fillOpacity: 0.8 }}>
              {data.policyCompliance.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Agent Performance */}
      <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Agent Performance Breakdown
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          interaction_scores: empathy_score · policy_score · resolution_score per agent
        </p>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={data.agentPerformance}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} opacity={0.5} />
            <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} axisLine={{ stroke: 'var(--border)' }} />
            <YAxis domain={[60, 100]} tick={{ fontSize: 12, fill: 'var(--muted-foreground)' }} axisLine={{ stroke: 'var(--border)' }} />
            <Tooltip content={<MinimalTooltip />} cursor={{ fill: 'var(--muted)', opacity: 0.2 }} />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            <Bar dataKey="empathy" fill="var(--primary)" radius={[4, 4, 0, 0]} name="Empathy" activeBar={{ fillOpacity: 0.8 }} />
            <Bar dataKey="policy" fill="var(--success)" radius={[4, 4, 0, 0]} name="Policy" activeBar={{ fillOpacity: 0.8 }} />
            <Bar dataKey="resolution" fill="var(--accent-foreground)" radius={[4, 4, 0, 0]} name="Resolution" activeBar={{ fillOpacity: 0.8 }} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-7 gap-4">
        {/* Agent Leaderboard */}
        <div className="col-span-2 bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-center gap-2 mb-1">
            <Star className="w-4 h-4 text-warning" />
            <h3 className="text-foreground font-semibold text-[16px] mb-1">
              Agent Leaderboard
            </h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            agent_performance_snapshots — avg_overall_score
          </p>
          <div className="space-y-3">
            {leaderboard.map((agent, index) => (
              <div key={agent.name} className="flex items-center gap-3">
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-[13px] font-bold flex-shrink-0 ${
                    index === 0
                      ? "bg-primary/20 text-primary"
                      : index === 1
                      ? "bg-muted text-muted-foreground"
                      : "bg-background text-muted-foreground"
                  }`}
                >
                  {index + 1}
                </div>
                <span className="text-[13px] font-semibold text-foreground">
                  {agent.name}
                </span>
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  agent.overallScore >= 85
                    ? "bg-success"
                    : agent.overallScore >= 75
                    ? "bg-primary"
                    : "bg-destructive"
                }`}
                style={{ width: `${agent.overallScore}%` }}
              />
            </div>
            <span
              className={`text-[14px] font-black ${
                agent.overallScore >= 85
                  ? "text-success"
                  : agent.overallScore >= 75
                  ? "text-primary"
                  : "text-destructive"
              }`}
            >
              {agent.overallScore}%
            </span>
              {agent.overallScore >= 85 ? (
                <TrendingUp className="w-4 h-4 text-success" />
              ) : (
                <TrendingDown className="w-4 h-4 text-destructive" />
              )}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Interactions */}
        <div className="col-span-5 bg-card rounded-[14px] border border-border p-5 transition-all">
          <h3 className="text-[16px] font-semibold text-foreground mb-1">
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
                className={`block border rounded-[10px] p-3.5 transition-all active:scale-[0.99] ${
                  interaction.hasViolation
                    ? "bg-destructive/5 border-destructive/20 hover:border-destructive/40"
                    : "bg-card border-border hover:border-primary/50 hover:bg-muted/10"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-semibold text-foreground">
                      {interaction.agentName}
                    </span>
                    {interaction.hasViolation && (
                      <span className="px-2 py-0.5 bg-destructive/10 text-destructive rounded-full text-[11px] font-medium">
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
                            ? "var(--success)"
                            : interaction.overallScore >= 75
                            ? "var(--primary)"
                            : "var(--destructive)",
                      }}
                    >
                      {interaction.overallScore}%
                    </div>
                    <div
                      className={`text-[11px] font-medium ${
                        interaction.resolved ? "text-success" : "text-destructive"
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
                  <span className="px-2 py-1 bg-muted/50 text-muted-foreground rounded text-[10px] font-bold uppercase tracking-wider">
                    Empathy
                  </span>
                  <span className="px-2 py-1 bg-muted/50 text-muted-foreground rounded text-[10px] font-bold uppercase tracking-wider">
                    Policy
                  </span>
                  <span className="px-2 py-1 bg-muted/50 text-muted-foreground rounded text-[10px] font-bold uppercase tracking-wider">
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
