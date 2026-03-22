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
          <p className="text-foreground text-sm">Failed to load dashboard data</p>
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: "Average Score", value: `${data.kpis.avgScore}%`, sub: "overall average", icon: BarChart2, color: "text-primary", bg: "bg-primary/10" },
          { label: "Calls Processed", value: data.kpis.totalCalls, sub: "completed calls", icon: Phone, color: "text-success", bg: "bg-success/10" },
          { label: "Resolution Rate", value: `${data.kpis.resolutionRate}%`, sub: "of completed calls", icon: CheckCircle, color: "text-success", bg: "bg-success/10" },
          { label: "Policy Violations", value: data.kpis.violationCount, sub: "interactions flagged", icon: AlertTriangle, color: "text-destructive", bg: "bg-destructive/10" },
        ].map((kpi) => (
          <div key={kpi.label} className="bg-card rounded-[14px] border border-border p-5 transition-all">
            <div className="flex items-start justify-between mb-3">
              <h2 className="text-[12px] font-bold text-muted-foreground uppercase tracking-wider">{kpi.label}</h2>
              <div className={`w-9 h-9 ${kpi.bg} rounded-xl flex items-center justify-center`}>
                <kpi.icon className={`w-[18px] h-[18px] ${kpi.color}`} />
              </div>
            </div>
            <div className="text-[36px] font-bold text-foreground leading-none mb-1" style={{ fontFamily: "var(--font-serif)" }}>
              {kpi.value}
            </div>
            <div className="text-[12px] text-muted-foreground font-medium">{kpi.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Weekly Trend */}
        <div className="lg:col-span-2 bg-card rounded-[14px] border border-border p-6 transition-all">
          <h2 className="text-[16px] font-bold text-foreground mb-1">Weekly Score Trends</h2>
          <p className="text-[11px] italic text-muted-foreground mb-6">interaction_scores.overall_score avg per day</p>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.weeklyTrend}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.3} />
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} dy={10} />
                <YAxis hide domain={[70, 95]} />
                <Tooltip content={<MinimalTooltip />} />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="var(--primary)"
                  strokeWidth={3}
                  dot={{ r: 4, fill: "var(--primary)", strokeWidth: 2, stroke: "var(--card)" }}
                  activeDot={{ r: 6, strokeWidth: 0 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Emotion Dist */}
        <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
          <h2 className="text-[16px] font-bold text-foreground mb-1">Emotion Distribution</h2>
          <p className="text-[11px] italic text-muted-foreground mb-6">utterances.emotion — distribution</p>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.emotionDistribution} cx="50%" cy="50%" innerRadius={50} outerRadius={70} paddingAngle={4} dataKey="value">
                  {data.emotionDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                  ))}
                </Pie>
                <Tooltip content={<MinimalTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {data.emotionDistribution.map((e) => (
              <div key={e.name} className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
                <span className="text-[11px] font-bold text-muted-foreground uppercase">{e.name} {e.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Leaderboard & Recent */}
      <div className="grid grid-cols-1 lg:grid-cols-7 gap-6">
        {/* Leaderboard */}
        <div className="lg:col-span-3 bg-card rounded-[14px] border border-border p-6">
          <div className="flex items-center gap-2 mb-1">
            <Star className="w-4 h-4 text-warning" />
            <h2 className="text-[16px] font-bold text-foreground">Agent Leaderboard</h2>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-6">avg_overall_score per agent</p>
          <div className="space-y-4">
            {leaderboard.map((a, i) => (
              <div key={a.name} className="flex items-center gap-4">
                <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-[12px] font-bold ${i === 0 ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" : "bg-muted text-muted-foreground"}`}>
                  {i + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[13px] font-bold text-foreground truncate">{a.name}</span>
                    <span className="text-[13px] font-black text-primary">{a.overallScore}%</span>
                  </div>
                  <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary rounded-full transition-all duration-1000" style={{ width: `${a.overallScore}%` }} />
                  </div>
                </div>
                {a.trend === "up" ? <TrendingUp className="w-4 h-4 text-success" /> : <TrendingDown className="w-4 h-4 text-destructive" />}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Interactions */}
        <div className="lg:col-span-4 bg-card rounded-[14px] border border-border p-6">
          <h2 className="text-[16px] font-bold text-foreground mb-1">Recent Interactions</h2>
          <p className="text-[11px] italic text-muted-foreground mb-6">sorted by overall_score asc</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {sortedInteractions.slice(0, 4).map((item) => (
              <Link key={item.id} to={`/manager/inspector/${item.id}`} className="group p-4 border border-border rounded-xl hover:bg-muted/5 transition-all">
                <div className="flex justify-between items-start mb-2">
                  <div className="min-w-0">
                    <h4 className="text-[13px] font-bold text-foreground truncate group-hover:text-primary transition-colors">{item.agentName}</h4>
                    <p className="text-[10px] text-muted-foreground">{item.date} · {item.duration}</p>
                  </div>
                  <div className="text-[18px] font-bold text-primary" style={{ fontFamily: "var(--font-serif)" }}>{item.overallScore}%</div>
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${item.hasViolation ? "bg-destructive/10 text-destructive border-destructive/20" : "bg-success/10 text-success border-success/20"}`}>
                    {item.hasViolation ? "VIOLATION" : "CLEAN"}
                  </span>
                  <span className="text-[10px] font-bold text-muted-foreground ml-auto">{item.language}</span>
                </div>
              </Link>
            ))}
          </div>
          <Link to="/manager/inspector" className="block text-center mt-6 text-[12px] font-bold text-primary hover:underline">View All Interactions →</Link>
        </div>
      </div>
    </div>
  );
}
