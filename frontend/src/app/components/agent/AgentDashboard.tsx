import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { Star, Phone, Target, Zap, Loader2, AlertTriangle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getAgentProfile, getAgents, getUserMe, type AgentProfile } from "../../services/api";

const MinimalTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/90 backdrop-blur-sm border border-border px-2 py-1 rounded-lg shadow-xl -mt-8">
        <p className="text-[12px] font-bold text-success">{payload[0].value}%</p>
      </div>
    );
  }
  return null;
};

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
          try {
            const currentUser = await getUserMe();
            if (currentUser.role === "agent") {
              targetId = currentUser.id;
            }
          } catch {
            // Fall back to the first available agent when auth lookup is unavailable.
          }
        }

        if (!targetId) {
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

    void loadAgent();
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
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const displayAvgResponseTime = data.avgResponseTime.trim().toLowerCase().endsWith("s")
    ? data.avgResponseTime.trim()
    : `${data.avgResponseTime}s`;

  return (
    <div className="p-6 space-y-6">
      <div
        className="rounded-2xl p-7 text-white"
        style={{ background: "linear-gradient(135deg, #064E3B 0%, #0F766E 100%)" }}
      >
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-white/60 mb-2">
              MY PERFORMANCE
            </div>
            <h2 className="text-[28px] leading-none mb-2" style={{ fontFamily: "var(--font-serif)" }}>
              {data.name}
            </h2>
            <p className="text-[13px] text-white/60">
              {data.role} • VocalMind Corp
            </p>
          </div>

          <div className="text-right">
            <div className="text-[56px] leading-none mb-1" style={{ fontFamily: "var(--font-serif)" }}>
              {data.overallScore}%
            </div>
            <div className="text-[11px] text-white/50">
              Overall Score
            </div>
          </div>
        </div>

        <div className="h-px bg-primary/20 mb-5" />

        <div className="grid grid-cols-3 gap-6">
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: "var(--font-serif)" }}>
              {data.callsThisWeek}
            </div>
            <div className="text-[11px] text-white/50">
              Calls This Week
            </div>
          </div>
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: "var(--font-serif)" }}>
              #{data.teamRank}
            </div>
            <div className="text-[11px] text-white/50">
              Team Rank
            </div>
          </div>
          <div>
            <div className="text-[36px] leading-none mb-1" style={{ fontFamily: "var(--font-serif)" }}>
              {data.resolutionRate}%
            </div>
            <div className="text-[11px] text-white/50">
              Resolution Rate
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <h2 className="text-[13px] font-bold text-muted-foreground">Overall Score</h2>
            <div className="w-9 h-9 bg-success/10 rounded-xl flex items-center justify-center">
              <Star className="w-[18px] h-[18px] text-success" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-success mb-1" style={{ fontFamily: "var(--font-serif)" }}>
            {data.overallScore}%
          </div>
          <div className="text-[12px] text-muted-foreground">
            from all calls
          </div>
        </div>

        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <h2 className="text-[13px] font-bold text-muted-foreground">Total Calls</h2>
            <div className="w-9 h-9 bg-success/10 rounded-xl flex items-center justify-center">
              <Phone className="w-[18px] h-[18px] text-success" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-success mb-1" style={{ fontFamily: "var(--font-serif)" }}>
            {data.totalCalls}
          </div>
          <div className="text-[12px] text-muted-foreground">
            processed calls
          </div>
        </div>

        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <h2 className="text-[13px] font-bold text-muted-foreground">Resolution Rate</h2>
            <div className="w-9 h-9 bg-[#F5F3FF] rounded-xl flex items-center justify-center">
              <Target className="w-[18px] h-[18px] text-[#8B5CF6]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-accent-foreground mb-1" style={{ fontFamily: "var(--font-serif)" }}>
            {data.resolutionRate}%
          </div>
          <div className="text-[12px] text-muted-foreground">
            issues resolved
          </div>
        </div>

        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <div className="flex items-start justify-between mb-3">
            <h2 className="text-[13px] font-bold text-muted-foreground">Avg Response</h2>
            <div className="w-9 h-9 bg-[#FFFBEB] rounded-xl flex items-center justify-center">
              <Zap className="w-[18px] h-[18px] text-[#F59E0B]" />
            </div>
          </div>
          <div className="text-[40px] leading-none text-warning mb-1" style={{ fontFamily: "var(--font-serif)" }}>
            {displayAvgResponseTime}
          </div>
          <div className="text-[12px] text-muted-foreground">
            response time
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <h2 className="text-[15px] font-bold text-foreground mb-1">My Score Breakdown</h2>
          <p className="text-[11px] italic text-muted-foreground mb-5">
            Averaged across my calls: empathy, policy, and resolution.
          </p>

          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-foreground">Empathy Score</span>
                <span className="text-[13px] font-semibold text-primary">{data.empathyScore}%</span>
              </div>
              <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all"
                  style={{ width: `${data.empathyScore}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-foreground">Policy Adherence</span>
                <span className="text-[13px] font-semibold text-success">{data.policyScore}%</span>
              </div>
              <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-success rounded-full transition-all"
                  style={{ width: `${data.policyScore}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[13px] text-foreground">Resolution</span>
                <span className="text-[13px] font-semibold text-accent-foreground">{data.resolutionScore}%</span>
              </div>
              <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent-foreground rounded-full"
                  style={{ width: `${data.resolutionScore}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
          <h2 className="text-[15px] font-bold text-foreground mb-1">My Weekly Trend</h2>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            Interaction score trend for my calls this week.
          </p>
          <ResponsiveContainer width="100%" height={190}>
            <LineChart data={data.weeklyTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} opacity={0.5} />
              <XAxis dataKey="day" tick={{ fontSize: 12, fill: "var(--muted-foreground)" }} axisLine={{ stroke: "var(--border)" }} />
              <YAxis domain={[70, 100]} tick={{ fontSize: 12, fill: "var(--muted-foreground)" }} axisLine={{ stroke: "var(--border)" }} />
              <Tooltip content={<MinimalTooltip />} cursor={{ stroke: "var(--success)", strokeWidth: 1 }} />
              <Line
                type="monotone"
                dataKey="score"
                stroke="var(--success)"
                strokeWidth={3}
                dot={{ fill: "var(--card)", stroke: "var(--success)", strokeWidth: 2, r: 5 }}
                activeDot={{ r: 7, strokeWidth: 0 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
        <h2 className="text-[15px] font-bold text-foreground mb-1">My Recent Calls</h2>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          Personal calls only, sorted by date descending.
        </p>

        <div className="space-y-3">
          {data.recentCalls.map((call) => (
            <Link
              key={call.id}
              to={`/agent/calls/${call.id}`}
              className={`block border rounded-[10px] p-3.5 transition-all active:scale-[0.99] ${
                call.hasReview
                  ? "bg-warning/5 border-warning/20 hover:border-warning/40"
                  : "bg-card border-border hover:border-success/50 hover:bg-muted/10"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-semibold text-foreground">
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
                      fontFamily: "var(--font-serif)",
                      color: call.score >= 85 ? "var(--success)" : call.score >= 75 ? "var(--primary)" : "var(--warning)",
                    }}
                  >
                    {call.score}%
                  </div>
                  <div className={`text-[11px] font-bold ${call.resolved ? "text-success" : "text-destructive"}`}>
                    {call.resolved ? "Resolved" : "Unresolved"}
                  </div>
                </div>
              </div>
              <div className="text-[12px] text-muted-foreground">
                {call.duration} • {call.language}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
