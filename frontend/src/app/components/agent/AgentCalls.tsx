import { useEffect, useState } from "react";
import { Link } from "react-router";
import { AlertTriangle, Clock3, Loader2, PhoneCall, ShieldAlert } from "lucide-react";
import { getInteractions, type InteractionSummary } from "../../services/api";

export function AgentCalls() {
  const [calls, setCalls] = useState<InteractionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadCalls = async () => {
      try {
        const rows = await getInteractions();
        setCalls(rows);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load calls");
      } finally {
        setLoading(false);
      }
    };

    void loadCalls();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-[#10B981] animate-spin" />
        <span className="ml-3 text-[#6B7280] text-sm">Loading your calls...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-[#6B7280] text-sm">Failed to load calls</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const completedCalls = calls.filter((call) => call.status === "completed").length;
  const reviewNeeded = calls.filter((call) => call.hasViolation).length;

  return (
    <div className="p-6 space-y-6">
      <div
        className="rounded-2xl p-7 text-white"
        style={{ background: "linear-gradient(135deg, #0F172A 0%, #102A43 100%)" }}
      >
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-white/60 mb-2">
              MY CALLS
            </div>
            <h2 className="text-[28px] leading-none mb-3" style={{ fontFamily: "var(--font-serif)" }}>
              Review recent conversations
            </h2>
            <p className="max-w-2xl text-[13px] text-white/65">
              This view only includes your own sessions. Open any call to review transcript, emotion shifts, and saved coaching insights.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 min-w-[260px]">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Total</div>
              <div className="text-[30px] leading-none" style={{ fontFamily: "var(--font-serif)" }}>{calls.length}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Completed</div>
              <div className="text-[30px] leading-none" style={{ fontFamily: "var(--font-serif)" }}>{completedCalls}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Need Review</div>
              <div className="text-[30px] leading-none" style={{ fontFamily: "var(--font-serif)" }}>{reviewNeeded}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">Latest Status</div>
              <div className="text-[18px] leading-none font-semibold">{calls[0]?.status ?? "No calls"}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-card rounded-[14px] border border-border p-5 transition-all">
        <h2 className="text-[15px] font-bold text-foreground mb-1">Recent Calls</h2>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          Personal sessions only, newest first.
        </p>

        {calls.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border bg-muted/20 px-6 py-10 text-center">
            <PhoneCall className="w-10 h-10 text-muted-foreground/60 mx-auto mb-3" />
            <h3 className="text-[15px] font-semibold text-foreground mb-1">No calls yet</h3>
            <p className="text-[13px] text-muted-foreground">
              Once your sessions are processed, they will appear here.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {calls.map((call) => {
              const scoreTone =
                call.overallScore >= 85
                  ? "text-success"
                  : call.overallScore >= 75
                    ? "text-primary"
                    : "text-warning";

              return (
                <Link
                  key={call.id}
                  to={`/agent/calls/${call.id}`}
                  className="block rounded-2xl border border-border bg-background/60 p-4 transition-all hover:border-primary/40 hover:bg-muted/10"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[14px] font-semibold text-foreground">{call.date}</span>
                        <span className="text-muted-foreground/60">•</span>
                        <span className="text-[13px] text-muted-foreground">{call.time}</span>
                        {call.hasViolation && (
                          <span className="inline-flex items-center gap-1 rounded-full border border-warning/30 bg-warning/10 px-2.5 py-1 text-[11px] font-semibold text-warning">
                            <ShieldAlert className="w-3 h-3" />
                            Review needed
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-[12px] text-muted-foreground flex-wrap">
                        <span>{call.duration}</span>
                        <span>•</span>
                        <span>{call.language}</span>
                        <span>•</span>
                        <span className="capitalize">{call.status}</span>
                        <span>•</span>
                        <span className="inline-flex items-center gap-1">
                          <Clock3 className="w-3.5 h-3.5" />
                          {call.responseTime}
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className={`text-[30px] leading-none mb-1 ${scoreTone}`} style={{ fontFamily: "var(--font-serif)" }}>
                        {call.overallScore}%
                      </div>
                      <div className={`text-[12px] font-semibold ${call.resolved ? "text-success" : "text-destructive"}`}>
                        {call.resolved ? "Resolved" : "Unresolved"}
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
