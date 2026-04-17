import { Link, useParams } from "react-router";
import { ArrowLeft, Play, Headphones, ThumbsUp, ThumbsDown, CheckCircle, XCircle, Flag, Loader2, AlertTriangle as AlertTriangleIcon } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { getInteractionDetail, getAudioUrl, type InteractionDetail } from "../../services/api";

export function SessionDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [flaggedEvents, setFlaggedEvents] = useState<string[]>([]);
  const [flaggedViolations, setFlaggedViolations] = useState<string[]>([]);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleJumpTo = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(e => console.error("Playback failed:", e));
    }
  };

  useEffect(() => {
    if (!id) return;
    getInteractionDetail(id)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading session...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangleIcon className="w-10 h-10 text-warning mx-auto mb-3" />
          <p className="text-foreground text-sm">Failed to load session</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const interaction = data.interaction;
  const utterances = data.utterances;
  const emotionEvents = data.emotionEvents;

  const getScoreColor = (score: number) => {
    if (score >= 85) return "var(--success)";
    if (score >= 75) return "var(--primary)";
    return "var(--destructive)";
  };

  const getEmotionStyle = (emotion: string) => {
    switch (emotion) {
      case "neutral":
        return { bg: "var(--muted)", text: "var(--muted-foreground)", label: "Neutral" };
      case "happy":
        return { bg: "rgba(16, 185, 129, 0.1)", text: "var(--success)", label: "Happy" };
      case "angry":
        return { bg: "rgba(239, 68, 68, 0.1)", text: "var(--destructive)", label: "Angry" };
      case "frustrated":
        return { bg: "rgba(245, 158, 11, 0.1)", text: "var(--warning)", label: "Frustrated" };
      default:
        return { bg: "var(--muted)", text: "var(--muted-foreground)", label: "Neutral" };
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Back Button */}
      <Link
        to="/manager/inspector"
        className="inline-flex items-center gap-2 text-[13px] font-semibold text-primary hover:underline"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Session Inspector
      </Link>

      {/* Call Header Card */}
      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="text-label mb-2">SESSION INSPECTOR</div>
            <h2 className="text-[22px] font-bold text-foreground mb-2">
              {interaction.agentName}
            </h2>
            <div className="text-[13px] text-muted-foreground mb-2">
              {interaction.date} · {interaction.time} · {interaction.duration} · {interaction.language}
            </div>
          </div>

          <div className="flex flex-col items-center">
            <div className="relative w-[90px] h-[90px]">
              <svg className="w-full h-full -rotate-90">
                <circle cx="45" cy="45" r="38" fill="none" stroke="var(--border)" strokeWidth="7" />
                <circle
                  cx="45"
                  cy="45"
                  r="38"
                  fill="none"
                  stroke={getScoreColor(interaction.overallScore)}
                  strokeWidth="7"
                  strokeLinecap="round"
                  strokeDasharray={`${(interaction.overallScore / 100) * 238.76} 238.76`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[20px] font-bold" style={{ fontFamily: "var(--font-serif)", color: getScoreColor(interaction.overallScore) }}>
                  {interaction.overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="h-px bg-border mb-6" />

        {/* Score Grid */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Empathy", score: interaction.empathyScore, color: "var(--primary)" },
            { label: "Policy", score: interaction.policyScore, color: "var(--success)" },
            { label: "Resolution", score: interaction.resolutionScore, color: "var(--primary)" },
            { label: "Resp. Time", score: interaction.responseTime, color: "var(--success)", suffix: "s" },
          ].map((s) => (
            <div key={s.label} className="bg-muted/10 rounded-xl p-3 text-center border border-border/50">
              <div className="text-[11px] text-muted-foreground mb-1 uppercase tracking-wider font-bold">{s.label}</div>
              <div className="text-[18px] font-bold" style={{ color: s.color }}>
                {s.score}{s.suffix || "%"}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transcript Card */}
        <div className="bg-card rounded-[14px] border border-border p-6">
          <h3 className="text-[16px] font-bold text-foreground mb-1">Transcript</h3>
          <p className="text-[11px] italic text-muted-foreground mb-4">utterances ordered by sequence_index</p>
          <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
            {utterances.map((u) => {
              const isAgent = u.speaker === "agent";
              return (
                <div key={u.id} className={`flex gap-3 ${isAgent ? "" : "flex-row-reverse"}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${isAgent ? "bg-primary/20 text-primary" : "bg-success/20 text-success"}`}>
                    {isAgent ? "A" : "C"}
                  </div>
                  <div className={`flex-1 p-3 rounded-2xl text-[13px] ${isAgent ? "bg-primary/5 rounded-tl-none" : "bg-success/5 rounded-tr-none"}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-foreground/80">{isAgent ? interaction.agentName : "Customer"}</span>
                      <span className="text-[10px] text-muted-foreground">{u.timestamp}</span>
                    </div>
                    <p className="text-foreground leading-relaxed">{u.text}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="space-y-6">
          {/* Emotion Events */}
          <div className="bg-card rounded-[14px] border border-border p-6">
            <h3 className="text-[16px] font-bold text-foreground mb-1">Emotion Events</h3>
            <p className="text-[11px] italic text-muted-foreground mb-4">emotion_events — AI-detected emotional shifts</p>
            <div className="space-y-4">
              {emotionEvents.map((e) => (
                <div key={e.id} className="p-4 border border-border rounded-xl bg-muted/5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[12px] font-bold text-foreground capitalize">{e.fromEmotion} → {e.toEmotion}</span>
                      <span className="text-[11px] text-muted-foreground">Δ {e.delta}</span>
                      <span className="px-1.5 py-0.5 bg-muted/30 rounded text-[10px] font-bold uppercase">{e.speaker}</span>
                    </div>
                    <button 
                      onClick={() => handleJumpTo(e.jumpToSeconds)}
                      className="text-[11px] font-bold text-primary hover:underline flex items-center gap-1"
                    >
                      <Play className="w-3 h-3 fill-current" />
                      Jump to {e.timestamp}
                    </button>
                  </div>
                  <p className="text-[12px] text-muted-foreground italic leading-relaxed">"{e.justification}"</p>
                  {flaggedEvents.includes(e.id) ? (
                    feedbackSubmitted.includes(e.id) ? (
                      <div className="text-[11px] text-success font-bold mt-2 pt-2 border-t border-border/50">
                        Feedback recorded — queued for model retraining
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 pt-2 border-t border-border/50">
                        <span className="text-[11px] text-muted-foreground">Was this detection accurate?</span>
                        <button onClick={() => setFeedbackSubmitted(prev => [...prev, e.id])} className="text-[11px] font-bold text-success hover:underline">Accurate</button>
                        <button onClick={() => setFeedbackSubmitted(prev => [...prev, e.id])} className="text-[11px] font-bold text-destructive hover:underline">Incorrect</button>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center justify-end pt-2 border-t border-border/50">
                      <button onClick={() => setFlaggedEvents(prev => [...prev, e.id])} className="text-[11px] font-bold text-muted-foreground hover:text-foreground flex items-center gap-1">
                        <Flag className="w-3 h-3" /> Flag as incorrect
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Policy Violations */}
          <div className="bg-card rounded-[14px] border border-border p-6">
            <h3 className="text-[16px] font-bold text-foreground mb-1">Policy Violations</h3>
            <p className="text-[11px] italic text-muted-foreground mb-4">policy_compliance WHERE is_compliant = FALSE</p>
            <div className="space-y-4">
              {data.policyViolations.map((v) => (
                <div key={v.id} className="p-4 bg-destructive/5 border border-destructive/10 rounded-xl space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[14px] font-bold text-foreground">{v.policyTitle}</span>
                    <span className="text-[12px] font-bold text-destructive">{v.score}%</span>
                  </div>
                  <p className="text-[12px] text-muted-foreground leading-relaxed">{v.reasoning}</p>
                  {flaggedViolations.includes(v.id) ? (
                    feedbackSubmitted.includes(v.id) ? (
                      <div className="text-[11px] text-success font-bold mt-2 pt-2 border-t border-destructive/10">
                        Feedback recorded — queued for model retraining
                      </div>
                    ) : (
                      <div className="flex items-center gap-3 pt-2 border-t border-destructive/10">
                        <span className="text-[11px] text-muted-foreground">Was this verdict correct?</span>
                        <button onClick={() => setFeedbackSubmitted(prev => [...prev, v.id])} className="text-[11px] font-bold text-success hover:underline">Correct</button>
                        <button onClick={() => setFeedbackSubmitted(prev => [...prev, v.id])} className="text-[11px] font-bold text-destructive hover:underline">Incorrect</button>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center justify-end pt-2 border-t border-destructive/10">
                      <button onClick={() => setFlaggedViolations(prev => [...prev, v.id])} className="text-[11px] font-bold text-muted-foreground hover:text-foreground flex items-center gap-1">
                        <Flag className="w-3 h-3" /> Flag as incorrect
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
