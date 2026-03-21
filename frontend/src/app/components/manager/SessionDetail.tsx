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
        <Loader2 className="w-8 h-8 text-[#3B82F6] animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading session...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangleIcon className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-muted-foreground text-sm">Failed to load session</p>
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
        return { bg: "rgba(6, 95, 70, 0.1)", text: "var(--success)", label: "Happy" };
      case "angry":
        return { bg: "rgba(153, 27, 27, 0.1)", text: "var(--destructive)", label: "Angry" };
      case "frustrated":
        return { bg: "#FFFBEB", text: "#92400E", label: "Frustrated" };
      default:
        return { bg: "var(--muted)", text: "var(--muted-foreground)", label: "Neutral" };
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Back Button */}
      <Link
        to="/manager/inspector"
        className="inline-flex items-center gap-2 text-[13px] font-semibold text-[#3B82F6] hover:underline"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Session Inspector
      </Link>

      {/* Call Header Card */}
      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <div className="flex items-start justify-between mb-6">
          {/* Left: Info */}
          <div>
            <div className="text-label mb-2">
              SESSION INSPECTOR
            </div>
            <h2 className="text-[22px] font-bold text-foreground mb-2">
              {interaction.agentName}
            </h2>
            <div className="text-[13px] text-muted-foreground mb-2">
              {interaction.date} 2025 · {interaction.time} · {interaction.duration} · {interaction.language}
            </div>
            {interaction.hasOverlap && (
              <span className="inline-block px-2.5 py-1 bg-warning/10 text-warning border border-warning/20 rounded-full text-[11px] font-medium">
                ⚠ Overlap detected
              </span>
            )}
            <div className="text-label-foreground text-[11px] mt-2" style={{ fontFamily: 'var(--font-mono)' }}>
              {interaction.id} · completed
            </div>
          </div>

          {/* Right: Score Ring */}
          <div className="flex flex-col items-center">
            <div className="relative w-[90px] h-[90px]">
              <svg className="w-full h-full -rotate-90">
                <circle
                  cx="45"
                  cy="45"
                  r="38"
                  fill="none"
                  stroke="var(--border)"
                  strokeWidth="7"
                />
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
                <span className="text-[20px] font-normal" style={{ fontFamily: 'var(--font-serif)', color: `var(--${interaction.overallScore >= 85 ? 'success' : interaction.overallScore >= 75 ? 'primary' : 'destructive'})` }}>
                  {interaction.overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-[#E5E7EB] mb-4" />

        {/* Audio Player (if available) */}
      {interaction.audioFilePath && (
        <div className="bg-muted/30 border border-border rounded-2xl p-4 flex items-center gap-4">
          <div className="w-10 h-10 bg-primary/10 text-primary rounded-xl flex items-center justify-center shrink-0">
            <Headphones className="w-5 h-5" />
          </div>
          <div className="flex-1">
            <p className="text-[14px] font-semibold text-foreground mb-2">Session Recording</p>
            <audio 
              ref={audioRef}
              controls 
              className="w-full h-8" 
              src={getAudioUrl(interaction.id)}
              preload="metadata"
            >
              Your browser does not support the audio element.
            </audio>
          </div>
        </div>
      )}

      {/* Score Grid */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-primary/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Empathy</div>
            <div className="text-[18px] font-semibold text-primary">
              {interaction.empathyScore}%
            </div>
          </div>
          <div className="bg-success/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Policy</div>
            <div className="text-[18px] font-semibold text-success">
              {interaction.policyScore}%
            </div>
          </div>
          <div className="bg-accent/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Resolution</div>
            <div className="text-[18px] font-semibold text-foreground">
              {interaction.resolutionScore}%
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Resp. Time</div>
            <div className="text-[18px] font-semibold text-foreground">
              {interaction.responseTime}s
            </div>
          </div>
        </div>
      </div>

      {/* Transcript Card */}
      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <h3 className="text-[16px] font-semibold text-foreground mb-1">
          Transcript
        </h3>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          utterances ordered by sequence_index
        </p>

        <div className="space-y-4 max-h-[280px] overflow-y-auto">
          {utterances.map((utterance) => {
            const isAgent = utterance.speaker === "agent";
            const emotionStyle = getEmotionStyle(utterance.emotion);

            return (
              <div
                key={utterance.id}
                className={`flex gap-3 ${isAgent ? "" : "flex-row-reverse"}`}
              >
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0 ${
                    isAgent ? "bg-[#2563EB]" : "bg-[#059669]"
                  }`}
                >
                  {isAgent ? "A" : "C"}
                </div>

                {/* Bubble */}
                  <div
                    className={`flex-1 max-w-[80%] p-3 ${
                      isAgent
                        ? "bg-primary/10 rounded-[0_12px_12px_12px]"
                        : "bg-success/10 rounded-[12px_0_12px_12px]"
                    }`}
                    dir="rtl"
                  >
                  {/* Header */}
                  <div className={`flex items-center gap-2 mb-1 ${isAgent ? "" : "flex-row-reverse"}`}>
                    <span className="text-[13px] font-semibold text-muted-foreground">
                      {isAgent ? interaction.agentName : "Customer"}
                    </span>
                    <span className="text-[12px] text-label-foreground">
                      {utterance.timestamp}
                    </span>
                    <span
                      className="px-2 py-0.5 rounded-full text-[11px] font-semibold"
                      style={{ backgroundColor: emotionStyle.bg, color: emotionStyle.text }}
                    >
                      {emotionStyle.label} {Math.round(utterance.confidence * 100)}%
                    </span>
                  </div>

                  {/* Text */}
                  <p className="text-[14px] text-foreground font-medium leading-relaxed">
                    {utterance.text}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Emotion Events Card */}
      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <h3 className="text-[16px] font-semibold text-foreground mb-1">
          Emotion Events
        </h3>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          emotion_events — AI-detected emotional shifts with LLM justification
        </p>

        <div className="space-y-4">
          {emotionEvents.map((event) => {
            const isFlagged = flaggedEvents.includes(event.id);
            const hasSubmitted = feedbackSubmitted.includes(event.id);
            const fromStyle = getEmotionStyle(event.fromEmotion);
            const toStyle = getEmotionStyle(event.toEmotion);

            return (
              <div key={event.id} className="border border-border rounded-xl p-4 space-y-3 bg-muted/10">
                {/* Row 1: Emotion Change + Jump Button */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="px-2.5 py-1 rounded-full text-[11px] font-bold"
                      style={{ fontFamily: 'var(--font-mono)', backgroundColor: 'var(--muted)', color: 'var(--muted-foreground)', border: '1px solid var(--border)' }}
                    >
                      {event.timestamp}
                    </span>
                    <span
                      className="px-2 py-0.5 rounded-full text-[11px] font-semibold"
                      style={{ backgroundColor: fromStyle.bg, color: fromStyle.text }}
                    >
                      {fromStyle.label}
                    </span>
                    <span className="text-muted-foreground/60">→</span>
                    <span
                      className="px-2 py-0.5 rounded-full text-[11px] font-semibold"
                      style={{ backgroundColor: toStyle.bg, color: toStyle.text }}
                    >
                      {toStyle.label}
                    </span>
                    <span className="text-[12px] text-muted-foreground/80 font-medium">Δ {event.delta}</span>
                    <span className="px-2 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded text-[11px] font-bold uppercase tracking-wider">
                      {event.speaker}
                    </span>
                  </div>

                  <button 
                    onClick={() => handleJumpTo(event.jumpToSeconds)}
                    className="flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary border border-primary/30 rounded-lg text-[12px] font-bold hover:bg-primary/20 transition-all shadow-sm"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    Jump to {event.timestamp}
                  </button>
                </div>

                {/* Row 2: Justification */}
                <div className="bg-background border-l-4 border-primary rounded p-3 shadow-inner">
                  <p className="text-[12px] italic text-muted-foreground leading-relaxed">
                    {event.justification}
                  </p>
                </div>

                {/* RLHF Feedback */}
                <div className="pt-3 border-t border-border">
                  {!hasSubmitted ? (
                    <>
                      {!isFlagged ? (
                        <button
                          onClick={() => setFlaggedEvents([...flaggedEvents, event.id])}
                          className="flex items-center gap-2 px-3 py-1.5 bg-muted/30 text-muted-foreground border border-border rounded text-[12px] hover:bg-muted font-medium transition-colors"
                        >
                          <Flag className="w-3.5 h-3.5" />
                          Flag as incorrect
                        </button>
                      ) : (
                        <div className="space-y-2">
                          <div className="text-[11px] text-[#9CA3AF] mb-2">
                            Was this detection accurate?
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => setFeedbackSubmitted([...feedbackSubmitted, event.id])}
                              className="flex items-center gap-2 px-3 py-2 bg-[#ECFDF5] text-[#065F46] rounded-lg text-[12px] font-medium hover:bg-[#D1FAE5] transition-colors"
                            >
                              <ThumbsUp className="w-3.5 h-3.5" />
                              Accurate
                            </button>
                            <button
                              onClick={() => setFeedbackSubmitted([...feedbackSubmitted, event.id])}
                              className="flex items-center gap-2 px-3 py-2 bg-[#FEF2F2] text-[#991B1B] rounded-lg text-[12px] font-medium hover:bg-[#FEE2E2] transition-colors"
                            >
                              <ThumbsDown className="w-3.5 h-3.5" />
                              Incorrect
                            </button>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-[13px] text-success font-semibold flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      Feedback recorded — queued for retraining
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Policy Violations Card */}
      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <h3 className="text-[16px] font-semibold text-foreground mb-1">
          Policy Violations
        </h3>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          policy_compliance WHERE is_compliant = FALSE — only violated policies are shown here
        </p>

        {data.policyViolations.length > 0 ? (
          <div className="space-y-4">
            {data.policyViolations.map((violation) => {
              const isFlagged = flaggedViolations.includes(violation.id);
              const hasSubmitted = feedbackSubmitted.includes(violation.id);

              return (
                <div key={violation.id} className="bg-destructive/5 border border-destructive/20 rounded-xl p-4 space-y-3">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2 flex-1">
                      <XCircle className="w-[15px] h-[15px] text-[#EF4444] flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[14px] font-bold text-foreground">
                            {violation.policyTitle}
                          </span>
                          <span className="px-2 py-0.5 bg-destructive/10 text-destructive border border-destructive/20 rounded-full text-[10px] font-bold uppercase tracking-wider">
                            {violation.category}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="flex-1 h-2 bg-muted/30 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#FCA5A5] rounded-full"
                              style={{ width: `${violation.score}%` }}
                            />
                          </div>
                          <span className="text-[12px] font-bold text-destructive">
                            {violation.score}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* LLM Reasoning */}
                  <p className="text-[12px] text-muted-foreground/80 leading-relaxed">
                    {violation.reasoning}
                  </p>

                  {/* RLHF Feedback */}
                  <div className="pt-3 border-t border-[#FDE68A]">
                    {!hasSubmitted ? (
                      <>
                        {!isFlagged ? (
                          <button
                            onClick={() => setFlaggedViolations([...flaggedViolations, violation.id])}
                            className="flex items-center gap-2 px-3 py-1.5 bg-card text-muted-foreground border border-border rounded text-[12px] hover:bg-muted transition-colors"
                          >
                            <Flag className="w-3.5 h-3.5" />
                            Flag as incorrect
                          </button>
                        ) : (
                          <div className="space-y-2">
                            <div className="text-[11px] text-[#9CA3AF] mb-2">
                              Was this verdict correct?
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => setFeedbackSubmitted([...feedbackSubmitted, violation.id])}
                                className="flex items-center gap-2 px-3 py-2 bg-[#ECFDF5] text-[#065F46] rounded-lg text-[12px] font-medium hover:bg-[#D1FAE5] transition-colors"
                              >
                                <ThumbsUp className="w-3.5 h-3.5" />
                                Correct
                              </button>
                              <button
                                onClick={() => setFeedbackSubmitted([...feedbackSubmitted, violation.id])}
                                className="flex items-center gap-2 px-3 py-2 bg-[#FEF2F2] text-[#991B1B] rounded-lg text-[12px] font-medium hover:bg-[#FEE2E2] transition-colors"
                              >
                                <ThumbsDown className="w-3.5 h-3.5" />
                                Incorrect
                              </button>
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-[13px] text-[#059669] flex items-center gap-2">
                        <CheckCircle className="w-4 h-4" />
                        Feedback recorded — queued for model retraining
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-[#10B981] py-4">
            <CheckCircle className="w-5 h-5" />
            <span className="text-[14px] font-medium">
              All policies passed for this call
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
