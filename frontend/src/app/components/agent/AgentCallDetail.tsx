import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Target, Play, Headphones, Loader2, AlertTriangle } from "lucide-react";
import { getInteractionDetail, getAudioUrl, type InteractionDetail } from "../../services/api";
import { EmotionComparisonPanel } from "../manager/EmotionComparisonPanel.tsx";
import { formatResponseTime } from "../../utils/interactionFormat";

export function AgentCallDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleJumpTo = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch((e) => console.error("Playback failed:", e));
    }
  };

  useEffect(() => {
    if (!id) return;
    getInteractionDetail(id, {
      includeLLMTriggers: true,
    })
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => {
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-[#10B981] animate-spin" />
        <span className="ml-3 text-[#6B7280] text-sm">Loading call details...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-[#6B7280] text-sm">Failed to load call details</p>
          <p className="text-[#9CA3AF] text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const interaction = data.interaction;
  const utterances = data.utterances;
  const emotionEvents = data.emotionEvents;
  const policyViolations = data.policyViolations;
  const llmTriggers = data.llmTriggers;

  const callData = {
    date: interaction.date,
    time: interaction.time,
    duration: interaction.duration,
    language: interaction.language,
    overallScore: interaction.overallScore,
    empathyScore: interaction.empathyScore,
    policyScore: interaction.policyScore,
    resolutionScore: interaction.resolutionScore,
    responseTime: formatResponseTime(interaction.responseTime),
  };

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
        return { bg: "#FFFBEB", text: "#92400E", label: "Frustrated" };
      default:
        return { bg: "var(--muted)", text: "var(--muted-foreground)", label: "Neutral" };
    }
  };

  return (
    <div className="p-6 space-y-6">
      <Link
        to="/agent/calls"
        className="inline-flex items-center gap-2 text-[13px] font-semibold text-[#10B981] hover:underline"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to My Calls
      </Link>

      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="text-label mb-2">
              CALL DETAIL
            </div>
            <h2 className="text-[22px] font-bold text-foreground mb-2">
              {callData.date} • {callData.time}
            </h2>
            <div className="text-[13px] text-muted-foreground mb-2">
              {callData.duration} • {callData.language}
            </div>
          </div>

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
                  stroke={getScoreColor(callData.overallScore)}
                  strokeWidth="7"
                  strokeLinecap="round"
                  strokeDasharray={`${(callData.overallScore / 100) * 238.76} 238.76`}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[20px] font-normal" style={{ fontFamily: "var(--font-serif)", color: getScoreColor(callData.overallScore) }}>
                  {callData.overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="h-px bg-border mb-4" />

        {interaction.audioFilePath && (
          <div className="bg-muted/30 border border-border rounded-2xl p-4 flex items-center gap-4 mb-6 transition-all shadow-inner">
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

        <div className="grid grid-cols-4 gap-4">
          <div className="bg-primary/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Empathy</div>
            <div className="text-[18px] font-semibold text-primary">
              {callData.empathyScore}%
            </div>
          </div>
          <div className="bg-success/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Policy</div>
            <div className="text-[18px] font-semibold text-success">
              {callData.policyScore}%
            </div>
          </div>
          <div className="bg-accent/10 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Resolution</div>
            <div className="text-[18px] font-semibold text-foreground">
              {callData.resolutionScore}%
            </div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-[11px] text-muted-foreground mb-1">Resp. Time</div>
            <div className="text-[18px] font-semibold text-foreground">
              {callData.responseTime}
            </div>
          </div>
        </div>
      </div>

      {policyViolations.length > 0 && (
        <div className="bg-warning/5 border border-warning/20 rounded-[14px] p-6 transition-all">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-[15px] h-[15px] text-warning" />
            <h3 className="text-[14px] font-semibold text-warning">
              Coaching Points
            </h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            Areas to focus on from saved policy compliance findings.
          </p>

          <div className="space-y-3">
            {policyViolations.map((violation) => (
              <div key={violation.id} className="bg-card border border-warning/20 rounded-[10px] p-3.5">
                <h4 className="text-[14px] font-semibold text-foreground mb-2">
                  {violation.policyTitle}
                </h4>
                <p className="text-[12px] text-muted-foreground mb-2">
                  {violation.reasoning}
                </p>
                <div className="text-[12px] font-semibold text-warning">
                  Score: {violation.score}% • target 80%+
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.emotionComparison && (
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
          <EmotionComparisonPanel data={data.emotionComparison} />
        </div>
      )}

      {llmTriggers && (
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm space-y-4">
          <div>
            <h3 className="text-[16px] font-semibold text-[#111827] mb-1">LLM Coaching Insights</h3>
            <p className="text-[11px] italic text-[#9CA3AF]">
              Cached coaching insights saved during processing. Use call reprocessing to regenerate them.
            </p>
          </div>

          {!llmTriggers.available ? (
            <div className="rounded-lg border border-[#FECACA] bg-[#FEF2F2] p-3 text-[12px] text-[#991B1B]">
              LLM coaching insights unavailable.
              {llmTriggers.error ? ` ${llmTriggers.error}` : ""}
            </div>
          ) : (
            <>
              {llmTriggers.processAdherence && (
                <div className="rounded-lg border border-[#E5E7EB] p-4 text-[12px] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#111827]">Process Status</span>
                    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${llmTriggers.processAdherence.isResolved ? "bg-[#ECFDF5] text-[#065F46]" : "bg-[#FEF2F2] text-[#991B1B]"}`}>
                      {llmTriggers.processAdherence.isResolved ? "Resolved" : "Needs follow-up"}
                    </span>
                  </div>
                  <p className="text-[#374151]"><span className="text-[#6B7280]">Topic:</span> {llmTriggers.processAdherence.detectedTopic}</p>
                  <p className="text-[#374151]"><span className="text-[#6B7280]">Efficiency:</span> {llmTriggers.processAdherence.efficiencyScore}/10</p>
                  {llmTriggers.processAdherence.missingSopSteps.length > 0 && (
                    <ul className="list-disc ml-5 text-[#374151] space-y-1">
                      {llmTriggers.processAdherence.missingSopSteps.map((step, idx) => (
                        <li key={`agent-missing-step-${idx}`}>{step}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {llmTriggers.nliPolicy && (
                <div className="rounded-lg border border-[#E5E7EB] p-4 text-[12px] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[#111827]">Policy Consistency</span>
                    <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-[#EFF6FF] text-[#1D4ED8]">
                      {llmTriggers.nliPolicy.nliCategory}
                    </span>
                  </div>
                  <p className="text-[#374151]">{llmTriggers.nliPolicy.justification}</p>
                </div>
              )}
            </>
          )}
        </div>
      )}

      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <h3 className="text-[16px] font-semibold text-foreground mb-1">
          Transcript
        </h3>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          Utterances ordered by sequence index.
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
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-primary-foreground text-xs font-bold flex-shrink-0 ${
                    isAgent ? "bg-primary" : "bg-muted text-muted-foreground"
                  }`}
                >
                  {isAgent ? "A" : "C"}
                </div>

                <div
                  className={`flex-1 max-w-[80%] p-3 ${
                    isAgent
                      ? "bg-success/10 rounded-[0_12px_12px_12px]"
                      : "bg-muted/50 rounded-[12px_0_12px_12px]"
                  }`}
                >
                  <div className={`flex items-center gap-2 mb-1 ${isAgent ? "" : "flex-row-reverse"}`}>
                    <span className="text-[13px] font-semibold text-muted-foreground">
                      {isAgent ? "Me" : "Customer"}
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

                  <p className="text-[14px] text-foreground">
                    {utterance.text}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-card rounded-[14px] border border-border p-6 transition-all">
        <h3 className="text-[16px] font-semibold text-foreground mb-1">
          Customer Emotion Journey
        </h3>
        <p className="text-[11px] italic text-muted-foreground mb-4">
          Emotion changes across the call timeline.
        </p>

        <div className="space-y-4">
          {emotionEvents.map((event) => {
            const fromStyle = getEmotionStyle(event.fromEmotion);
            const toStyle = getEmotionStyle(event.toEmotion);
            const isPositive = event.toEmotion === "happy";

            return (
              <div
                key={event.id}
                className={`border rounded-xl p-4 space-y-3 ${
                  isPositive
                    ? "bg-success/5 border-success/30"
                    : "bg-destructive/5 border-destructive/20"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="px-2.5 py-1 rounded-full text-[11px] font-bold"
                      style={{ fontFamily: "var(--font-mono)", backgroundColor: "var(--muted)", color: "var(--muted-foreground)", border: "1px solid var(--border)" }}
                    >
                      {event.timestamp}
                    </span>
                    <span className="text-[13px] text-[#6B7280] font-medium">
                      Customer mood:
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
                  </div>

                  <button
                    onClick={() => handleJumpTo(event.jumpToSeconds)}
                    className="flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary border border-primary/30 rounded-lg text-[12px] font-bold hover:bg-primary/20 transition-all shadow-sm"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    Jump to {event.timestamp}
                  </button>
                </div>

                <div className="bg-background border-l-4 border-success rounded p-3 shadow-inner">
                  <p className="text-[12px] italic text-muted-foreground leading-relaxed">
                    {event.justification}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
