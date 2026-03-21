import { useState, useEffect, useRef } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Target, Play, Headphones, Loader2, AlertTriangle } from "lucide-react";
import { getInteractionDetail, getAudioUrl, type InteractionDetail } from "../../services/api";
import { EmotionComparisonPanel } from "../manager/EmotionComparisonPanel.tsx";

export function AgentCallDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshingLLM, setRefreshingLLM] = useState(false);
  const [llmRefreshTick, setLlmRefreshTick] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleJumpTo = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(e => console.error("Playback failed:", e));
    }
  };

  useEffect(() => {
    if (!id) return;
    getInteractionDetail(id, {
      includeLLMTriggers: true,
      llmForceRerun: llmRefreshTick > 0,
    })
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => {
        setLoading(false);
        setRefreshingLLM(false);
      });
  }, [id, llmRefreshTick]);

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
    responseTime: interaction.responseTime,
  };

  const getScoreColor = (score: number) => {
    if (score >= 85) return "#10B981";
    if (score >= 75) return "#3B82F6";
    return "#F59E0B";
  };

  const getEmotionStyle = (emotion: string) => {
    switch (emotion) {
      case "neutral":
        return { bg: "#F1F5F9", text: "#475569", label: "Neutral" };
      case "happy":
        return { bg: "#ECFDF5", text: "#065F46", label: "Happy" };
      case "angry":
        return { bg: "#FEF2F2", text: "#991B1B", label: "Angry" };
      case "frustrated":
        return { bg: "#FFFBEB", text: "#92400E", label: "Frustrated" };
      default:
        return { bg: "#F1F5F9", text: "#475569", label: "Neutral" };
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Back Button */}
      <Link
        to="/agent"
        className="inline-flex items-center gap-2 text-[13px] font-semibold text-[#10B981] hover:underline"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to My Calls
      </Link>

      {/* Call Header Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <div className="flex items-start justify-between mb-6">
          {/* Left: Info */}
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[#9CA3AF] mb-2">
              CALL DETAIL
            </div>
            <h2 className="text-[22px] font-bold text-[#111827] mb-2">
              {callData.date} · {callData.time}
            </h2>
            <div className="text-[13px] text-[#6B7280] mb-2">
              {callData.duration} · {callData.language}
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
                  stroke="#E5E7EB"
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
                <span className="text-[20px] font-normal" style={{ fontFamily: 'var(--font-serif)', color: getScoreColor(callData.overallScore) }}>
                  {callData.overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-[#E5E7EB] mb-4" />

        {/* Audio Player (if available) */}
        {interaction.audioFilePath && (
          <div className="bg-white border border-[#E5E7EB] rounded-2xl p-4 flex items-center gap-4 mb-6">
            <div className="w-10 h-10 bg-[#EFF6FF] text-[#3B82F6] rounded-xl flex items-center justify-center shrink-0">
              <Headphones className="w-5 h-5" />
            </div>
            <div className="flex-1">
              <p className="text-[14px] font-semibold text-[#374151] mb-2">Session Recording</p>
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
          <div className="bg-[#EFF6FF] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Empathy</div>
            <div className="text-[18px] font-semibold text-[#1D4ED8]">
              {callData.empathyScore}%
            </div>
          </div>
          <div className="bg-[#ECFDF5] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Policy</div>
            <div className="text-[18px] font-semibold text-[#065F46]">
              {callData.policyScore}%
            </div>
          </div>
          <div className="bg-[#F5F3FF] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Resolution</div>
            <div className="text-[18px] font-semibold text-[#6D28D9]">
              {callData.resolutionScore}%
            </div>
          </div>
          <div className="bg-[#FFFBEB] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Resp. Time</div>
            <div className="text-[18px] font-semibold text-[#92400E]">
              {callData.responseTime}
            </div>
          </div>
        </div>
      </div>

      {/* Coaching Points Card (only if violations exist) */}
      {policyViolations.length > 0 && (
        <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded-[14px] p-6">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-[15px] h-[15px] text-[#92400E]" />
            <h3 className="text-[14px] font-semibold text-[#92400E]">
              Coaching Points
            </h3>
          </div>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            Areas to focus on — sourced from policy_compliance WHERE is_compliant = FALSE
          </p>

          <div className="space-y-3">
            {policyViolations.map((violation) => (
              <div key={violation.id} className="bg-white border border-[#FDE68A] rounded-[10px] p-3.5">
                <h4 className="text-[14px] font-semibold text-[#111827] mb-2">
                  {violation.policyTitle}
                </h4>
                <p className="text-[12px] text-[#4B5563] mb-2">
                  {violation.reasoning}
                </p>
                <div className="text-[12px] font-semibold text-[#D97706]">
                  Score: {violation.score}% — target 80%+
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Emotion Comparison Panel */}
      {data.emotionComparison && (
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
          <EmotionComparisonPanel data={data.emotionComparison} />
        </div>
      )}

      {llmTriggers && (
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm space-y-4">
          <div>
            <h3 className="text-[16px] font-semibold text-[#111827] mb-1">LLM Coaching Insights</h3>
            <p className="text-[11px] italic text-[#9CA3AF]">Process and policy consistency checks</p>
            <button
              onClick={() => {
                setRefreshingLLM(true);
                setLlmRefreshTick((v) => v + 1);
              }}
              className="mt-2 px-3 py-1.5 border border-[#BFDBFE] bg-[#EFF6FF] text-[#1D4ED8] rounded text-[12px] font-semibold hover:bg-[#DBEAFE] transition-colors"
            >
              {refreshingLLM ? "Refreshing..." : "Refresh LLM"}
            </button>
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

      {/* Transcript Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Transcript
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
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
                    isAgent ? "bg-[#10B981]" : "bg-[#E5E7EB] text-[#6B7280]"
                  }`}
                >
                  {isAgent ? "M" : "C"}
                </div>

                {/* Bubble */}
                <div
                  className={`flex-1 max-w-[80%] p-3 ${
                    isAgent
                      ? "bg-[#ECFDF5] rounded-[0_12px_12px_12px]"
                      : "bg-[#F9FAFB] rounded-[12px_0_12px_12px]"
                  }`}
                  dir="rtl"
                >
                  {/* Header */}
                  <div className={`flex items-center gap-2 mb-1 ${isAgent ? "" : "flex-row-reverse"}`}>
                    <span className="text-[13px] font-semibold text-[#6B7280]">
                      {isAgent ? "Me" : "Customer"}
                    </span>
                    <span className="text-[12px] text-[#9CA3AF]">
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
                  <p className="text-[14px] text-[#374151]">
                    {utterance.text}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Customer Emotion Journey Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Customer Emotion Journey
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          emotion_events — how customer sentiment changed during this call
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
                    ? "bg-[#ECFDF5] border-[#A7F3D0]"
                    : "bg-[#FEF2F2] border-[#FECACA]"
                }`}
              >
                {/* Row 1: Emotion Change + Jump Button */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="px-2.5 py-1 rounded-full text-[11px] font-semibold"
                      style={{ fontFamily: 'var(--font-mono)', backgroundColor: "#F3F4F6", border: "1px solid #E5E7EB" }}
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
                    <span className="text-[#6B7280]">→</span>
                    <span
                      className="px-2 py-0.5 rounded-full text-[11px] font-semibold"
                      style={{ backgroundColor: toStyle.bg, color: toStyle.text }}
                    >
                      {toStyle.label}
                    </span>
                  </div>

                  <button 
                    onClick={() => handleJumpTo(event.jumpToSeconds)}
                    className="flex items-center gap-2 px-4 py-2 bg-[#EFF6FF] text-[#2563EB] border border-[#BFDBFE] rounded-lg text-[12px] font-semibold hover:bg-[#DBEAFE] transition-colors"
                  >
                    <Play className="w-3 h-3" />
                    Jump to {event.timestamp}
                  </button>
                </div>

                {/* Row 2: Justification */}
                <div className="bg-white/60 border-l-4 border-[#10B981] rounded p-3">
                  <p className="text-[12px] italic text-[#6B7280]">
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
