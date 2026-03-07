import { Link, useParams } from "react-router";
import { ArrowLeft, Play, ThumbsUp, ThumbsDown, CheckCircle, XCircle, Flag } from "lucide-react";
import { useState } from "react";
import { mockInteractions, mockUtterances, mockEmotionEvents, mockPolicyViolations } from "../../data/mockData";

export function SessionDetail() {
  const { id } = useParams();
  const interaction = mockInteractions.find((i) => i.id === id) || mockInteractions[0];
  const utterances = mockUtterances.filter((u) => u.interactionId === (id || interaction.id));
  const emotionEvents = mockEmotionEvents.filter((e) => e.interactionId === (id || interaction.id));
  
  const [flaggedEvents, setFlaggedEvents] = useState<string[]>([]);
  const [flaggedViolations, setFlaggedViolations] = useState<string[]>([]);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string[]>([]);

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
        to="/manager/inspector"
        className="inline-flex items-center gap-2 text-[13px] font-semibold text-[#3B82F6] hover:underline"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Session Inspector
      </Link>

      {/* Call Header Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <div className="flex items-start justify-between mb-6">
          {/* Left: Info */}
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[#9CA3AF] mb-2">
              SESSION INSPECTOR
            </div>
            <h2 className="text-[22px] font-bold text-[#111827] mb-2">
              {interaction.agentName}
            </h2>
            <div className="text-[13px] text-[#6B7280] mb-2">
              {interaction.date} 2025 · {interaction.time} · {interaction.duration} · {interaction.language}
            </div>
            {interaction.hasOverlap && (
              <span className="inline-block px-2.5 py-1 bg-[#FFFBEB] text-[#92400E] border border-[#FDE68A] rounded-full text-[11px] font-medium">
                ⚠ Overlap detected
              </span>
            )}
            <div className="text-[11px] text-[#9CA3AF] mt-2" style={{ fontFamily: 'var(--font-mono)' }}>
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
                  stroke="#E5E7EB"
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
                <span className="text-[20px] font-normal" style={{ fontFamily: 'var(--font-serif)', color: getScoreColor(interaction.overallScore) }}>
                  {interaction.overallScore}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-px bg-[#E5E7EB] mb-4" />

        {/* Score Grid */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-[#EFF6FF] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Empathy</div>
            <div className="text-[18px] font-semibold text-[#1D4ED8]">
              {interaction.empathyScore}%
            </div>
          </div>
          <div className="bg-[#ECFDF5] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Policy</div>
            <div className="text-[18px] font-semibold text-[#065F46]">
              {interaction.policyScore}%
            </div>
          </div>
          <div className="bg-[#F5F3FF] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Resolution</div>
            <div className="text-[18px] font-semibold text-[#6D28D9]">
              {interaction.resolutionScore}%
            </div>
          </div>
          <div className="bg-[#FFFBEB] rounded-lg p-3 text-center">
            <div className="text-[11px] text-[#6B7280] mb-1">Resp. Time</div>
            <div className="text-[18px] font-semibold text-[#92400E]">
              {interaction.responseTime}s
            </div>
          </div>
        </div>
      </div>

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
                    isAgent ? "bg-[#2563EB]" : "bg-[#059669]"
                  }`}
                >
                  {isAgent ? "A" : "C"}
                </div>

                {/* Bubble */}
                <div
                  className={`flex-1 max-w-[80%] p-3 ${
                    isAgent
                      ? "bg-[#EFF6FF] rounded-[0_12px_12px_12px]"
                      : "bg-[#ECFDF5] rounded-[12px_0_12px_12px]"
                  }`}
                  dir="rtl"
                >
                  {/* Header */}
                  <div className={`flex items-center gap-2 mb-1 ${isAgent ? "" : "flex-row-reverse"}`}>
                    <span className="text-[13px] font-semibold text-[#6B7280]">
                      {isAgent ? interaction.agentName : "Customer"}
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
                  <p className={`text-[14px] ${isAgent ? "text-[#1E3A5F]" : "text-[#374151]"}`}>
                    {utterance.text}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Emotion Events Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Emotion Events
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          emotion_events — AI-detected emotional shifts with LLM justification
        </p>

        <div className="space-y-4">
          {emotionEvents.map((event) => {
            const isFlagged = flaggedEvents.includes(event.id);
            const hasSubmitted = feedbackSubmitted.includes(event.id);
            const fromStyle = getEmotionStyle(event.fromEmotion);
            const toStyle = getEmotionStyle(event.toEmotion);

            return (
              <div key={event.id} className="border border-[#E5E7EB] rounded-xl p-4 space-y-3">
                {/* Row 1: Emotion Change + Jump Button */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span
                      className="px-2.5 py-1 rounded-full text-[11px] font-semibold"
                      style={{ fontFamily: 'var(--font-mono)', backgroundColor: "#F3F4F6", border: "1px solid #E5E7EB" }}
                    >
                      {event.timestamp}
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
                    <span className="text-[12px] text-[#9CA3AF]">Δ {event.delta}</span>
                    <span className="px-2 py-0.5 bg-[#ECFDF5] text-[#065F46] rounded text-[11px] font-medium">
                      {event.speaker}
                    </span>
                  </div>

                  <button className="flex items-center gap-2 px-4 py-2 bg-[#EFF6FF] text-[#2563EB] border border-[#BFDBFE] rounded-lg text-[12px] font-semibold hover:bg-[#DBEAFE] transition-colors">
                    <Play className="w-3 h-3" />
                    Jump to {event.timestamp}
                  </button>
                </div>

                {/* Row 2: Justification */}
                <div className="bg-[#F9FAFB] border-l-4 border-[#3B82F6] rounded p-3">
                  <p className="text-[12px] italic text-[#6B7280]">
                    {event.justification}
                  </p>
                </div>

                {/* RLHF Feedback */}
                <div className="pt-3 border-t border-[#E5E7EB]">
                  {!hasSubmitted ? (
                    <>
                      {!isFlagged ? (
                        <button
                          onClick={() => setFlaggedEvents([...flaggedEvents, event.id])}
                          className="flex items-center gap-2 px-3 py-1.5 bg-[#F9FAFB] text-[#6B7280] rounded text-[12px] hover:bg-[#F3F4F6] transition-colors"
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
      </div>

      {/* Policy Violations Card */}
      <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-6 shadow-sm">
        <h3 className="text-[16px] font-semibold text-[#111827] mb-1">
          Policy Violations
        </h3>
        <p className="text-[11px] italic text-[#9CA3AF] mb-4">
          policy_compliance WHERE is_compliant = FALSE — only violated policies are shown here
        </p>

        {mockPolicyViolations.length > 0 ? (
          <div className="space-y-4">
            {mockPolicyViolations.map((violation) => {
              const isFlagged = flaggedViolations.includes(violation.id);
              const hasSubmitted = feedbackSubmitted.includes(violation.id);

              return (
                <div key={violation.id} className="bg-[#FEF2F2] border border-[#FECACA] rounded-xl p-4 space-y-3">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-2 flex-1">
                      <XCircle className="w-[15px] h-[15px] text-[#EF4444] flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[14px] font-semibold text-[#111827]">
                            {violation.policyTitle}
                          </span>
                          <span className="px-2 py-0.5 bg-[#FEE2E2] text-[#DC2626] rounded-full text-[11px] font-medium">
                            {violation.category}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="flex-1 h-2 bg-[#F3F4F6] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#FCA5A5] rounded-full"
                              style={{ width: `${violation.score}%` }}
                            />
                          </div>
                          <span className="text-[12px] font-semibold text-[#EF4444]">
                            {violation.score}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* LLM Reasoning */}
                  <p className="text-[12px] text-[#4B5563]">
                    {violation.reasoning}
                  </p>

                  {/* RLHF Feedback */}
                  <div className="pt-3 border-t border-[#FDE68A]">
                    {!hasSubmitted ? (
                      <>
                        {!isFlagged ? (
                          <button
                            onClick={() => setFlaggedViolations([...flaggedViolations, violation.id])}
                            className="flex items-center gap-2 px-3 py-1.5 bg-white text-[#6B7280] border border-[#FDE68A] rounded text-[12px] hover:bg-[#FFFBEB] transition-colors"
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
