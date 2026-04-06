import { Link, useParams } from "react-router";
import { ArrowLeft, Play, Headphones, Loader2, AlertTriangle as AlertTriangleIcon } from "lucide-react";
import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { getInteractionDetail, getAudioUrl, reprocessInteraction, type InteractionDetail } from "../../services/api";
import { EvidenceAnchoredExplainabilityPanel } from "./EvidenceAnchoredExplainabilityPanel";

const emotionTheme: Record<string, { label: string; color: string; surface: string }> = {
  neutral: { label: "Neutral", color: "#64748B", surface: "rgba(100, 116, 139, 0.16)" },
  happy: { label: "Happy", color: "#16A34A", surface: "rgba(22, 163, 74, 0.18)" },
  grateful: { label: "Grateful", color: "#059669", surface: "rgba(5, 150, 105, 0.16)" },
  interested: { label: "Interested", color: "#0EA5E9", surface: "rgba(14, 165, 233, 0.16)" },
  calmer: { label: "Calmer", color: "#14B8A6", surface: "rgba(20, 184, 166, 0.16)" },
  curious: { label: "Curious", color: "#8B5CF6", surface: "rgba(139, 92, 246, 0.18)" },
  surprised: { label: "Surprised", color: "#F59E0B", surface: "rgba(245, 158, 11, 0.18)" },
  professional: { label: "Professional", color: "#4F46E5", surface: "rgba(79, 70, 229, 0.16)" },
  informative: { label: "Informative", color: "#2563EB", surface: "rgba(37, 99, 235, 0.16)" },
  empathetic: { label: "Empathetic", color: "#7C3AED", surface: "rgba(124, 58, 237, 0.16)" },
  helpful: { label: "Helpful", color: "#0891B2", surface: "rgba(8, 145, 178, 0.16)" },
  warm: { label: "Warm", color: "#EA580C", surface: "rgba(234, 88, 12, 0.16)" },
  frustrated: { label: "Frustrated", color: "#DC2626", surface: "rgba(220, 38, 38, 0.18)" },
  angry: { label: "Angry", color: "#B91C1C", surface: "rgba(185, 28, 28, 0.18)" },
  sad: { label: "Sad", color: "#475569", surface: "rgba(71, 85, 105, 0.18)" },
  unknown: { label: "Unknown", color: "#94A3B8", surface: "rgba(148, 163, 184, 0.16)" },
};

function normalizeEmotionLabel(value?: string): string {
  const key = (value || "neutral").trim().toLowerCase();
  if (emotionTheme[key]) {
    return key;
  }
  if (key.includes("fear")) {
    return "frustrated";
  }
  if (key.includes("disgust")) {
    return "frustrated";
  }
  if (key.includes("frustr")) {
    return "frustrated";
  }
  if (key.includes("calm")) {
    return "calmer";
  }
  if (key.includes("surpris")) {
    return "surprised";
  }
  if (key.includes("other") || key.includes("unk")) {
    return "unknown";
  }
  return "neutral";
}

function getEmotionMeta(value?: string) {
  return emotionTheme[normalizeEmotionLabel(value)] || emotionTheme.neutral;
}

function normalizeConfidence(value?: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0.5;
  }
  return Math.max(0, Math.min(1, numeric));
}

const emotionValence: Record<string, number> = {
  happy: 0.95,
  grateful: 0.82,
  interested: 0.35,
  calmer: 0.2,
  curious: 0.22,
  surprised: 0.1,
  professional: 0.12,
  informative: 0.08,
  empathetic: 0.25,
  helpful: 0.3,
  warm: 0.35,
  neutral: 0,
  unknown: 0,
  frustrated: -0.72,
  angry: -0.95,
  sad: -0.58,
};

function emotionSignal(value?: string, confidence?: number): number {
  const label = normalizeEmotionLabel(value);
  const valence = emotionValence[label] ?? 0;
  const normalizedConfidence = normalizeConfidence(confidence);
  // Keep direction from emotion while confidence controls distance from neutral.
  const strength = 0.55 + (normalizedConfidence * 0.45);
  return valence * strength;
}

function formatClock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function buildTrendCurve(points: Array<{ x: number; y: number }>): string {
  if (!points.length) {
    return "";
  }
  if (points.length === 1) {
    return `M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`;
  }
  let path = `M${points[0].x.toFixed(2)},${points[0].y.toFixed(2)}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];
    const cx = (p1.x + p2.x) / 2;
    path += ` C${cx.toFixed(2)},${p1.y.toFixed(2)} ${cx.toFixed(2)},${p2.y.toFixed(2)} ${p2.x.toFixed(2)},${p2.y.toFixed(2)}`;
  }
  return path;
}

function isElementFullyVisible(container: HTMLDivElement, element: HTMLElement): boolean {
  const cTop = container.scrollTop;
  const cBottom = cTop + container.clientHeight;
  const eTop = element.offsetTop;
  const eBottom = eTop + element.offsetHeight;
  return eTop >= cTop && eBottom <= cBottom;
}

export function SessionDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [reprocessing, setReprocessing] = useState(false);
  const [activeUtteranceId, setActiveUtteranceId] = useState<string | null>(null);
  const [windowedUtteranceId, setWindowedUtteranceId] = useState<string | null>(null);
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const activeUtteranceIdRef = useRef<string | null>(null);
  const lastAutoScrollIdRef = useRef<string | null>(null);

  const findUtteranceByTime = (time: number) => {
    for (let idx = 0; idx < utterances.length; idx += 1) {
      const current = utterances[idx];
      const next = utterances[idx + 1];
      const start = current.startTime || 0;
      const end = current.endTime || next?.startTime || start + 4;
      if (time >= start && time <= end) {
        return current;
      }
    }
    return null;
  };

  const syncTranscriptToAudio = (time: number, shouldAutoScroll: boolean) => {
    const currentUtterance = findUtteranceByTime(time);
    if (!currentUtterance || currentUtterance.id === activeUtteranceIdRef.current) {
      return;
    }

    activeUtteranceIdRef.current = currentUtterance.id;
    setActiveUtteranceId(currentUtterance.id);

    if (!shouldAutoScroll) {
      return;
    }

    const utteranceEl = document.getElementById(`utterance-${currentUtterance.id}`);
    if (!utteranceEl || !transcriptRef.current) {
      return;
    }

    if (currentUtterance.id === lastAutoScrollIdRef.current) {
      return;
    }

    if (!isElementFullyVisible(transcriptRef.current, utteranceEl)) {
      utteranceEl.scrollIntoView({ behavior: "smooth", block: "center" });
      lastAutoScrollIdRef.current = currentUtterance.id;
    }
  };

  const utterances = useMemo(() => (Array.isArray(data?.utterances) ? data.utterances : []), [data]);
  const emotionEvents = useMemo(() => (Array.isArray(data?.emotionEvents) ? data.emotionEvents : []), [data]);

  const windowedUtterance = useMemo(() => {
    if (!windowedUtteranceId) {
      return null;
    }
    const idx = utterances.findIndex((u) => u.id === windowedUtteranceId);
    if (idx < 0) {
      return null;
    }
    const start = Math.max(0, idx - 2);
    const end = Math.min(utterances.length, idx + 3);
    return {
      selected: utterances[idx],
      context: utterances.slice(start, end),
    };
  }, [utterances, windowedUtteranceId]);

  const totalDurationSeconds = useMemo(() => {
    const maxUtteranceEnd = utterances.reduce((max, u) => Math.max(max, u.endTime || u.startTime || 0), 0);
    if (maxUtteranceEnd > 0) {
      return maxUtteranceEnd;
    }
    const chunks = (data?.interaction.duration || "0:00").split(":");
    if (chunks.length === 2) {
      const mins = Number(chunks[0]) || 0;
      const secs = Number(chunks[1]) || 0;
      return mins * 60 + secs;
    }
    return 1;
  }, [data, utterances]);

  const emotionTriggers = useMemo(() => {
    if (data?.emotionTriggers) {
      return data.emotionTriggers;
    }
    if (data?.llmTriggers) {
      return {
        available: data.llmTriggers.available,
        error: data.llmTriggers.error,
        orgFilter: data.llmTriggers.orgFilter,
        forcedRerun: data.llmTriggers.forcedRerun,
        interactionId: data.llmTriggers.interactionId,
        emotionShift: data.llmTriggers.emotionShift,
        derived: data.llmTriggers.derived,
      };
    }
    return null;
  }, [data]);

  const interactionStatus = (data?.interaction.status || "").toLowerCase();
  const isProcessing = ["pending", "processing"].includes(interactionStatus);

  const refreshDetail = useCallback(async (initialLoad = false) => {
    if (!id) {
      return;
    }

    if (initialLoad) {
      setLoading(true);
    }

    try {
      const detail = await getInteractionDetail(id, { includeLLMTriggers: true });
      setData(detail);
      setLoadError(null);
    } catch (err) {
      if (initialLoad) {
        setLoadError(err instanceof Error ? err.message : "Failed to load session");
      }
    } finally {
      if (initialLoad) {
        setLoading(false);
      }
    }
  }, [id]);

  const handleReprocess = async () => {
    if (!id || reprocessing) {
      return;
    }

    setReprocessing(true);
    setActionError(null);
    try {
      await reprocessInteraction(id);
      await refreshDetail(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reprocess interaction";
      if (message.includes("409")) {
        try {
          await reprocessInteraction(id, { force: true });
          await refreshDetail(false);
          return;
        } catch {
          setActionError("This interaction is already being processed. Wait a few seconds and try again.");
        }
      } else {
        setActionError(message);
      }
    } finally {
      setReprocessing(false);
    }
  };

  const ragCompliance = useMemo(() => {
    if (data?.ragCompliance) {
      return data.ragCompliance;
    }
    if (data?.llmTriggers) {
      return {
        available: data.llmTriggers.available,
        error: data.llmTriggers.error,
        orgFilter: data.llmTriggers.orgFilter,
        forcedRerun: data.llmTriggers.forcedRerun,
        interactionId: data.llmTriggers.interactionId,
        processAdherence: data.llmTriggers.processAdherence,
        nliPolicy: data.llmTriggers.nliPolicy,
      };
    }
    return null;
  }, [data]);

  const explainability = useMemo(() => {
    if (data?.llmTriggers?.explainability) {
      return data.llmTriggers.explainability;
    }
    return {
      triggerAttributions: [
        ...(emotionTriggers?.explainability?.triggerAttributions || []),
        ...(ragCompliance?.explainability?.triggerAttributions || []),
      ],
      claimProvenance: ragCompliance?.explainability?.claimProvenance || [],
    };
  }, [data, emotionTriggers, ragCompliance]);

  const handleJumpTo = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(e => console.error("Playback failed:", e));
    }
  };

  useEffect(() => {
    void refreshDetail(true);
  }, [refreshDetail]);

  useEffect(() => {
    if (!id || !data || !isProcessing) {
      return;
    }

    const interval = window.setInterval(() => {
      void refreshDetail(false);
    }, 4000);

    return () => window.clearInterval(interval);
  }, [data, id, isProcessing, refreshDetail]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !data) return;

    const handleTimeUpdate = () => {
      const currentTime = audio.currentTime;
      setCurrentTimeSeconds(currentTime);
      syncTranscriptToAudio(currentTime, true);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleSeeked = () => {
      setCurrentTimeSeconds(audio.currentTime);
      syncTranscriptToAudio(audio.currentTime, true);
    };

    const handleLoadedMetadata = () => {
      setCurrentTimeSeconds(audio.currentTime || 0);
      syncTranscriptToAudio(audio.currentTime || 0, false);
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("seeked", handleSeeked);
    audio.addEventListener("loadedmetadata", handleLoadedMetadata);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("seeked", handleSeeked);
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [data, utterances]);

  const speakerEmotionSummary = useMemo(() => {
    const agent = new Set<string>();
    const client = new Set<string>();
    utterances.forEach((u) => {
      const candidate = normalizeEmotionLabel(u.fusedEmotion || u.emotion);
      if (u.speaker === "agent") {
        agent.add(candidate);
      } else {
        client.add(candidate);
      }
    });
    return {
      agent: [...agent].slice(0, 8),
      client: [...client].slice(0, 8),
    };
  }, [utterances]);

  const emotionGraphData = useMemo(() => {
    const width = 960;
    const height = 220;
    const padding = 24;
    const plotLeftPct = 4;
    const plotWidthPct = 92;
    const agentPoints: Array<{ x: number; y: number }> = [];
    const clientPoints: Array<{ x: number; y: number }> = [];
    
    // Map emotion signal -1..1 to Y coordinates (1 is top/positive, -1 is bottom/negative).
    const mapSignalToY = (signal: number) => {
      const range = height - padding * 2;
      const bounded = Math.max(-1, Math.min(1, signal));
      return height - padding - (((bounded + 1) / 2) * range);
    };

    const markers = utterances.map((u) => {
      const start = Math.max(0, u.startTime || 0);
      const timelinePct = totalDurationSeconds > 0 ? (start / totalDurationSeconds) * 100 : 0;
      const leftPct = plotLeftPct + (timelinePct / 100) * plotWidthPct;
      const emotion = getEmotionMeta(u.fusedEmotion || u.emotion);
      const confidence = normalizeConfidence(u.fusedConfidence ?? u.confidence ?? u.textConfidence);
      const signal = emotionSignal(u.fusedEmotion || u.emotion, confidence);
      const x = (leftPct / 100) * width;
      const y = mapSignalToY(signal);

      if (u.speaker === "agent") {
        agentPoints.push({ x, y });
      } else {
        clientPoints.push({ x, y });
      }
      return {
        id: u.id,
        start,
        leftPct,
        yPct: (y / height) * 100,
        speaker: u.speaker,
        color: emotion.color,
        label: emotion.label,
        confidence,
        signal,
        timestamp: u.timestamp,
      };
    });

    const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1].map((ratio) => {
      const seconds = totalDurationSeconds * ratio;
      return {
        label: formatClock(seconds),
        leftPct: plotLeftPct + ratio * plotWidthPct,
      };
    });

    return {
      width,
      height,
      markers,
      ticks,
      plotLeftPct,
      plotWidthPct,
      agentPath: buildTrendCurve(agentPoints),
      clientPath: buildTrendCurve(clientPoints),
    };
  }, [utterances, totalDurationSeconds]);

  const playbackTimelinePct = Math.max(0, Math.min(100, (currentTimeSeconds / totalDurationSeconds) * 100));
  const playbackLineLeft = emotionGraphData.plotLeftPct + (playbackTimelinePct / 100) * emotionGraphData.plotWidthPct;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading session...</span>
      </div>
    );
  }

  if (loadError || !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangleIcon className="w-10 h-10 text-warning mx-auto mb-3" />
          <p className="text-foreground text-sm">Failed to load session</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{loadError}</p>
        </div>
      </div>
    );
  }

  const interaction = data.interaction;
  const policyCards = ragCompliance?.policyViolations?.length ? ragCompliance.policyViolations : data.policyViolations;
  const missingSopSteps = Array.isArray(ragCompliance?.processAdherence?.missingSopSteps)
    ? ragCompliance.processAdherence.missingSopSteps
    : [];

  const getScoreColor = (score: number) => {
    if (score >= 85) return "var(--success)";
    if (score >= 75) return "var(--primary)";
    return "var(--destructive)";
  };

  const responseTimeText = interaction.responseTime.endsWith("s") ? interaction.responseTime : `${interaction.responseTime}s`;
  const progressPercent = Math.max(0, Math.min(100, (currentTimeSeconds / totalDurationSeconds) * 100));
  const statusBadgeClass = isProcessing
    ? "border-amber-200 text-amber-700 bg-amber-50"
    : interactionStatus === "failed"
      ? "border-destructive/30 text-destructive bg-destructive/10"
      : interactionStatus === "completed"
        ? "border-emerald-200 text-emerald-700 bg-emerald-50"
        : "border-slate-200 text-slate-600 bg-slate-100";
  const explainabilityCardCount =
    (explainability?.triggerAttributions?.length || 0) + (explainability?.claimProvenance?.length || 0);
  const reviewSignals = [
    { label: "Utterances", value: `${utterances.length}` },
    { label: "Evidence cards", value: `${explainabilityCardCount}` },
    { label: "Policy flags", value: `${policyCards.length}` },
  ];
  const processingSteps = [
    { label: "Transcription", done: !isProcessing || utterances.length > 0 },
    { label: "Explainability", done: !isProcessing || explainabilityCardCount > 0 },
    { label: "Compliance", done: !isProcessing || Boolean(ragCompliance?.available) },
  ];
  const sessionNarrative = isProcessing
    ? "This call is still moving through the evaluation pipeline. The page will refresh automatically as transcript, explainability, and policy analysis become available."
    : explainabilityCardCount > 0
      ? "This session now includes evidence-backed explainability, so supervisors can move from a score to the exact span and policy evidence behind it."
      : "This session summary is ready. Use the transcript, evaluation cards, and evidence panel to review the conversation from playback through policy reasoning.";

  const handleProgressChange = (value: number) => {
    if (!audioRef.current) {
      return;
    }
    const nextTime = (Math.max(0, Math.min(100, value)) / 100) * totalDurationSeconds;
    audioRef.current.currentTime = nextTime;
    setCurrentTimeSeconds(nextTime);
    syncTranscriptToAudio(nextTime, true);
  };

  return (
    <div className="p-4 md:p-8 space-y-6 max-w-[1600px] mx-auto bg-[#F8FAFC] dark:bg-[#090B10] min-h-screen text-slate-900 dark:text-slate-100">
      {/* Top Navigation */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <Link
          to="/manager/inspector"
          className="inline-flex items-center gap-2 text-[14px] font-semibold text-slate-500 dark:text-slate-300 hover:text-primary transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Sessions
        </Link>
        <div className="flex flex-wrap items-center gap-3">
          {interactionStatus && (
            <div className={`px-3 py-1.5 rounded-full text-[11px] font-bold uppercase tracking-wider border ${statusBadgeClass}`}>
              {interactionStatus}
            </div>
          )}
          <button
            type="button"
            onClick={() => void handleReprocess()}
            disabled={reprocessing || !id}
            className="h-9 px-4 rounded-full bg-slate-900 text-white text-[12px] font-bold hover:bg-slate-800 transition-colors disabled:opacity-50"
          >
            {reprocessing ? "Reprocessing..." : "Reprocess"}
          </button>
          <div className="px-4 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] font-bold text-slate-400 dark:text-slate-300 font-mono">
            ID: {id?.padEnd(32, '0')}
          </div>
        </div>
      </div>

      {actionError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12px] font-semibold text-destructive">
          {actionError}
        </div>
      )}

      {/* Top Header Card */}
      <div className="overflow-hidden rounded-[28px] border border-border bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.18),transparent_30%),linear-gradient(135deg,rgba(255,255,255,0.98),rgba(241,245,249,0.98))] p-8 shadow-sm dark:border-slate-700 dark:bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.18),transparent_30%),linear-gradient(135deg,#0f172a,#111827)]">
        <div className="flex flex-col gap-8 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/15 bg-primary/10 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.22em] text-primary">
              Session review
            </div>
            <h2 className="mb-2 text-[30px] font-extrabold tracking-tight text-slate-900 dark:text-slate-100">
              {interaction.agentName}
            </h2>
            <div className="flex flex-wrap items-center gap-3 text-[13px] font-medium text-slate-500 dark:text-slate-300">
              <span>{interaction.date}</span>
              <span className="h-1 w-1 rounded-full bg-slate-300" />
              <span>{interaction.time}</span>
              <span className="h-1 w-1 rounded-full bg-slate-300" />
              <span>{interaction.duration}</span>
              <span className="h-1 w-1 rounded-full bg-slate-300" />
              <span className="uppercase">{interaction.language}</span>
            </div>
            <p className="mt-4 max-w-2xl text-[14px] leading-6 text-slate-600 dark:text-slate-300">
              {sessionNarrative}
            </p>
            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              {reviewSignals.map((signal) => (
                <div key={signal.label} className="rounded-2xl border border-slate-200/80 bg-white/75 px-4 py-3 shadow-sm dark:border-slate-700 dark:bg-slate-900/70">
                  <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">{signal.label}</div>
                  <div className="mt-1 text-[22px] font-extrabold text-slate-900 dark:text-slate-100">{signal.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex w-full flex-col gap-4 xl:w-auto">
            <div className="flex items-center gap-8 overflow-x-auto pb-2 xl:justify-end">
              <div className="flex shrink-0 flex-col items-center">
                <span className="mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">Overall Score</span>
                <div className="relative h-[74px] w-[74px]">
                  <svg className="h-full w-full -rotate-90">
                    <circle cx="37" cy="37" r="31" fill="none" stroke="#E2E8F0" strokeWidth="6" />
                    <circle
                      cx="37"
                      cy="37"
                      r="31"
                      fill="none"
                      stroke={getScoreColor(interaction.overallScore)}
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={`${(interaction.overallScore / 100) * 194.7} 194.7`}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-[18px] font-extrabold" style={{ color: getScoreColor(interaction.overallScore) }}>
                      {interaction.overallScore}
                    </span>
                  </div>
                </div>
              </div>

              <div className="hidden h-16 w-px bg-slate-200 dark:bg-slate-700 sm:block" />

              <div className="flex items-center gap-8 xl:gap-10">
                {[
                  { label: "Empathy", score: interaction.empathyScore, color: "#3B82F6", unit: "%" },
                  { label: "Policy", score: interaction.policyScore, color: "#10B981", unit: "%" },
                  { label: "Resolution", score: interaction.resolutionScore, color: "#8B5CF6", unit: "%" },
                  { label: "Response Time", score: responseTimeText, color: "#334155", unit: "" },
                ].map((s) => (
                  <div key={s.label} className="flex flex-col items-center text-center">
                    <div className="mb-2 whitespace-nowrap text-[10px] font-extrabold uppercase tracking-wider text-slate-400">{s.label}</div>
                    <div className="text-[20px] font-extrabold" style={{ color: s.color }}>
                      {s.score}{s.unit}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {isProcessing && (
              <div className="rounded-2xl border border-amber-200/60 bg-amber-50/80 px-4 py-4 dark:border-amber-900/40 dark:bg-amber-950/20">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-amber-700 dark:text-amber-300">Processing state</div>
                    <div className="mt-1 text-[13px] font-medium text-amber-900 dark:text-amber-100">
                      Auto-refresh is on while new transcript and evaluation artifacts arrive.
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {processingSteps.map((step) => (
                      <div
                        key={step.label}
                        className={`rounded-full border px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.16em] ${
                          step.done
                            ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300"
                            : "border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300"
                        }`}
                      >
                        {step.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="space-y-6">
        {/* Primary Review Surface: Player, Transcript, and Emotion Graph */}
        <div className="space-y-6">
          
          {/* New Audio Player Layout */}
          <div className="rounded-[24px] border border-border bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(239,246,255,0.96))] p-6 pr-8 shadow-sm dark:border-slate-700 dark:bg-[linear-gradient(135deg,#0f172a,#172554)]">
            <audio 
              ref={audioRef} 
              src={getAudioUrl(interaction.id)} 
              preload="metadata"
            />
            <div className="flex items-center gap-6">
              <button 
                onClick={() => {
                  if (audioRef.current) {
                    isPlaying ? audioRef.current.pause() : audioRef.current.play();
                  }
                }}
                className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-[#3B82F6] text-white shadow-[0_8px_16px_rgba(59,130,246,0.3)] transition-colors hover:bg-blue-600"
              >
                {isPlaying ? (
                  <div className="flex h-4 w-4 justify-between">
                    <div className="w-1.5 rounded-sm bg-white"></div>
                    <div className="w-1.5 rounded-sm bg-white"></div>
                  </div>
                ) : (
                  <Play className="ml-1.5 h-6 w-6 fill-current" />
                )}
              </button>

              <div className="flex flex-wrap gap-2">
                <span className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.16em] text-blue-700 dark:border-blue-900/40 dark:bg-blue-950/30 dark:text-blue-200">
                  Auto-sync transcript
                </span>
                <span className="rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.16em] text-slate-600 dark:border-slate-700 dark:bg-slate-900/70 dark:text-slate-300">
                  {utterances.length} utterances
                </span>
              </div>
            </div>

            <div className="mt-5 flex flex-1 flex-col justify-center pt-2">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[12px] font-bold text-slate-500 dark:text-slate-300">{formatClock(currentTimeSeconds)}</span>
                <span className="text-[12px] font-bold text-slate-400 dark:text-slate-400">{formatClock(totalDurationSeconds)}</span>
              </div>
              <div className="group relative mb-1 h-[6px] w-full rounded-full bg-slate-100 dark:bg-slate-800">
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={0.1}
                  value={progressPercent || 0}
                  onChange={(event) => handleProgressChange(Number(event.target.value))}
                  className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0"
                  aria-label="Audio seek"
                />
                <div className="pointer-events-none absolute bottom-0 left-0 top-0 rounded-full bg-[#3B82F6]" style={{ width: `${progressPercent}%` }} />
                <div 
                  className="pointer-events-none absolute top-1/2 z-0 h-3.5 w-3.5 -translate-y-1/2 rounded-full border-2 border-white bg-[#3B82F6] shadow transition-transform group-hover:scale-125 dark:border-slate-950" 
                  style={{ left: `calc(${progressPercent}% - 6px)` }} 
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-[12px] text-slate-500 dark:text-slate-300">
                <span>Click any transcript span or evidence card to jump playback.</span>
                <span className="font-semibold">{isProcessing ? "Playback available while analysis finishes" : "Playback synced with transcript view"}</span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-900 rounded-[24px] border border-border dark:border-slate-700 overflow-hidden shadow-sm flex flex-col">
            <div className="flex items-center gap-3 border-b border-slate-100 p-6 pb-4">
              <h3 className="text-[18px] font-extrabold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                Session Transcript
              </h3>
            </div>

            <div className="p-6 bg-slate-50/50 dark:bg-slate-900/40">
            {/* Emotion Graph Segment */}
            <div className="mb-8 p-5 bg-white dark:bg-slate-900 rounded-[16px] border border-slate-200 dark:border-slate-700 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] uppercase tracking-[0.15em] font-extrabold text-slate-500 dark:text-slate-300">Emotion Graph</span>
                <span className="text-[11px] font-bold text-slate-400 dark:text-slate-300">Duration {formatClock(totalDurationSeconds)}</span>
              </div>
              
              {/* Present Agent / Client Emotion Summary inline */}
              <div className="flex flex-col gap-4 mb-6">
                <div className="flex items-center gap-3">
                  <span className="text-[10px] font-extrabold uppercase tracking-wider text-primary w-[50px]">Agent</span>
                  <div className="flex flex-wrap gap-2">
                    {speakerEmotionSummary.agent.length === 0 ? (
                      <span className="text-[11px] text-slate-400 font-medium">No labels</span>
                    ) : speakerEmotionSummary.agent.map((emotion) => {
                      const meta = getEmotionMeta(emotion);
                      return (
                        <span key={`agent-chip-${emotion}`} className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] font-bold shadow-sm" style={{ backgroundColor: meta.surface, color: meta.color }}>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: meta.color }}></span>{meta.label}
                        </span>
                      );
                    })}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] font-extrabold uppercase tracking-wider text-success w-[50px]">Customer</span>
                  <div className="flex flex-wrap gap-2">
                    {speakerEmotionSummary.client.length === 0 ? (
                      <span className="text-[11px] text-slate-400 font-medium">No labels</span>
                    ) : speakerEmotionSummary.client.map((emotion) => {
                      const meta = getEmotionMeta(emotion);
                      return (
                        <span key={`client-chip-${emotion}`} className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-[11px] font-bold shadow-sm" style={{ backgroundColor: meta.surface, color: meta.color }}>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: meta.color }}></span>{meta.label}
                        </span>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* The Graph Canvas itself */}
              <div className="relative rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0B0F19] overflow-hidden shadow-sm pt-4 pb-3">
                <div className="px-8 pb-3 flex flex-wrap items-center gap-4 text-[10px] font-bold uppercase tracking-[0.14em] text-slate-500 dark:text-slate-300">
                  <span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[#3B82F6]" />Agent (solid)</span>
                  <span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[#10B981]" />Customer (dashed)</span>
                  <span className="inline-flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-indigo-500" />Playback</span>
                </div>
                {/* Visual Guidelines for Graph Values (Positive, Neutral, Negative) */}
                <div className="absolute inset-x-8 top-[40px] bottom-[30px]">
                  <div className="absolute left-0 right-0 top-[20%] border-t border-dashed border-slate-100 dark:border-slate-800/80">
                    <span className="absolute -left-6 -top-3.5 text-[9px] uppercase tracking-wider font-bold text-success/70">Positive</span>
                  </div>
                  <div className="absolute left-0 right-0 top-[50%] border-t border-slate-200/80 dark:border-slate-700/80">
                    <span className="absolute -left-6 -top-3.5 text-[9px] uppercase tracking-wider font-bold text-slate-400/70 dark:text-slate-400/70">Neutral</span>
                  </div>
                  <div className="absolute left-0 right-0 top-[80%] border-t border-dashed border-slate-100 dark:border-slate-800/80">
                    <span className="absolute -left-6 -top-3.5 text-[9px] uppercase tracking-wider font-bold text-destructive/70">Negative</span>
                  </div>
                </div>

                <div className="relative h-[220px] mx-8 mt-1">
                  <svg viewBox={`0 0 ${emotionGraphData.width} ${emotionGraphData.height}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full pointer-events-none drop-shadow-sm">
                    {/* Agent Path (Solid Smooth Curve) */}
                    {emotionGraphData.agentPath && (
                      <>
                        <path d={emotionGraphData.agentPath} fill="none" stroke="rgba(59, 130, 246, 0.2)" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
                        <path d={emotionGraphData.agentPath} fill="none" stroke="#3B82F6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                      </>
                    )}
                    {/* Customer Path (Dashed Smooth Curve) */}
                    {emotionGraphData.clientPath && (
                      <>
                        <path d={emotionGraphData.clientPath} fill="none" stroke="rgba(16, 185, 129, 0.2)" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
                        <path d={emotionGraphData.clientPath} fill="none" stroke="#10B981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="6 4" />
                      </>
                    )}
                  </svg>

                  {emotionGraphData.markers.map((marker, index) => {
                    const isActive = activeUtteranceId === marker.id;
                    const zIndex = isActive ? 20 : 10;
                    return (
                      <button
                        key={`emotion-marker-${marker.id}-${index}`}
                        type="button"
                        onClick={() => handleJumpTo(marker.start)}
                        className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border-[2.5px] transition-all duration-200 ${isActive ? "w-5 h-5 scale-110" : "w-3.5 h-3.5 hover:scale-125"}`}
                        style={{
                          left: `${marker.leftPct}%`,
                          top: `${marker.yPct}%`,
                          backgroundColor: marker.color,
                          borderColor: "#FFFFFF",
                          boxShadow: isActive ? `0 0 0 4px ${marker.color}40, 0 4px 12px rgba(0,0,0,0.3)` : "0 2px 4px rgba(0,0,0,0.15)",
                          zIndex
                        }}
                        title={`${marker.timestamp} • ${marker.speaker} • ${marker.label} • ${Math.round(marker.confidence * 100)}% confidence • ${marker.signal.toFixed(2)} intensity`}
                      />
                    );
                  })}

                  <div className="absolute top-0 bottom-0 w-[2px] bg-indigo-500 shadow-[0_0_12px_rgba(99,102,241,0.5)] pointer-events-none z-30 transition-all duration-100 ease-linear" style={{ left: `${playbackLineLeft}%` }}>
                    <div className="absolute -top-1.5 -translate-x-[calc(50%-1px)] w-3 h-3 rounded-full bg-indigo-500 ring-4 ring-indigo-500/20" />
                    <div className="absolute -bottom-1 -translate-x-[calc(50%-1px)] w-2.5 h-2.5 rounded-full bg-indigo-500" />
                  </div>
                </div>

                <div className="px-8 mt-2">
                  <div className="flex items-center justify-between text-[11px] font-bold text-slate-400 dark:text-slate-500 pt-3 border-t border-slate-100 dark:border-slate-800">
                    {emotionGraphData.ticks.map((tick) => (
                      <span key={`graph-tick-${tick.label}-${tick.leftPct}`}>{tick.label}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Transcript Chat Interface */}
            <div ref={transcriptRef} className="space-y-4 max-h-[36rem] overflow-y-auto pr-3 pl-2 scroll-smooth">
              {utterances.map((u) => {
                const isAgent = u.speaker === "agent";
                const isActive = activeUtteranceId === u.id;
                const emotionMeta = getEmotionMeta(u.fusedEmotion || u.emotion);
                const emotionConfidence = normalizeConfidence(u.fusedConfidence ?? u.confidence ?? u.textConfidence);
                return (
                  <div 
                    key={u.id} 
                    id={`utterance-${u.id}`}
                    onClick={() => handleJumpTo(u.startTime)}
                    className={`group flex items-end gap-3 cursor-pointer transition-all duration-300 max-w-[94%] overflow-visible ${isAgent ? "mr-auto" : "ml-auto flex-row-reverse"} ${isActive ? "opacity-100 scale-[1.01]" : "opacity-60 hover:opacity-100 grayscale-[30%] hover:grayscale-0"}`}
                  >
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold shrink-0 shadow-sm ring-2 ring-transparent ${isActive ? "ring-primary/40" : ""} ${isAgent ? "bg-primary text-white" : "bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-200"}`}>
                      {isAgent ? "A" : "C"}
                    </div>
                    <div className="flex-1 flex flex-col gap-1.5 pt-1">
                      <div className={`flex items-center gap-2 px-1 ${isAgent ? "" : "flex-row-reverse"}`}>
                        <span className="font-extrabold text-[12px] text-slate-700 dark:text-slate-100">{isAgent ? interaction.agentName : "Customer"}</span>
                        <span className="text-[10px] font-bold text-slate-400 dark:text-slate-300">{u.timestamp}</span>
                      </div>
                      <div className={`p-4 shadow-sm text-[14px] leading-relaxed transition-colors border ${isAgent ? "bg-[#EEF2FF] dark:bg-[#20263A] border-[#E0E7FF] dark:border-[#324166] text-[#312E81] dark:text-[#DDE6FF] rounded-2xl rounded-bl-sm" : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-100 rounded-2xl rounded-br-sm"}`}>
                        {u.text}
                      </div>
                      <div className={`flex items-center gap-2 ${isAgent ? "" : "justify-end"} pt-0.5`}>
                        <span
                          className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wide"
                          style={{ backgroundColor: `${emotionMeta.color}15`, color: emotionMeta.color }}
                        >
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: emotionMeta.color }}></span>
                          {emotionMeta.label}
                        </span>
                        <span className="text-[10px] font-semibold text-slate-400 dark:text-slate-300">
                          {Math.round(emotionConfidence * 100)}% confidence
                        </span>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setWindowedUtteranceId(u.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px] font-bold text-slate-500 dark:text-slate-300 hover:text-primary"
                        >
                          Open Window
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          </div>
        </div>

        {/* Secondary Review Surface: Evidence and Evaluation Cards */}
        <div className="space-y-6">

          <EvidenceAnchoredExplainabilityPanel explainability={explainability} onJumpTo={handleJumpTo} />

          <div className="grid gap-6 xl:grid-cols-2 items-start">
          {/* Automated Evaluation Card (RAG Compliance + Policy) */}
          <div className="bg-white dark:bg-slate-900 rounded-[24px] border border-border dark:border-slate-700 shadow-sm flex flex-col overflow-hidden">
            <div className="p-6 bg-slate-900 flex items-start justify-between">
              <div>
                <h3 className="text-[18px] font-extrabold text-white flex items-center gap-3">
                  <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
                  Automated Evaluation
                </h3>
                <p className="text-[13px] font-medium text-slate-400 mt-1">LLM process & policy scan</p>
              </div>
              <button className="w-10 h-10 flex shrink-0 items-center justify-center rounded-xl bg-white/10 hover:bg-white/20 text-white transition-colors">
                <Play className="w-4 h-4 fill-current ml-0.5" />
              </button>
            </div>
            
            <div className="p-6 space-y-8">
              {!ragCompliance?.available ? (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5 dark:border-slate-700 dark:bg-slate-800">
                  {isProcessing ? (
                    <div className="space-y-4">
                      <div>
                        <p className="text-[13px] font-semibold text-slate-700 dark:text-slate-100">
                          Policy and SOP checks are still being assembled.
                        </p>
                        <p className="mt-1 text-[13px] leading-6 text-slate-500 dark:text-slate-300">
                          Retrieval provenance will appear here once transcription completes and policy chunks are scored.
                        </p>
                      </div>
                      <div className="grid gap-3">
                        {processingSteps.map((step) => (
                          <div key={`rag-${step.label}`} className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-900/70">
                            <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-slate-500 dark:text-slate-300">{step.label}</span>
                            <span className={`rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.16em] ${step.done ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300" : "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-300"}`}>
                              {step.done ? "Ready" : "Queued"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="text-[13px] font-medium text-slate-500 dark:text-slate-300">
                      {`RAG compliance analysis unavailable.${ragCompliance?.error ? ` ${ragCompliance.error}` : ""}`}
                    </p>
                  )}
                </div>
              ) : (
                <>
                  {ragCompliance.processAdherence && (
                    <div className={`relative pl-5 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-1 before:rounded-full space-y-5 ${ragCompliance.processAdherence.isResolved ? "before:bg-success" : "before:bg-destructive"}`}>
                      <div className="flex items-center justify-between">
                        <h4 className="text-[16px] font-extrabold text-slate-800 dark:text-slate-100">Process Adherence</h4>
                        <span className={`px-2.5 py-1 text-[11px] font-extrabold rounded-md uppercase tracking-wider ${ragCompliance.processAdherence.isResolved ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"}`}>{ragCompliance.processAdherence.isResolved ? "Resolved" : "Unresolved"}</span>
                      </div>

                      <div className="grid grid-cols-[100px_1fr] gap-3 text-[13px]">
                        <span className="text-slate-500 font-medium">Topic:</span>
                        <span className="font-extrabold text-slate-800 dark:text-slate-100 text-right">{ragCompliance.processAdherence.detectedTopic}</span>
                        <span className="text-slate-500 font-medium">Efficiency:</span>
                        <span className="font-extrabold text-slate-800 dark:text-slate-100 text-right">{ragCompliance.processAdherence.efficiencyScore}/10</span>
                      </div>

                      <div>
                        <span className="text-[11px] font-extrabold text-[#3B82F6] uppercase tracking-wider mb-2 block">AI Reasoning</span>
                        <p className="text-[13px] text-slate-600 dark:text-slate-300 leading-relaxed font-medium">
                          {ragCompliance.processAdherence.justification}
                        </p>
                      </div>

                      {missingSopSteps.length > 0 && (
                        <div>
                          <span className="text-[11px] font-extrabold text-destructive uppercase tracking-wider mb-2 block">Missing Steps</span>
                          <ul className="text-[13px] font-medium space-y-2">
                            {missingSopSteps.map((step, i) => (
                              <li key={i} className="flex items-start gap-2 text-slate-600 dark:text-slate-300">
                                <span className="w-1.5 h-1.5 rounded-full bg-destructive mt-1.5 shrink-0"></span>
                                {step}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {ragCompliance.nliPolicy && (
                    <div className="relative pl-5 pt-8 border-t border-slate-100 before:absolute before:left-0 before:top-8 before:bottom-0 before:w-1 before:bg-warning before:rounded-full space-y-5">
                      <div className="flex items-center justify-between">
                        <h4 className="text-[16px] font-extrabold text-slate-800 dark:text-slate-100">Policy Inference</h4>
                        <span className="px-2.5 py-1 bg-warning/10 text-warning text-[11px] font-extrabold rounded-md uppercase tracking-wider">{ragCompliance.nliPolicy.nliCategory || "Warning"}</span>
                      </div>
                      <div>
                        <span className="text-[11px] font-extrabold text-[#3B82F6] uppercase tracking-wider mb-2 block">AI Reasoning</span>
                        <p className="text-[13px] text-slate-600 dark:text-slate-300 leading-relaxed font-medium">
                          {ragCompliance.nliPolicy.justification}
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2 pt-1">
                        {ragCompliance.nliPolicy.policyVersion && (
                          <span className="px-2 py-1 text-[10px] font-bold rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                            Policy Version: {ragCompliance.nliPolicy.policyVersion}
                          </span>
                        )}
                        {ragCompliance.nliPolicy.policyCategory && (
                          <span className="px-2 py-1 text-[10px] font-bold rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 uppercase">
                            Category: {ragCompliance.nliPolicy.policyCategory}
                          </span>
                        )}
                        {ragCompliance.nliPolicy.conflictResolutionApplied && (
                          <span className="px-2 py-1 text-[10px] font-bold rounded-md bg-warning/10 text-warning uppercase">
                            Conflict Resolved
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Emotion Trigger Reasoning */}
          {emotionTriggers && (
            <div className="bg-white dark:bg-slate-900 rounded-[24px] border border-border dark:border-slate-700 shadow-sm flex flex-col overflow-hidden">
              <div className="p-6 pb-4 border-b border-slate-100">
                <h3 className="text-[18px] font-extrabold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                  <Headphones className="w-5 h-5 text-purple-500" />
                  Emotion Trigger Reasoning
                </h3>
              </div>
              <div className="p-6">
                {!emotionTriggers.available ? (
                  <div className="rounded-2xl border border-purple-200 bg-purple-50/70 p-5 dark:border-purple-900/40 dark:bg-purple-950/20">
                    {isProcessing ? (
                      <div className="space-y-3">
                        <p className="text-[13px] font-semibold text-purple-800 dark:text-purple-200">
                          Emotion reasoning is waiting on transcript-aligned cues.
                        </p>
                        <p className="text-[13px] leading-6 text-purple-700/80 dark:text-purple-200/80">
                          Once the call is transcribed, this card will anchor acoustic or transcript mismatches to exact utterance spans.
                        </p>
                        <div className="grid gap-2">
                          {[
                            "Customer emotion summary",
                            "Trigger reasoning",
                            "Counterfactual coaching",
                          ].map((item) => (
                            <div key={item} className="rounded-xl border border-purple-200/80 bg-white/80 px-4 py-3 text-[12px] font-medium text-purple-700 dark:border-purple-900/40 dark:bg-slate-900/70 dark:text-purple-200">
                              {item}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <p className="text-[13px] font-medium text-destructive">
                        {`Emotion trigger analysis unavailable.${emotionTriggers.error ? ` ${emotionTriggers.error}` : ""}`}
                      </p>
                    )}
                  </div>
                ) : emotionTriggers.emotionShift && (
                  <div className="space-y-6">
                    {/* Customer Emotion Banner */}
                    <div className="rounded-xl border border-purple-200 dark:border-purple-900/50 bg-purple-50 dark:bg-purple-900/20 p-4">
                      <span className="text-[10px] text-purple-700 dark:text-purple-300 uppercase font-extrabold tracking-widest block mb-1">Full-Call Customer Emotion</span>
                      <p className="text-[15px] font-bold text-purple-800 dark:text-purple-200 capitalize">
                        {emotionTriggers.emotionShift.currentCustomerEmotion || "unknown"}
                      </p>
                      <p className="text-[12px] text-purple-700/90 dark:text-purple-200/90 mt-1 leading-relaxed">
                        {emotionTriggers.emotionShift.currentEmotionReasoning || "insufficient evidence"}
                      </p>
                    </div>

                    {/* Dissonance Detection */}
                    <div className="flex items-center gap-3">
                      <span className={`px-3 py-1.5 rounded-full text-[11px] font-extrabold uppercase tracking-wider border ${emotionTriggers.emotionShift.isDissonanceDetected ? "bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800" : "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800"}`}>
                        {emotionTriggers.emotionShift.isDissonanceDetected
                          ? `Alert: ${emotionTriggers.emotionShift.dissonanceType || "Dissonance"}`
                          : "No Dissonance"}
                      </span>
                      {emotionTriggers.emotionShift.insufficientEvidence && (
                        <span className="px-2.5 py-1 rounded-full text-[10px] font-bold bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700">
                          Limited Evidence
                        </span>
                      )}
                    </div>

                    {/* AI Reasoning */}
                    <div className="relative pl-5 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-1 before:bg-purple-500 before:rounded-full">
                      <span className="text-[11px] font-extrabold text-purple-600 uppercase tracking-wider mb-2 block">AI REASONING</span>
                      <p className="text-[13px] text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                        {emotionTriggers.emotionShift.rootCause}
                      </p>
                    </div>

                    {/* Evidence Quotes */}
                    {emotionTriggers.emotionShift.evidenceQuotes?.length > 0 && (
                      <div>
                        <span className="text-[11px] font-extrabold text-slate-500 uppercase tracking-wider mb-2 block">Evidence Quotes</span>
                        <div className="space-y-2">
                          {emotionTriggers.emotionShift.evidenceQuotes.map((q: string, i: number) => (
                            <div key={i} className="pl-4 border-l-2 border-slate-200 dark:border-slate-700 py-1">
                              <p className="text-[12px] italic text-slate-600 dark:text-slate-300">&ldquo;{q}&rdquo;</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Correction */}
                    <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700">
                      <span className="text-[10px] text-slate-500 dark:text-slate-300 uppercase font-extrabold tracking-widest block mb-1">Recommended Correction</span>
                      <p className="text-[13px] italic text-slate-800 dark:text-slate-200">{emotionTriggers.emotionShift.counterfactualCorrection}</p>
                    </div>

                    {/* Derived Signals */}
                    {emotionTriggers.derived && (
                      <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700">
                          <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest block mb-0.5">Acoustic</span>
                          <span className="text-[13px] font-bold text-slate-700 dark:text-slate-200 capitalize">{emotionTriggers.derived.acousticEmotion}</span>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700">
                          <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest block mb-0.5">Fused</span>
                          <span className="text-[13px] font-bold text-slate-700 dark:text-slate-200 capitalize">{emotionTriggers.derived.fusedEmotion}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          </div>

        </div>
      </div>

      {windowedUtterance && (
        <div className="fixed inset-0 z-50 bg-black/55 backdrop-blur-[2px] flex items-center justify-center p-4">
          <div className="w-full max-w-3xl max-h-[82vh] overflow-hidden rounded-2xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-[#0F131C] shadow-2xl">
            <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
              <div>
                <h4 className="text-[15px] font-extrabold text-slate-900 dark:text-slate-100">Conversation Window</h4>
                <p className="text-[12px] text-slate-500 dark:text-slate-300">Centered around selected utterance at {windowedUtterance.selected.timestamp}</p>
              </div>
              <button
                type="button"
                onClick={() => setWindowedUtteranceId(null)}
                className="text-[12px] font-bold text-slate-500 dark:text-slate-300 hover:text-primary"
              >
                Close
              </button>
            </div>
            <div className="p-5 space-y-3 max-h-[68vh] overflow-y-auto">
              {windowedUtterance.context.map((u) => {
                const isSelected = u.id === windowedUtterance.selected.id;
                const isAgent = u.speaker === "agent";
                return (
                  <div
                    key={`window-${u.id}`}
                    className={`rounded-xl border p-3 ${isSelected ? "border-primary bg-primary/10" : "border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900"}`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[11px] font-bold text-slate-700 dark:text-slate-200">{isAgent ? interaction.agentName : "Customer"}</span>
                      <span className="text-[10px] font-bold text-slate-400 dark:text-slate-300">{u.timestamp}</span>
                    </div>
                    <p className="text-[13px] text-slate-700 dark:text-slate-200">{u.text}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
