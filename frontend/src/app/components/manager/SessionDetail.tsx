import { Link, useParams } from "react-router";
import { ArrowLeft, Play, Flag, Loader2, AlertTriangle as AlertTriangleIcon, RefreshCw, ChevronDown, ChevronRight } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { getInteractionDetail, getAudioUrl, reprocessInteraction, type InteractionDetail } from "../../services/api";

export function SessionDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [flaggedEvents, setFlaggedEvents] = useState<string[]>([]);
  const [flaggedViolations, setFlaggedViolations] = useState<string[]>([]);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string[]>([]);
  const [reprocessing, setReprocessing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [audioEpoch, setAudioEpoch] = useState(0);
  const [showProvenance, setShowProvenance] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const audioObjectUrlRef = useRef<string | null>(null);

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

  useEffect(() => {
    if (!id) {
      setAudioSrc(null);
      return;
    }
    let cancelled = false;
    const revoke = () => {
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current);
        audioObjectUrlRef.current = null;
      }
    };
    revoke();
    setAudioSrc(null);

    void fetch(getAudioUrl(id), { credentials: "include" })
      .then((res) => {
        if (!res.ok) throw new Error(`audio ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        const url = URL.createObjectURL(blob);
        audioObjectUrlRef.current = url;
        setAudioSrc(url);
      })
      .catch(() => {
        if (!cancelled) setAudioSrc(null);
      });

    return () => {
      cancelled = true;
      revoke();
    };
  }, [id, audioEpoch]);

  const handleReprocess = async () => {
    if (!id || reprocessing) {
      return;
    }
    setActionError(null);
    setReprocessing(true);
    try {
      await reprocessInteraction(id);
      const refreshed = await getInteractionDetail(id, { skipCache: true });
      setData(refreshed);
      setAudioEpoch((n) => n + 1);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to reprocess interaction";
      if (message.includes("409")) {
        try {
          await reprocessInteraction(id, { force: true });
          const refreshed = await getInteractionDetail(id, { skipCache: true });
          setData(refreshed);
          setAudioEpoch((n) => n + 1);
          return;
        } catch {
          setActionError("This interaction is already processing. Please wait and try again.");
        }
      } else {
        setActionError(message);
      }
    } finally {
      setReprocessing(false);
    }
  };

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
  const isFailedInteraction = String(interaction.status || "").toLowerCase() === "failed";

  const getScoreColor = (score: number) => {
    if (score >= 85) return "var(--success)";
    if (score >= 75) return "var(--primary)";
    return "var(--destructive)";
  };

  const emotionAliases: Record<string, string> = {
    fearful: "frustrated",
    sad: "frustrated",
    surprised: "neutral",
    disgusted: "angry",
  };

  const normalizeEmotion = (raw: string) => emotionAliases[raw] ?? raw;

  const getEmotionStyle = (emotion: string) => {
    const e = normalizeEmotion(emotion);
    switch (e) {
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
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Link
          to="/manager/inspector"
          className="inline-flex items-center gap-2 text-[13px] font-semibold text-primary hover:underline"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Session Inspector
        </Link>
        {isFailedInteraction && (
          <button
            type="button"
            onClick={() => void handleReprocess()}
            disabled={reprocessing}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-border px-3 text-[12px] font-semibold text-foreground hover:bg-muted disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${reprocessing ? "animate-spin" : ""}`} />
            {reprocessing ? "Reprocessing..." : "Reprocess"}
          </button>
        )}
      </div>

      {actionError && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-[12px] font-medium text-destructive">
          {actionError}
        </div>
      )}

      {data.processingFailures && data.processingFailures.length > 0 && (
        <div className="rounded-xl border border-destructive/25 bg-destructive/5 px-4 py-3 text-[12px] text-foreground">
          <p className="font-semibold text-destructive mb-2">Processing errors (from pipeline jobs)</p>
          <ul className="list-disc space-y-1 pl-4 text-muted-foreground">
            {data.processingFailures.map((f, i) => (
              <li key={`${f.stage}-${i}`}>
                <span className="font-mono text-[11px] text-foreground">{f.stage}</span>
                {f.errorMessage ? `: ${f.errorMessage}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}

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

        <div className="mt-6 border-t border-border pt-6">
          <div className="text-label mb-2">Call recording</div>
          {audioSrc ? (
            <audio
              ref={audioRef}
              key={audioSrc}
              src={audioSrc}
              controls
              preload="metadata"
              className="h-10 w-full rounded-lg"
            />
          ) : (
            <p className="text-[12px] text-muted-foreground">Loading audio… If this persists, check login and API CORS.</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Transcript Card */}
        <div className="bg-card rounded-[14px] border border-border p-6 lg:col-span-7">
          <h3 className="text-[16px] font-bold text-foreground mb-1">Transcript</h3>
          <p className="text-[11px] italic text-muted-foreground mb-4">utterances ordered by sequence_index</p>
          <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
            {utterances.map((u) => {
              const isAgent = u.speaker === "agent";
              const emotionStyle = getEmotionStyle(u.emotion);
              return (
                <div key={u.id} className={`flex gap-3 ${isAgent ? "" : "flex-row-reverse"}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${isAgent ? "bg-primary/20 text-primary" : "bg-success/20 text-success"}`}>
                    {isAgent ? "A" : "C"}
                  </div>
                  <div className={`flex-1 p-3 rounded-2xl text-[13px] ${isAgent ? "bg-primary/5 rounded-tl-none" : "bg-success/5 rounded-tr-none"}`}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-foreground/80">{isAgent ? interaction.agentName : "Customer"}</span>
                        <span
                          className="rounded-full px-1.5 py-0.5 text-[10px] font-semibold"
                          style={{ backgroundColor: emotionStyle.bg, color: emotionStyle.text }}
                        >
                          {emotionStyle.label} {Math.round((u.confidence || 0) * 100)}%
                        </span>
                      </div>
                      <span className="text-[10px] text-muted-foreground">{u.timestamp}</span>
                    </div>
                    <p className="text-foreground leading-relaxed">{u.text}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="space-y-6 lg:col-span-5 lg:sticky lg:top-6 self-start">
          {/* Emotion Events */}
          <div className="bg-card rounded-[14px] border border-border p-6">
            <h3 className="text-[16px] font-bold text-foreground mb-1">Emotion Events</h3>
            <p className="text-[11px] italic text-muted-foreground mb-4">emotion_events — AI-detected emotional shifts</p>
            <div className="space-y-4 max-h-[360px] overflow-y-auto pr-1">
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
                      <div className="flex flex-col gap-2 pt-2 border-t border-border/50 sm:flex-row sm:items-center sm:flex-wrap">
                        <span className="text-[11px] text-muted-foreground">
                          Does this AI emotion-shift match what happened on the call?
                        </span>
                        <div className="flex flex-wrap gap-2">
                          <button type="button" onClick={() => setFeedbackSubmitted(prev => [...prev, e.id])} className="text-[11px] font-bold text-success hover:underline">Yes, looks right</button>
                          <button type="button" onClick={() => setFeedbackSubmitted(prev => [...prev, e.id])} className="text-[11px] font-bold text-destructive hover:underline">No, wrong detection</button>
                        </div>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center justify-end pt-2 border-t border-border/50">
                      <button type="button" onClick={() => setFlaggedEvents(prev => [...prev, e.id])} className="text-[11px] font-bold text-muted-foreground hover:text-foreground flex items-center gap-1">
                        <Flag className="w-3 h-3" /> Dispute AI finding
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
            <div className="space-y-4 max-h-[320px] overflow-y-auto pr-1">
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
                      <div className="flex flex-col gap-2 pt-2 border-t border-destructive/10 sm:flex-row sm:items-center sm:flex-wrap">
                        <span className="text-[11px] text-muted-foreground">
                          Does this policy violation call match the transcript and context?
                        </span>
                        <div className="flex flex-wrap gap-2">
                          <button type="button" onClick={() => setFeedbackSubmitted(prev => [...prev, v.id])} className="text-[11px] font-bold text-success hover:underline">Yes, fair finding</button>
                          <button type="button" onClick={() => setFeedbackSubmitted(prev => [...prev, v.id])} className="text-[11px] font-bold text-destructive hover:underline">No, unfair / wrong call</button>
                        </div>
                      </div>
                    )
                  ) : (
                    <div className="flex items-center justify-end pt-2 border-t border-destructive/10">
                      <button type="button" onClick={() => setFlaggedViolations(prev => [...prev, v.id])} className="text-[11px] font-bold text-muted-foreground hover:text-foreground flex items-center gap-1">
                        <Flag className="w-3 h-3" /> Dispute AI finding
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {data.llmTriggers && data.llmTriggers.available && (
        <div className="bg-card rounded-[14px] border border-border p-6 transition-all space-y-6">
          <h3 className="text-[16px] font-bold text-foreground mb-1">Automated Evaluation</h3>
          <p className="text-[11px] italic text-muted-foreground mb-4">LLM trigger analysis saved during processing</p>

          {data.llmTriggers.emotionShift && (
            <div className="rounded-xl border border-border p-4 space-y-2">
              <h4 className="text-[14px] font-semibold text-foreground">Emotion Trigger Reasoning</h4>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[12px] text-muted-foreground">Dissonance:</span>
                <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-warning/10 text-warning">{data.llmTriggers.emotionShift.dissonanceType}</span>
              </div>
              <p className="text-[12px] text-muted-foreground">{data.llmTriggers.emotionShift.rootCause}</p>
              {data.llmTriggers.emotionShift.counterfactualCorrection && (
                <p className="text-[12px] text-foreground italic">Counterfactual: {data.llmTriggers.emotionShift.counterfactualCorrection}</p>
              )}
            </div>
          )}

          {data.llmTriggers.processAdherence && (
            <div className="rounded-xl border border-border p-4 space-y-2">
              <h4 className="text-[14px] font-semibold text-foreground">Process Adherence</h4>
              <div className="flex items-center gap-2">
                <span className="text-[12px] text-muted-foreground">Topic:</span>
                <span className="text-[12px] font-semibold text-foreground">{data.llmTriggers.processAdherence.detectedTopic}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[12px] text-muted-foreground">Status:</span>
                <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${data.llmTriggers.processAdherence.isResolved ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"}`}>
                  {data.llmTriggers.processAdherence.isResolved ? "Resolved" : "Needs follow-up"}
                </span>
              </div>
              <p className="text-[12px] text-muted-foreground">{data.llmTriggers.processAdherence.justification}</p>
              {data.llmTriggers.processAdherence.missingSopSteps.length > 0 && (
                <ul className="list-disc ml-5 text-[12px] text-muted-foreground space-y-1">
                  {data.llmTriggers.processAdherence.missingSopSteps.map((step, idx) => (
                    <li key={idx}>{step}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {data.llmTriggers.nliPolicy && (
            <div className="rounded-xl border border-border p-4 space-y-2">
              <h4 className="text-[14px] font-semibold text-foreground">Policy Inference</h4>
              <div className="flex items-center gap-2">
                <span className="text-[12px] text-muted-foreground">Category:</span>
                <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-primary/10 text-primary">{data.llmTriggers.nliPolicy.nliCategory}</span>
              </div>
              <p className="text-[12px] text-muted-foreground">{data.llmTriggers.nliPolicy.justification}</p>
            </div>
          )}

          {data.llmTriggers.explainability && (
            <div className="rounded-xl border border-border p-4 space-y-4">
              <h4 className="text-[14px] font-semibold text-foreground">Evidence-Anchored Explainability</h4>
              <div className="space-y-3">
                {data.llmTriggers.explainability.triggerAttributions.map((attr) => (
                  <div key={attr.attributionId} className="rounded-lg border border-border/50 p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-[12px] font-semibold text-foreground">{attr.title}</span>
                      <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-muted text-muted-foreground">{attr.verdict}</span>
                    </div>
                    <p className="text-[11px] text-muted-foreground">{attr.triggerType} — {attr.reasoning}</p>
                    {attr.evidenceSpan && (
                      <p className="text-[11px] italic text-foreground/80">"{attr.evidenceSpan.quote}"</p>
                    )}
                  </div>
                ))}
              </div>

              <div>
                <p className="text-[11px] text-muted-foreground mb-2">Claim to evidence to verdict</p>
                <button
                  type="button"
                  onClick={() => setShowProvenance((v) => !v)}
                  className="flex items-center gap-1 text-[12px] font-semibold text-primary hover:underline"
                >
                  {showProvenance ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  Retrieval Provenance Scoring
                </button>
                {showProvenance && (
                  <div className="mt-2 space-y-3">
                    {data.llmTriggers!.explainability!.claimProvenance.map((claim) => (
                      <div key={claim.claimId} className="rounded-lg border border-border/50 p-3 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-[12px] font-semibold text-foreground">{claim.claimText}</span>
                          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-muted text-muted-foreground">{claim.nliVerdict}</span>
                        </div>
                        {claim.retrievedPolicy && (
                          <p className="text-[11px] text-muted-foreground">Policy: {claim.retrievedPolicy.clause}</p>
                        )}
                        <p className="text-[11px] italic text-foreground/80">{claim.reasoning}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
