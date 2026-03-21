import { Link, useParams } from "react-router";
import { 
  ArrowLeft, Play, Pause, Headphones, ThumbsUp, ThumbsDown, 
  CheckCircle, XCircle, Flag, Loader2, AlertTriangle as AlertTriangleIcon, Search, FileText, HelpCircle,
  MessageSquare, Shield, BookOpen
} from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { 
  getInteractionDetail, getAudioUrl, queryRag, 
  type InteractionDetail, type RAGQueryResponse, type LLMEvidenceCitation
} from "../../services/api";
import { EmotionComparisonPanel } from "./EmotionComparisonPanel.tsx";



function formatTime(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

export function SessionDetail() {
  const { id } = useParams();
  const [data, setData] = useState<InteractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshingLLM, setRefreshingLLM] = useState(false);
  const [llmRefreshTick, setLlmRefreshTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Audio state
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Transcript Scroll Refs
  const transcriptContainerRef = useRef<HTMLDivElement>(null);
  const utteranceRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Feedback State
  const [flaggedEvents, setFlaggedEvents] = useState<string[]>([]);
  const [flaggedViolations, setFlaggedViolations] = useState<string[]>([]);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string[]>([]);

  // RAG Query State
  const [ragQuery, setRagQuery] = useState("");
  const [ragMode, setRagMode] = useState<"answer" | "compliance">("answer");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResult, setRagResult] = useState<RAGQueryResponse | null>(null);
  const [ragError, setRagError] = useState<string | null>(null);
  const [showHelpModal, setShowHelpModal] = useState(false);

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

  // Handle Audio Time Update and Auto-Scroll
  useEffect(() => {
    if (!data?.utterances || !transcriptContainerRef.current) return;
    
    // Find active utterance based on currentTime
    const activeUtterance = data.utterances.find(u => 
      currentTime >= u.startTime && currentTime <= (u.endTime || u.startTime + 5)
    );

    if (activeUtterance) {
      const element = utteranceRefs.current.get(activeUtterance.id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [currentTime, data?.utterances]);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play().catch(e => console.error("Playback failed:", e));
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = Number(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleJumpTo = (seconds: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(e => console.error("Playback failed:", e));
      setIsPlaying(true);
    }
  };

  const handleRagQuery = async () => {
    if (!ragQuery.trim()) return;
    setRagLoading(true);
    setRagError(null);
    try {
      const res = await queryRag({
        query: ragQuery,
        mode: ragMode
        // ==================== org_filter temporarily disabled to allow global knowledge search -> important :) =====================
      });
      setRagResult(res);
    } catch (err: any) {
      setRagError(err.message);
    } finally {
      setRagLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#F8FAFC]">
        <Loader2 className="w-10 h-10 text-[#3B82F6] animate-spin" />
        <span className="ml-3 text-[#6B7280] font-medium">Loading session...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#F8FAFC]">
        <div className="text-center p-8 bg-white rounded-2xl shadow-sm border border-[#E5E7EB]">
          <AlertTriangleIcon className="w-12 h-12 text-[#F59E0B] mx-auto mb-4" />
          <p className="text-[#374151] font-semibold text-lg">Failed to load session</p>
          <p className="text-[#6B7280] text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  const { interaction, utterances, emotionEvents, llmTriggers, policyViolations, emotionComparison } = data;

  const getScoreColor = (score: number) => {
    if (score >= 85) return "#10B981";
    if (score >= 75) return "#3B82F6";
    return "#F59E0B";
  };

  const getEmotionStyle = (emotion: string) => {
    switch (emotion) {
      case "neutral": return { bg: "#F1F5F9", text: "#475569", label: "Neutral" };
      case "happy": return { bg: "#ECFDF5", text: "#065F46", label: "Happy" };
      case "angry": return { bg: "#FEF2F2", text: "#991B1B", label: "Angry" };
      case "frustrated": return { bg: "#FFFBEB", text: "#92400E", label: "Frustrated" };
      default: return { bg: "#F1F5F9", text: "#475569", label: "Neutral" };
    }
  };

  return (
    <div className="p-4 md:p-8 bg-[#F8FAFC] min-h-screen">
      <div className="max-w-[1400px] mx-auto space-y-6">
        
        {/* Top Header */}
        <div className="flex items-center justify-between">
          <Link
            to="/manager/inspector"
            className="inline-flex items-center gap-2 text-sm font-semibold text-[#64748B] hover:text-[#0F172A] transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Sessions
          </Link>
          <div className="text-xs font-semibold px-3 py-1 bg-white border border-[#E2E8F0] rounded-full text-[#64748B] shadow-sm">
            ID: {interaction.id}
          </div>
        </div>

        {/* Status Header */}
        <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col">
            <h1 className="text-3xl font-bold tracking-tight text-[#0F172A] mb-1">
              {interaction.agentName}
            </h1>
            <div className="flex items-center gap-3 text-sm text-[#64748B]">
              <span>{interaction.date} • {interaction.time}</span>
              <span className="w-1 h-1 rounded-full bg-[#CBD5E1]" />
              <span>{interaction.duration}</span>
              <span className="w-1 h-1 rounded-full bg-[#CBD5E1]" />
              <span className="uppercase tracking-wider text-xs font-bold">{interaction.language}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex flex-col items-center">
              <span className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Overall Score</span>
              <div className="relative w-20 h-20">
                <svg className="w-full h-full -rotate-90">
                  <circle cx="40" cy="40" r="34" fill="none" stroke="#F1F5F9" strokeWidth="6" />
                  <circle
                    cx="40" cy="40" r="34" fill="none"
                    stroke={getScoreColor(interaction.overallScore)}
                    strokeWidth="6" strokeLinecap="round"
                    strokeDasharray={`${(interaction.overallScore / 100) * 213.6} 213.6`}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-lg font-bold" style={{ color: getScoreColor(interaction.overallScore) }}>
                    {interaction.overallScore}
                  </span>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              <div className="bg-[#F8FAFC] p-3 rounded-xl border border-[#F1F5F9] text-center min-w-[90px]">
                <div className="text-[10px] uppercase font-bold text-[#64748B] mb-1">Empathy</div>
                <div className="text-lg font-semibold text-[#3B82F6]">{interaction.empathyScore}%</div>
              </div>
              <div className="bg-[#F8FAFC] p-3 rounded-xl border border-[#F1F5F9] text-center min-w-[90px]">
                <div className="text-[10px] uppercase font-bold text-[#64748B] mb-1">Policy</div>
                <div className="text-lg font-semibold text-[#10B981]">{interaction.policyScore}%</div>
              </div>
              <div className="bg-[#F8FAFC] p-3 rounded-xl border border-[#F1F5F9] text-center min-w-[90px]">
                <div className="text-[10px] uppercase font-bold text-[#64748B] mb-1">Resolution</div>
                <div className="text-lg font-semibold text-[#8B5CF6]">{interaction.resolutionScore}%</div>
              </div>
              <div className="bg-[#F8FAFC] p-3 rounded-xl border border-[#F1F5F9] text-center min-w-[90px]">
                <div className="text-[10px] uppercase font-bold text-[#64748B] mb-1">Response Time</div>
                <div className="text-lg font-semibold text-[#0F172A]">{interaction.responseTime}</div>
              </div>
            </div>
          </div>
        </div>

        {/* 2-Column Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* LEFT COLUMN: Media & Transcript */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Custom Audio Player */}
            {interaction.audioFilePath && (
              <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm overflow-hidden relative">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#3B82F6] to-[#8B5CF6]" />
                <div className="flex items-center gap-6">
                  <button 
                    onClick={togglePlay}
                    className="w-14 h-14 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#2563EB] flex items-center justify-center text-white shadow-lg hover:shadow-xl hover:scale-105 transition-all outline-none"
                  >
                    {isPlaying ? <Pause className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current ml-1" />}
                  </button>
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between text-xs font-semibold text-[#64748B]">
                      <span>{formatTime(currentTime)}</span>
                      <span>{formatTime(duration)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0" 
                      max={duration || 100} 
                      value={currentTime} 
                      onChange={handleSeek}
                      className="w-full h-2 bg-[#E2E8F0] rounded-lg appearance-none cursor-pointer accent-[#3B82F6]"
                    />
                  </div>
                </div>
                <audio 
                  ref={audioRef}
                  src={getAudioUrl(interaction.id)}
                  onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
                  onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />
              </div>
            )}

            {/* Transcript with Auto-Scroll */}
            <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm flex flex-col" style={{ height: "600px" }}>
              <div className="px-6 py-4 border-b border-[#E2E8F0] flex items-center justify-between bg-[#F8FAFC] rounded-t-2xl">
                <h3 className="font-bold text-[#0F172A] flex items-center gap-2">
                  <FileText className="w-4 h-4 text-[#3B82F6]" /> Session Transcript
                </h3>
              </div>
              <div 
                ref={transcriptContainerRef}
                className="flex-1 overflow-y-auto p-6 space-y-5 scroll-smooth"
              >
                {utterances.map((utterance) => {
                  const isAgent = utterance.speaker === "agent";
                  const emotionStyle = getEmotionStyle(utterance.emotion);
                  const isActive = isPlaying && currentTime >= utterance.startTime && currentTime <= (utterance.endTime || utterance.startTime + 5);

                  return (
                    <div
                      key={utterance.id}
                      ref={(el) => { if (el) utteranceRefs.current.set(utterance.id, el); }}
                      className={`flex gap-4 group transition-opacity ${isAgent ? "" : "flex-row-reverse"} ${isActive ? "opacity-100 scale-[1.02]" : "opacity-80 hover:opacity-100"}`}
                    >
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0 shadow-sm transition-transform ${isAgent ? "bg-gradient-to-br from-[#3B82F6] to-[#1D4ED8]" : "bg-gradient-to-br from-[#10B981] to-[#047857]"}`}>
                        {isAgent ? "A" : "C"}
                      </div>
                      <div 
                        className={`flex-1 max-w-[80%] p-4 shadow-sm border ${
                          isAgent
                            ? `bg-white rounded-[0_16px_16px_16px] ${isActive ? "border-[#3B82F6] ring-1 ring-[#3B82F6] shadow-md" : "border-[#E2E8F0]"}`
                            : `bg-[#F8FAFC] rounded-[16px_0_16px_16px] ${isActive ? "border-[#10B981] ring-1 ring-[#10B981] shadow-md" : "border-[#E2E8F0]"}`
                        }`}
                        dir="auto"
                      >
                        <div className={`flex items-center gap-3 mb-2`}>
                          <span className="text-sm font-bold text-[#0F172A]">
                            {isAgent ? interaction.agentName : "Customer"}
                          </span>
                          <button 
                            onClick={() => handleJumpTo(utterance.startTime)} 
                            className="text-xs font-semibold text-[#94A3B8] hover:text-[#3B82F6] transition-colors bg-[#F1F5F9] px-2 py-0.5 rounded cursor-pointer"
                          >
                            {utterance.timestamp}
                          </button>
                          <span
                            className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider"
                            style={{ backgroundColor: emotionStyle.bg, color: emotionStyle.text }}
                          >
                            {emotionStyle.label} {Math.round(utterance.confidence * 100)}%
                          </span>
                        </div>
                        <p className={`text-[15px] leading-relaxed text-start ${isAgent ? "text-[#334155]" : "text-[#475569]"}`}>
                          {utterance.text}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Emotion Comparison Panel */}
            {emotionComparison && (
              <div className="bg-white rounded-2xl border border-[#E2E8F0] p-6 shadow-sm">
                <EmotionComparisonPanel data={emotionComparison} />
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: AI Analysis & RAG Tools */}
          <div className="space-y-6">
            
            {/* LLM Triggers Card */}
            {llmTriggers && (
              <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm overflow-hidden">
                <div className="px-5 py-4 bg-gradient-to-r from-[#0F172A] to-[#1E293B] text-white flex justify-between items-center">
                  <div>
                    <h3 className="font-bold flex items-center gap-2">
                      <Search className="w-4 h-4 text-[#38BDF8]" /> Automated Evaluation
                    </h3>
                    <p className="text-xs text-[#94A3B8] mt-1">LLM process & policy scan</p>
                  </div>
                  <button
                    onClick={() => { setRefreshingLLM(true); setLlmRefreshTick(v => v + 1); }}
                    className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
                    title="Refresh LLM Analysis"
                  >
                    {refreshingLLM ? <Loader2 className="w-4 h-4 animate-spin text-white" /> : <Play className="w-4 h-4 text-white" />}
                  </button>
                </div>
                
                <div className="p-5 space-y-4">
                  {!llmTriggers.available ? (
                    <div className="p-3 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-sm text-[#991B1B]">
                      LLM trigger unavailable. {llmTriggers.error}
                    </div>
                  ) : (
                    <>
                      {llmTriggers.processAdherence && (
                        <div className="p-4 bg-[#F8FAFC] rounded-xl border border-[#E2E8F0] relative overflow-hidden">
                          <div className={`absolute left-0 top-0 bottom-0 w-1 ${llmTriggers.processAdherence.isResolved ? "bg-[#10B981]" : "bg-[#EF4444]"}`} />
                          <div className="flex justify-between items-start mb-3 ml-2">
                            <span className="text-sm font-bold text-[#0F172A]">Process Adherence</span>
                            <span className={`text-xs font-bold px-2 py-1 rounded-md ${llmTriggers.processAdherence.isResolved ? "bg-[#D1FAE5] text-[#065F46]" : "bg-[#FEE2E2] text-[#991B1B]"}`}>
                              {llmTriggers.processAdherence.isResolved ? 'Resolved' : 'Unresolved'}
                            </span>
                          </div>
                          <div className="ml-2 space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-[#64748B]">Topic:</span>
                              <span className="font-semibold text-[#0F172A]">{llmTriggers.processAdherence.detectedTopic}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-[#64748B]">Efficiency:</span>
                              <span className="font-semibold text-[#0F172A]">{llmTriggers.processAdherence.efficiencyScore}/10</span>
                            </div>
                            
                            {llmTriggers.processAdherence.justification && (
                              <div className="mt-3 pt-3 border-t border-[#E2E8F0]">
                                <span className="text-xs font-bold text-[#3B82F6] uppercase mb-1 block">AI Reasoning</span>
                                <p className="text-sm text-[#475569] leading-relaxed">
                                  {llmTriggers.processAdherence.justification}
                                </p>
                              </div>
                            )}

                            {llmTriggers.processAdherence.missingSopSteps.length > 0 && (
                              <div className="mt-3 pt-3 border-t border-[#E2E8F0]">
                                <span className="text-xs font-bold text-[#EF4444] uppercase mb-1 block">Missing Steps</span>
                                <ul className="list-disc pl-4 text-sm text-[#475569] space-y-1">
                                  {llmTriggers.processAdherence.missingSopSteps.map((s, i) => <li key={i}>{s}</li>)}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {llmTriggers.nliPolicy && (
                        <div className="p-4 bg-[#F8FAFC] rounded-xl border border-[#E2E8F0] relative overflow-hidden">
                           <div className={`absolute left-0 top-0 bottom-0 w-1 ${llmTriggers.nliPolicy.nliCategory.includes("Contradict") ? "bg-[#EF4444]" : "bg-[#3B82F6]"}`} />
                          <div className="flex justify-between items-start mb-2 ml-2">
                            <span className="text-sm font-bold text-[#0F172A]">Policy Inference</span>
                            <span className="text-xs font-bold px-2 py-1 rounded-md bg-[#EFF6FF] text-[#1D4ED8]">
                              {llmTriggers.nliPolicy.nliCategory}
                            </span>
                          </div>
                          {llmTriggers.nliPolicy.justification && (
                            <div className="ml-2 mt-3 pt-3 border-t border-[#E2E8F0]">
                              <span className="text-xs font-bold text-[#8B5CF6] uppercase mb-1 block">Policy Reasoning</span>
                              <p className="text-sm text-[#475569] leading-relaxed">
                                {llmTriggers.nliPolicy.justification}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {llmTriggers.emotionShift && llmTriggers.emotionShift.isDissonanceDetected && (
                        <div className="p-4 bg-[#F8FAFC] rounded-xl border border-[#E2E8F0] relative overflow-hidden">
                           <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#F59E0B]" />
                          <div className="flex justify-between items-start mb-2 ml-2">
                            <span className="text-sm font-bold text-[#0F172A]">Emotion Dissonance</span>
                            <span className="text-xs font-bold px-2 py-1 rounded-md bg-[#FEF3C7] text-[#D97706]">
                              {llmTriggers.emotionShift.dissonanceType}
                            </span>
                          </div>
                          <div className="ml-2 mt-3 pt-3 border-t border-[#E2E8F0]">
                            <span className="text-xs font-bold text-[#D97706] uppercase mb-1 block">Root Cause</span>
                            <p className="text-sm text-[#475569] leading-relaxed">
                              {llmTriggers.emotionShift.rootCause}
                            </p>
                          </div>
                          {llmTriggers.emotionShift.counterfactualCorrection && (
                            <div className="ml-2 mt-3 pt-3 border-t border-[#E2E8F0]">
                              <span className="text-xs font-bold text-[#10B981] uppercase mb-1 block">Correction Strategy</span>
                              <p className="text-sm text-[#475569] leading-relaxed font-medium italic">
                                {llmTriggers.emotionShift.counterfactualCorrection}
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            {/* RAG Knowledge Base Queries */}
            <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm overflow-hidden">
               <div className="px-5 py-4 bg-gradient-to-r from-[#8B5CF6] to-[#6D28D9] text-white flex items-center justify-between">
                 <div>
                   <h3 className="font-bold flex items-center gap-2">
                     <FileText className="w-4 h-4 text-[#C4B5FD]" /> Manual Knowledge Search
                   </h3>
                   <p className="text-xs text-[#E9D5FF] mt-1">Fact-check answers & verify policy</p>
                 </div>
                 <button 
                   onClick={() => setShowHelpModal(true)}
                   className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
                   title="Help"
                 >
                   <HelpCircle className="w-4 h-4 text-white" />
                 </button>
               </div>
               <div className="p-5 space-y-4">
                  <div className="flex gap-2 bg-[#F1F5F9] p-1 rounded-lg">
                    <button 
                      onClick={() => setRagMode("answer")}
                      className={`flex-1 text-xs font-bold py-2 rounded-md transition-shadow ${ragMode === "answer" ? "bg-white text-[#8B5CF6] shadow-sm" : "text-[#64748B] hover:text-[#0F172A]"}`}
                    >
                      Fact Check
                    </button>
                    <button 
                      onClick={() => setRagMode("compliance")}
                      className={`flex-1 text-xs font-bold py-2 rounded-md transition-shadow ${ragMode === "compliance" ? "bg-white text-[#8B5CF6] shadow-sm" : "text-[#64748B] hover:text-[#0F172A]"}`}
                    >
                      Policy Scan
                    </button>
                  </div>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={ragQuery}
                      onChange={(e) => setRagQuery(e.target.value)}
                      placeholder={ragMode === "answer" ? "Ask a specific FAQ question..." : "Paste agent text for policy check..."}
                      className="flex-1 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#8B5CF6] focus:border-transparent transition-all"
                      onKeyDown={(e) => e.key === 'Enter' && handleRagQuery()}
                    />
                    <button 
                      onClick={handleRagQuery}
                      disabled={ragLoading || !ragQuery.trim()}
                      className="bg-[#8B5CF6] hover:bg-[#7C3AED] text-white px-4 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center p-2"
                    >
                      {ragLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
                    </button>
                  </div>
                  
                  {ragError && (
                    <div className="p-3 bg-[#FEF2F2] border border-[#FECACA] rounded-xl text-xs text-[#991B1B]">
                      Search failed: {ragError}
                    </div>
                  )}

                  {ragResult && (
                    <div className="pt-4 border-t border-[#E2E8F0] space-y-3">
                      <div className="bg-[#F5F3FF] border border-[#DDD6FE] rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-bold uppercase tracking-wider text-[#6D28D9]">Extracted Answer</span>
                          <span className="text-[10px] bg-white px-2 py-0.5 rounded text-[#6D28D9] font-medium border border-[#DDD6FE]">{(ragResult.timing.total)}s</span>
                        </div>
                        <p className="text-sm text-[#4C1D95] font-medium leading-relaxed">
                          {ragResult.response}
                        </p>
                      </div>
                      
                      {ragResult.chunks.length > 0 && (
                        <div className="space-y-2">
                          <span className="text-xs font-bold text-[#64748B] uppercase px-1">Source Snippets</span>
                          {ragResult.chunks.slice(0, 2).map((chunk, idx) => (
                            <div key={idx} className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3 text-xs text-[#475569]">
                              <div className="flex justify-between mb-1 opacity-60">
                                <span>{chunk.metadata?.title || "Document"}</span>
                                <span>{chunk.score.toFixed(2)}</span>
                              </div>
                              <p className="line-clamp-3">{chunk.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
               </div>
            </div>

            {/* Emotion Events Listing */}
            <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm overflow-hidden">
               <div className="px-5 py-4 border-b border-[#E2E8F0] bg-[#F8FAFC]">
                 <h3 className="font-bold flex items-center gap-2 text-[#0F172A]">
                   Emotion Flags
                 </h3>
               </div>
               <div className="p-5 space-y-4 max-h-[400px] overflow-y-auto">
                {emotionEvents.map((event) => {
                  const fromStyle = getEmotionStyle(event.fromEmotion);
                  const toStyle = getEmotionStyle(event.toEmotion);
                  return (
                    <div key={event.id} className="border border-[#E2E8F0] rounded-xl p-4 bg-white transition-shadow hover:shadow-md">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-1 bg-[#F1F5F9] text-[#475569] font-mono text-[10px] rounded border border-[#E2E8F0]">{event.timestamp}</span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold" style={{ backgroundColor: fromStyle.bg, color: fromStyle.text }}>{fromStyle.label}</span>
                          <span className="text-[#94A3B8] text-xs">→</span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold" style={{ backgroundColor: toStyle.bg, color: toStyle.text }}>{toStyle.label}</span>
                        </div>
                        <button onClick={() => handleJumpTo(event.jumpToSeconds)} className="text-[#3B82F6] hover:text-[#1D4ED8] bg-[#EFF6FF] p-1.5 rounded-lg transition-colors">
                          <Play className="w-3.5 h-3.5 fill-current" />
                        </button>
                      </div>
                      <div className="bg-[#F8FAFC] border-l-2 border-[#CBD5E1] pl-3 py-1 mb-3">
                        <p className="text-xs text-[#64748B] italic">{event.justification}</p>
                      </div>
                      <div className="flex items-center justify-end gap-2 text-xs">
                        <button className="flex items-center gap-1.5 text-[#64748B] hover:text-[#0F172A] px-2 py-1 rounded hover:bg-[#F1F5F9] transition-colors"><ThumbsUp className="w-3.5 h-3.5" /> Accurate</button>
                        <button className="flex items-center gap-1.5 text-[#64748B] hover:text-[#EF4444] px-2 py-1 rounded hover:bg-[#FEF2F2] transition-colors"><ThumbsDown className="w-3.5 h-3.5" /> Incorrect</button>
                      </div>
                    </div>
                  );
                })}
               </div>
            </div>


          </div>
        </div>
      </div>
      
      {/* Help Modal Overlay */}
      {showHelpModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#0F172A]/40 backdrop-blur-sm px-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-md w-full border border-[#E2E8F0] overflow-hidden">
            <div className="px-6 py-4 border-b border-[#E2E8F0] flex justify-between items-center bg-[#F8FAFC]">
              <h3 className="font-bold text-[#0F172A] flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-[#8B5CF6]" /> 
                How to use Knowledge Search
              </h3>
              <button 
                onClick={() => setShowHelpModal(false)}
                className="text-[#94A3B8] hover:text-[#0F172A] transition-colors p-1"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-[#0F172A] bg-[#F5F3FF] text-[#6D28D9] px-3 py-1 rounded w-fit uppercase tracking-wider">Fact Check Mode</h4>
                <p className="text-[#475569] text-sm leading-relaxed">
                  Searches narrow, highly factual snippets. Use this to verify precise claims, statistics, numbers, or short definitions exactly as stated in your knowledge base.
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-[#0F172A] bg-[#F5F3FF] text-[#6D28D9] px-3 py-1 rounded w-fit uppercase tracking-wider">Policy Scan Mode</h4>
                <p className="text-[#475569] text-sm leading-relaxed">
                  Searches full, broad sections of the employee handbook or policies. Use this to paste long excerpts of what an agent said, and verify if their overall actions align with standard company rules.
                </p>
              </div>
            </div>
            <div className="px-6 py-4 bg-[#F8FAFC] border-t border-[#E2E8F0] flex justify-end">
               <button onClick={() => setShowHelpModal(false)} className="px-5 py-2 bg-[#8B5CF6] hover:bg-[#7C3AED] text-white rounded-xl text-sm font-semibold transition-colors">
                 Got it!
               </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
