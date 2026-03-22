import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, AlertCircle, ChevronDown, ChevronUp, Quote, MessageSquare, Shield, BookOpen, Headphones } from "lucide-react";

export interface EvidenceCitation {
  source: "acoustic" | "text" | "fused";
  speaker?: string;
  quote: string;
  utteranceIndex?: number;
}

export interface EmotionDistribution {
  emotion: string;
  count: number;
  pct: number;
}

export interface FusionQuality {
  acousticTextAgreementRate: number;
  fusedMatchesAcousticRate: number;
  fusedMatchesTextRate: number;
  disagreementCount: number;
}

export interface EmotionComparisonData {
  totalUtterances: number;
  distributions: {
    acoustic: EmotionDistribution[];
    text: EmotionDistribution[];
    fused: EmotionDistribution[];
  };
  quality: FusionQuality;
}

interface EmotionComparisonPanelProps {
  data: EmotionComparisonData;
}

const EMOTION_COLORS: Record<string, string> = {
  neutral: "#6B7280",
  happy: "#10B981",
  sad: "#8B5CF6",
  angry: "#EF4444",
  frustrated: "#F59E0B",
  empathetic: "#06B6D4",
  calm: "#3B82F6",
  disgust: "#EC4899",
  fear: "#F97316",
  surprise: "#FBBF24",
};

function getQualityIndicator(rate: number): { color: string; label: string; icon: string } {
  if (rate >= 80) return { color: "#10B981", label: "Excellent", icon: "✓" };
  if (rate >= 60) return { color: "#3B82F6", label: "Good", icon: "→" };
  if (rate >= 40) return { color: "#F59E0B", label: "Fair", icon: "⚠" };
  return { color: "#EF4444", label: "Poor", icon: "✗" };
}

function formatDistributionForChart(
  distributions: EmotionComparisonData["distributions"],
): Array<{ name: string; acoustic: number; text: number; fused: number }> {
  const emotionMap = new Map<string, { acoustic: number; text: number; fused: number }>();
  const allEmotions = new Set<string>();

  [...distributions.acoustic, ...distributions.text, ...distributions.fused].forEach((d) => {
    allEmotions.add(d.emotion);
  });

  allEmotions.forEach((emotion) => {
    emotionMap.set(emotion, {
      acoustic: distributions.acoustic.find((d) => d.emotion === emotion)?.count ?? 0,
      text: distributions.text.find((d) => d.emotion === emotion)?.count ?? 0,
      fused: distributions.fused.find((d) => d.emotion === emotion)?.count ?? 0,
    });
  });

  return Array.from(emotionMap.entries())
    .map(([emotion, counts]) => ({
      name: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      ...counts,
    }))
    .sort((a, b) => b.acoustic + b.text + b.fused - (a.acoustic + a.text + a.fused))
    .slice(0, 6);
}

export function EmotionComparisonPanel({ data }: EmotionComparisonPanelProps) {
  const [expandedEvidence, setExpandedEvidence] = useState(false);

  const chartData = formatDistributionForChart(data.distributions);
  const acousticQuality = getQualityIndicator(data.quality.acousticTextAgreementRate);
  const fusedAcousticQuality = getQualityIndicator(data.quality.fusedMatchesAcousticRate);
  const fusedTextQuality = getQualityIndicator(data.quality.fusedMatchesTextRate);



  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-[#EFF6FF] rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-[#3B82F6]" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-[#111827]">Emotion Intelligence</h3>
            <p className="text-xs text-[#6B7280]">{data.totalUtterances} utterances analyzed</p>
          </div>
        </div>
      </div>



      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4 hover:shadow-md transition-shadow">
          <div className="flex items-start justify-between mb-3">
            <span className="text-xs font-medium uppercase tracking-wide text-[#6B7280]">Acoustic ↔ Text</span>
            <div className="px-2 py-1 rounded text-xs font-semibold" style={{ backgroundColor: `${acousticQuality.color}20`, color: acousticQuality.color }}>
              {acousticQuality.icon}
            </div>
          </div>
          <div className="text-2xl font-bold text-[#111827] mb-1">{data.quality.acousticTextAgreementRate.toFixed(1)}%</div>
          <p className="text-xs text-[#6B7280]">{acousticQuality.label} agreement</p>
        </div>

        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4 hover:shadow-md transition-shadow">
          <div className="flex items-start justify-between mb-3">
            <span className="text-xs font-medium uppercase tracking-wide text-[#6B7280]">Fused → Acoustic</span>
            <div className="px-2 py-1 rounded text-xs font-semibold" style={{ backgroundColor: `${fusedAcousticQuality.color}20`, color: fusedAcousticQuality.color }}>
              {fusedAcousticQuality.icon}
            </div>
          </div>
          <div className="text-2xl font-bold text-[#111827] mb-1">{data.quality.fusedMatchesAcousticRate.toFixed(1)}%</div>
          <p className="text-xs text-[#6B7280]">Fusion aligns with audio</p>
        </div>

        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4 hover:shadow-md transition-shadow">
          <div className="flex items-start justify-between mb-3">
            <span className="text-xs font-medium uppercase tracking-wide text-[#6B7280]">Fused → Text</span>
            <div className="px-2 py-1 rounded text-xs font-semibold" style={{ backgroundColor: `${fusedTextQuality.color}20`, color: fusedTextQuality.color }}>
              {fusedTextQuality.icon}
            </div>
          </div>
          <div className="text-2xl font-bold text-[#111827] mb-1">{data.quality.fusedMatchesTextRate.toFixed(1)}%</div>
          <p className="text-xs text-[#6B7280]">Fusion aligns with text</p>
        </div>
      </div>

      {data.quality.disagreementCount > 0 && (
        <div className="rounded-lg p-4 flex items-start gap-3" style={{ backgroundColor: "#FEF2F2", borderLeft: "4px solid #EF4444" }}>
          <AlertCircle className="w-5 h-5 text-[#DC2626] flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-[#991B1B]">Cross-Modal Disagreement</p>
            <p className="text-xs text-[#7F1D1D] mt-1">
              {data.quality.disagreementCount} utterance{data.quality.disagreementCount !== 1 ? "s" : ""} show mismatch between acoustic and text emotions.
            </p>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg border border-[#E5E7EB] p-6">
        <h4 className="text-sm font-semibold text-[#111827] mb-4">Emotion Distribution</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E5E7EB" }} />
            <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} axisLine={{ stroke: "#E5E7EB" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#FFF",
                border: "1px solid #E5E7EB",
                borderRadius: "8px",
                boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "20px" }} />
            <Bar dataKey="acoustic" name="Acoustic (Audio)" fill="#3B82F6" />
            <Bar dataKey="text" name="Text (NLP)" fill="#10B981" />
            <Bar dataKey="fused" name="Fused (Composite)" fill="#8B5CF6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4">
          <h5 className="text-xs font-semibold uppercase text-[#6B7280] mb-3">Acoustic Emotions</h5>
          <div className="space-y-2">
            {data.distributions.acoustic.slice(0, 4).map((item) => (
              <div key={`ac-${item.emotion}`} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 flex-1">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: EMOTION_COLORS[item.emotion] || "#9CA3AF" }} />
                  <span className="text-[#111827] font-medium capitalize">{item.emotion}</span>
                </div>
                <span className="text-[#6B7280]">{item.count} ({item.pct.toFixed(1)}%)</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4">
          <h5 className="text-xs font-semibold uppercase text-[#6B7280] mb-3">Text Emotions</h5>
          <div className="space-y-2">
            {data.distributions.text.slice(0, 4).map((item) => (
              <div key={`tx-${item.emotion}`} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 flex-1">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: EMOTION_COLORS[item.emotion] || "#9CA3AF" }} />
                  <span className="text-[#111827] font-medium capitalize">{item.emotion}</span>
                </div>
                <span className="text-[#6B7280]">{item.count} ({item.pct.toFixed(1)}%)</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-[#E5E7EB] p-4">
          <h5 className="text-xs font-semibold uppercase text-[#6B7280] mb-3">Fused Emotions</h5>
          <div className="space-y-2">
            {data.distributions.fused.slice(0, 4).map((item) => (
              <div key={`fu-${item.emotion}`} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 flex-1">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: EMOTION_COLORS[item.emotion] || "#9CA3AF" }} />
                  <span className="text-[#111827] font-medium capitalize">{item.emotion}</span>
                </div>
                <span className="text-[#6B7280]">{item.count} ({item.pct.toFixed(1)}%)</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
