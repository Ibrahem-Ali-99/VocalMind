import { useMemo, useState, type ReactNode } from "react";
import type {
  ClaimProvenance,
  EvidenceAnchoredExplainability,
  TriggerAttribution,
} from "../../services/api";

const verdictTheme: Record<string, string> = {
  Supported: "bg-emerald-50 text-emerald-700 border-emerald-200",
  "Partial Attempt": "bg-amber-50 text-amber-700 border-amber-200",
  Neutral: "bg-slate-100 text-slate-700 border-slate-200",
  Contradiction: "bg-rose-50 text-rose-700 border-rose-200",
  "Cross-Modal Mismatch": "bg-purple-50 text-purple-700 border-purple-200",
  "No Trigger": "bg-sky-50 text-sky-700 border-sky-200",
  "Insufficient Evidence": "bg-slate-100 text-slate-500 border-slate-200",
};

function formatPercent(value?: number | null): string {
  const numeric = Number(value);
  if (value == null || !Number.isFinite(numeric)) {
    return "N/A";
  }
  return `${Math.round(Math.max(0, Math.min(1, numeric)) * 100)}%`;
}

function cleanEvidenceText(value?: string | null): string {
  return (value || "")
    .replace(/\r/g, "")
    .replace(/^\s*#{1,6}\s*/gm, "")
    .replace(/<!--.*?-->/gs, " ")
    .replace(/\s*Ã¢â‚¬Â¢\s*/g, " / ")
    .replace(/\s*â€¢\s*/g, " / ")
    .replace(/\|{2,}/g, " / ")
    .replace(/[-|]{4,}/g, " ")
    .replace(/^\s*[-*]\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function VerdictBadge({ verdict }: { verdict: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-wider ${
        verdictTheme[verdict] || verdictTheme.Neutral
      }`}
    >
      {verdict}
    </span>
  );
}

function JumpButton({
  timestamp,
  startSeconds,
  onJumpTo,
}: {
  timestamp?: string | null;
  startSeconds?: number | null;
  onJumpTo?: (seconds: number) => void;
}) {
  if (!onJumpTo || startSeconds == null) {
    return timestamp ? <span className="text-[11px] font-bold text-slate-400">{timestamp}</span> : null;
  }

  return (
    <button
      type="button"
      onClick={() => onJumpTo(startSeconds)}
      className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-wider text-slate-600 transition-colors hover:border-primary hover:text-primary"
    >
      {timestamp || "Jump"}
    </button>
  );
}

function Pager({
  count,
  index,
  onChange,
}: {
  count: number;
  index: number;
  onChange: (next: number) => void;
}) {
  if (count <= 1) {
    return null;
  }

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => onChange(Math.max(0, index - 1))}
        disabled={index === 0}
        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[10px] font-extrabold uppercase tracking-wider text-slate-600 disabled:opacity-40"
      >
        Prev
      </button>
      <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-400">
        {index + 1} / {count}
      </span>
      <button
        type="button"
        onClick={() => onChange(Math.min(count - 1, index + 1))}
        disabled={index === count - 1}
        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[10px] font-extrabold uppercase tracking-wider text-slate-600 disabled:opacity-40"
      >
        Next
      </button>
    </div>
  );
}

function TriggerCard({
  attribution,
  onJumpTo,
}: {
  attribution: TriggerAttribution;
  onJumpTo?: (seconds: number) => void;
}) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-extrabold uppercase tracking-[0.24em] text-slate-400">
            {cleanEvidenceText(attribution.triggerType)}
          </p>
          <h4 className="mt-1 text-[18px] font-extrabold text-slate-900">
            {cleanEvidenceText(attribution.title)}
          </h4>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-wider text-slate-600">
            Confidence {formatPercent(attribution.confidence)}
          </span>
          <VerdictBadge verdict={attribution.verdict} />
        </div>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="space-y-3">
          {attribution.evidenceSpan && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">
                  Evidence Span
                  {attribution.evidenceSpan.speaker
                    ? ` - ${cleanEvidenceText(attribution.evidenceSpan.speaker)}`
                    : ""}
                </p>
                <JumpButton
                  timestamp={attribution.evidenceSpan.timestamp}
                  startSeconds={attribution.evidenceSpan.startSeconds}
                  onJumpTo={onJumpTo}
                />
              </div>
              <p className="mt-2 text-[13px] italic leading-relaxed text-slate-700">
                &ldquo;{cleanEvidenceText(attribution.evidenceSpan.quote)}&rdquo;
              </p>
            </div>
          )}

          {attribution.policyReference && (
            <div className="rounded-xl border border-orange-200 bg-orange-50 p-3">
              <p className="text-[11px] font-extrabold uppercase tracking-wider text-orange-700">
                {attribution.policyReference.source === "sop" ? "SOP Clause" : "Policy Clause"}
              </p>
              <p className="mt-1 text-[12px] font-bold text-orange-900">
                {cleanEvidenceText(attribution.policyReference.reference)}
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-orange-900/80">
                {cleanEvidenceText(attribution.policyReference.clause)}
              </p>
              {attribution.policyReference.provenance && (
                <p className="mt-2 text-[11px] font-medium text-orange-800/70">
                  {cleanEvidenceText(attribution.policyReference.provenance)}
                </p>
              )}
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-slate-200 bg-white p-3">
            <p className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">Reasoning</p>
            <div className="mt-2 max-h-[15rem] overflow-y-auto pr-1 text-[13px] leading-relaxed text-slate-700">
              {cleanEvidenceText(attribution.reasoning)}
            </div>
          </div>

          {attribution.evidenceChain.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">Evidence Chain</p>
              <div className="mt-2 space-y-2">
                {attribution.evidenceChain.map((step) => (
                  <div key={step} className="rounded-xl bg-white px-3 py-2 text-[12px] font-medium text-slate-600">
                    {cleanEvidenceText(step)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}

function ClaimCard({
  claim,
  onJumpTo,
}: {
  claim: ClaimProvenance;
  onJumpTo?: (seconds: number) => void;
}) {
  return (
    <article className="rounded-2xl border border-teal-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-extrabold uppercase tracking-[0.24em] text-teal-600">
            Claim Provenance
          </p>
          <h4 className="mt-1 text-[18px] font-extrabold text-slate-900">
            {cleanEvidenceText(claim.claimText)}
          </h4>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-teal-50 px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-wider text-teal-700">
            Similarity {formatPercent(claim.semanticSimilarity)}
          </span>
          <VerdictBadge verdict={claim.nliVerdict} />
        </div>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <div className="space-y-3">
          {claim.claimSpan && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">
                  Agent Claim
                </p>
                <JumpButton
                  timestamp={claim.claimSpan.timestamp}
                  startSeconds={claim.claimSpan.startSeconds}
                  onJumpTo={onJumpTo}
                />
              </div>
              <p className="mt-2 text-[13px] italic leading-relaxed text-slate-700">
                &ldquo;{cleanEvidenceText(claim.claimSpan.quote)}&rdquo;
              </p>
            </div>
          )}

          {claim.retrievedPolicy && (
            <div className="rounded-xl border border-teal-200 bg-teal-50 p-3">
              <p className="text-[11px] font-extrabold uppercase tracking-wider text-teal-700">
                Retrieved Policy
              </p>
              <p className="mt-1 text-[12px] font-bold text-teal-900">
                {cleanEvidenceText(claim.retrievedPolicy.reference)}
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-teal-900/80">
                {cleanEvidenceText(claim.retrievedPolicy.clause)}
              </p>
              <p className="mt-2 text-[11px] font-medium text-teal-800/70">
                {cleanEvidenceText(claim.provenance)}
              </p>
            </div>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-3">
          <p className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">Reasoning</p>
          <div className="mt-2 max-h-[15rem] overflow-y-auto pr-1 text-[13px] leading-relaxed text-slate-700">
            {cleanEvidenceText(claim.reasoning)}
          </div>
          <p className="mt-4 text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Confidence {formatPercent(claim.confidence)}
          </p>
        </div>
      </div>
    </article>
  );
}

function PagedSection({
  sectionKey,
  title,
  accent,
  count,
  index,
  onChange,
  cardKey,
  children,
}: {
  sectionKey: string;
  title: string;
  accent: string;
  count: number;
  index: number;
  onChange: (next: number) => void;
  cardKey: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-white/[0.04] p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className={`text-[11px] font-extrabold uppercase tracking-[0.24em] ${accent}`}>{title}</p>
        <Pager count={count} index={index} onChange={onChange} />
      </div>
      <div
        key={`${sectionKey}-${cardKey}`}
        style={{ animation: "evidenceCardTurn 320ms cubic-bezier(0.22, 1, 0.36, 1)" }}
      >
        {children}
      </div>
    </div>
  );
}

export function EvidenceAnchoredExplainabilityPanel({
  explainability,
  onJumpTo,
}: {
  explainability?: EvidenceAnchoredExplainability | null;
  onJumpTo?: (seconds: number) => void;
}) {
  const triggerAttributions = explainability?.triggerAttributions ?? [];
  const claimProvenance = explainability?.claimProvenance ?? [];
  const [pages, setPages] = useState<Record<string, number>>({});
  const [activeSection, setActiveSection] = useState<string>("");

  const grouped = useMemo(
    () => ({
      sop: triggerAttributions.filter((item) => item.family === "sop"),
      policy: triggerAttributions.filter((item) => item.family === "policy"),
      emotion: triggerAttributions.filter((item) => item.family === "emotion"),
      claims: claimProvenance,
    }),
    [claimProvenance, triggerAttributions],
  );

  const sections = useMemo(
    () =>
      [
        { key: "sop", title: "SOP Violations", accent: "text-orange-300", items: grouped.sop, kind: "trigger" as const },
        { key: "policy", title: "Policy Findings", accent: "text-rose-300", items: grouped.policy, kind: "trigger" as const },
        { key: "emotion", title: "Span-Level Trigger Attribution", accent: "text-purple-300", items: grouped.emotion, kind: "trigger" as const },
        { key: "claims", title: "Retrieval Provenance Scoring", accent: "text-teal-300", items: grouped.claims, kind: "claim" as const },
      ].filter((section) => section.items.length > 0),
    [grouped],
  );

  if (!triggerAttributions.length && !claimProvenance.length) {
    return null;
  }

  const getIndex = (key: string, count: number) => Math.min(pages[key] ?? 0, Math.max(0, count - 1));
  const setIndex = (key: string, next: number) => setPages((current) => ({ ...current, [key]: next }));
  const currentSection = sections.find((section) => section.key === activeSection) || sections[0];
  const currentIndex = getIndex(currentSection.key, currentSection.items.length);
  const currentItem = currentSection.items[currentIndex];

  return (
    <section className="rounded-[24px] border border-slate-200 bg-[linear-gradient(180deg,#0F172A_0%,#111827_110%)] p-6 shadow-sm">
      <style>
        {`
          @keyframes evidenceCardTurn {
            0% {
              opacity: 0;
              transform: perspective(1100px) rotateY(-10deg) translateY(8px) scale(0.985);
            }
            100% {
              opacity: 1;
              transform: perspective(1100px) rotateY(0deg) translateY(0) scale(1);
            }
          }
        `}
      </style>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-extrabold uppercase tracking-[0.26em] text-orange-300">
            Evidence-Anchored Explainability
          </p>
          <h3 className="mt-2 text-[20px] font-extrabold text-white">Claim to evidence to verdict</h3>
          <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-300">
            Findings are grouped by type and shown as a compact review deck, so supervisors can page through evidence without growing the page into a long scroll.
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-[11px] font-bold uppercase tracking-wider text-slate-300">
          {triggerAttributions.length} trigger cards / {claimProvenance.length} provenance cards
        </div>
      </div>

      <div className="mt-6 space-y-4">
        <div className="flex flex-wrap gap-2">
          {sections.map((section) => {
            const isActive = section.key === currentSection.key;
            return (
              <button
                key={section.key}
                type="button"
                onClick={() => setActiveSection(section.key)}
                className={`rounded-full border px-3 py-2 text-[11px] font-extrabold uppercase tracking-[0.18em] transition-colors ${
                  isActive
                    ? "border-white/30 bg-white/14 text-white"
                    : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:text-white"
                }`}
              >
                {section.title} ({section.items.length})
              </button>
            );
          })}
        </div>

        <PagedSection
          sectionKey={currentSection.key}
          title={currentSection.title}
          accent={currentSection.accent}
          count={currentSection.items.length}
          index={currentIndex}
          onChange={(next) => setIndex(currentSection.key, next)}
          cardKey={`${currentSection.key}-${currentIndex}`}
        >
          {currentSection.kind === "claim" ? (
            <ClaimCard claim={currentItem as ClaimProvenance} onJumpTo={onJumpTo} />
          ) : (
            <TriggerCard attribution={currentItem as TriggerAttribution} onJumpTo={onJumpTo} />
          )}
        </PagedSection>
      </div>
    </section>
  );
}
