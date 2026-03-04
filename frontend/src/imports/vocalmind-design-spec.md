VOCALMIND — FULL FIGMA MAKE DESIGN PROMPT
AI-Powered Call Centre Evaluation Platform
Schema v5.1 · Dual-Role Dashboard (Manager + Agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


═══════════════════════════════════════════════════════════════════
SECTION 1 — PROJECT OVERVIEW
═══════════════════════════════════════════════════════════════════

Design a complete, production-ready multi-screen dashboard for VocalMind, an AI-powered call centre evaluation SaaS platform. The platform uses AI to transcribe recorded calls, detect agent and customer emotions, evaluate policy compliance, and generate performance scores.

There are TWO user roles with completely different dashboards:
  - MANAGER — full organizational visibility, oversight, AI assistant, knowledge base control
  - AGENT   — personal performance only, no visibility into other agents' data

These are SEPARATE login experiences — there is NO role-switcher toggle inside the app. A manager logs in and sees the manager dashboard. An agent logs in and sees the agent dashboard. They are two distinct product surfaces.

The design must be modern, data-dense, professional SaaS with a strong visual identity. Think: the design quality of Linear, Vercel, or Retool. Not generic, not template-like.


═══════════════════════════════════════════════════════════════════
SECTION 2 — DESIGN SYSTEM
═══════════════════════════════════════════════════════════════════

── COLOR PALETTE ──────────────────────────────────────────────────

Global Background:       #F3F4F6
Card Background:         #FFFFFF
Sidebar Background:      #0D1117
Sidebar Border:          #1F2937
Top Bar Background:      #FFFFFF
Top Bar Border:          #E5E7EB

Manager Accent:          #3B82F6  (Blue-500)
Manager Accent Dark:     #2563EB  (Blue-600)
Manager Accent Light:    #EFF6FF  (Blue-50)

Agent Accent:            #10B981  (Emerald-500)
Agent Accent Dark:       #059669  (Emerald-600)
Agent Accent Light:      #ECFDF5  (Emerald-50)

Success Green:           #10B981
Success Light:           #ECFDF5
Warning Amber:           #F59E0B
Warning Light:           #FFFBEB
Danger Red:              #EF4444
Danger Light:            #FEF2F2
Purple Accent:           #8B5CF6
Purple Light:            #F5F3FF

Text Primary:            #111827
Text Secondary:          #374151
Text Muted:              #6B7280
Text Disabled:           #9CA3AF
Border Default:          #E5E7EB
Border Subtle:           #F3F4F6

Chart Blue:              #3B82F6
Chart Green:             #10B981
Chart Purple:            #8B5CF6
Chart Amber:             #F59E0B
Chart Red:               #EF4444
Chart Gray:              #6B7280

── TYPOGRAPHY ─────────────────────────────────────────────────────

Font pairing: "DM Serif Display" for hero numbers, "DM Sans" for all UI text.
Fallback: "Playfair Display" + "Outfit". Never use Inter, Roboto, or Arial.

  Display Number (hero KPIs):    DM Serif Display, 48px, Regular
  Page Title (H1):               DM Sans, 22px, Bold (700)
  Section Title (H2):            DM Sans, 16px, SemiBold (600)
  Card Title (H3):               DM Sans, 14px, SemiBold (600)
  Body:                          DM Sans, 14px, Regular (400)
  Body Small:                    DM Sans, 13px, Regular (400)
  Label / Caption:               DM Sans, 11px, SemiBold (600), uppercase, 0.5px letter-spacing
  Monospace (IDs, SQL):          JetBrains Mono or Fira Code, 12px

── SPACING ────────────────────────────────────────────────────────

Base unit: 4px
Content padding: 24px
Card padding: 20px inner
Card gap in grid: 16px
Section gap: 24px
Border radius (cards): 14px
Border radius (buttons): 8px
Border radius (badges): 20px (pill)
Border radius (inputs): 10px
Sidebar width (expanded): 240px
Sidebar width (collapsed): 72px
Top bar height: 56px

── SHADOWS ────────────────────────────────────────────────────────

Card shadow:       0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)
Card hover shadow: 0 4px 12px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06)
Modal shadow:      0 20px 60px rgba(0,0,0,0.20)

── ICONOGRAPHY ────────────────────────────────────────────────────

Use Lucide icon set throughout.
Size: 16px for nav, 18px for cards, 14px for inline.
Icons inside colored containers: 18px inside 36×36px rounded-xl container.


═══════════════════════════════════════════════════════════════════
SECTION 3 — GLOBAL LAYOUT STRUCTURE
═══════════════════════════════════════════════════════════════════

Both manager and agent apps share the same structural shell:

┌──────────────┬──────────────────────────────────────────────────┐
│              │  TOP BAR  (56px, white, page title + actions)     │
│   SIDEBAR    ├──────────────────────────────────────────────────┤
│   240px      │                                                   │
│   dark       │  MAIN CONTENT AREA                               │
│              │  (scrollable, #F3F4F6 background)                │
│              │                                                   │
└──────────────┴──────────────────────────────────────────────────┘

There is NO role-switcher bar. Each role has its own separate login and its own app shell.

── SIDEBAR ────────────────────────────────────────────────────────

Background: #0D1117
Width: 240px (expanded), 72px (collapsed)
Border right: 1px solid #1F2937
Smooth width transition: 0.25s ease

Top section (logo area, 64px tall, border-bottom 1px #1F2937):
  - Logo: rounded-xl container (36×36, filled with role accent color) containing a Mic icon (white, 18px)
  - App name: "VocalMind" in DM Sans 18px Bold, white
  - When collapsed: show icon only

Role badge (below logo, only expanded):
  - Manager: #0C1A3A bg, #1D3A6E border, 8px radius
      "Manager Portal"  13px SemiBold #93C5FD
      "Full org access"  11px #6B7280
  - Agent: #062014 bg, #065F46 border, 8px radius
      "Agent Portal"  13px SemiBold #6EE7B7
      "Personal view only"  11px #6B7280

Navigation items (vertical list, 8px gap, 8px padding on each side):
  Each nav item:
    - 44px tall, full width, 12px horizontal padding, 8px border radius
    - Icon (18px, left) + label (DM Sans 14px Medium), 12px gap
    - Default: #6B7280 icon + text, transparent bg
    - Hover: #E5E7EB text, #1F2937 bg
    - Active: white text, role-accent bg, subtle glow

Manager nav items:
  1. LayoutDashboard — Dashboard
  2. Search          — Session Inspector
  3. MessageSquare   — Manager Assistant
  4. BookOpen        — Knowledge Base
  5. Settings        — Settings

Agent nav items:
  1. Activity  — My Performance
  2. Phone     — My Calls
  3. Settings  — Settings

Bottom of sidebar (border-top 1px #1F2937, 16px padding):
  - Collapse toggle: ChevronLeft / ChevronRight icon, #6B7280
  - User avatar: 32×32 circle, role accent bg, white initials
  - Name: DM Sans 12px Medium white
  - Role: DM Sans 11px #6B7280

── TOP BAR ────────────────────────────────────────────────────────

Background: #FFFFFF
Height: 56px
Border bottom: 1px solid (manager: #DBEAFE, agent: #D1FAE5)
Horizontal padding: 24px

Left:
  - Page title: DM Sans 16px Bold #111827
  - Role badge pill: 11px SemiBold uppercase
    Manager: #EFF6FF bg, #3B82F6 text, #BFDBFE border — "Manager"
    Agent:   #ECFDF5 bg, #10B981 text, #A7F3D0 border — "Agent"

Right:
  - Manager only: "Export" button with Download icon, 13px, gray outlined
  - Bell icon button: 32×32, #F9FAFB bg, #E5E7EB border
  - Avatar circle: 32×32, role accent bg, white initials


═══════════════════════════════════════════════════════════════════
SECTION 4 — MANAGER VIEW SCREENS
═══════════════════════════════════════════════════════════════════

All manager screens use blue accent (#3B82F6). Dark sidebar with blue active states.

──────────────────────────────────────────────────────────────────
SCREEN M-1: MANAGER DASHBOARD
──────────────────────────────────────────────────────────────────

Default landing screen. Content padding: 24px. Vertical scroll. Background: #F3F4F6.

SUBSECTION: KPI Cards Row
─────────────────────────
4 cards in a 4-column grid (gap: 16px). Each card:
  - White bg, 1px #E5E7EB border, 14px radius, 20px padding
  - Top row: label (11px SemiBold uppercase #9CA3AF) + icon container (36×36 rounded-xl)
  - Large number: DM Serif Display 40px in accent color
  - Subtitle: DM Sans 12px #9CA3AF

Card 1 — Average Score
  Icon: BarChart2, #EFF6FF container, #3B82F6 icon
  Value: "84.2%"  Color: #2563EB
  Subtitle: "↑ 2.3% from last week"
  Source: interaction_scores.overall_score — org-wide average

Card 2 — Calls Processed
  Icon: Phone, #ECFDF5 container, #10B981 icon
  Value: "342"  Color: #059669
  Subtitle: "this week"
  Source: interactions WHERE processing_status = 'completed'

Card 3 — Resolution Rate
  Icon: CheckCircle, #ECFDF5 container, #10B981 icon
  Value: "88%"  Color: #059669
  Subtitle: "of completed calls"
  Source: interaction_scores.was_resolved — proportion TRUE

Card 4 — Policy Violations
  Icon: AlertTriangle, #FEF2F2 container, #EF4444 icon
  Value: "12"  Color: #DC2626
  Subtitle: "interactions with at least 1 violation"
  Source: policy_compliance WHERE is_compliant = FALSE, distinct interaction_id count

SUBSECTION: Charts Row (2-column, 2:1 ratio)
────────────────────────────────────────────

Left card (2/3 width) — "Weekly Score Trends"
  Subtitle: 11px #9CA3AF italic — "interaction_scores.overall_score avg, grouped by interaction_date"

  Line chart (height: 200px):
    X-axis: Mon–Sun
    Two lines:
      Line 1 — Avg Score: #3B82F6, 2.5px stroke, dots at points
        Data: Mon 8.2, Tue 8.5, Wed 8.0, Thu 8.7, Fri 8.8, Sat 8.4, Sun 8.1
      Line 2 — Call Count: #10B981, 2px dashed
        Data: Mon 145, Tue 152, Wed 138, Thu 161, Fri 173, Sat 95, Sun 72
    Grid lines: #F3F4F6 horizontal only
    Legend below: colored dot + label, 12px #6B7280

Right card (1/3 width) — "Emotion Distribution"
  Subtitle: "utterances.emotion — distribution across all org interactions"

  Donut chart (height: 160px):
    Positive:   45%  #10B981
    Neutral:    35%  #6B7280
    Frustrated: 15%  #F59E0B
    Angry:       5%  #EF4444
    2px padding between segments

  Legend 2×2 grid below: 10px circle + label + %,  12px #6B7280

SUBSECTION: Policy Compliance Overview (full width card)
──────────────────────────────────────────────────────────
Title: "Policy Compliance by Category"
Subtitle: "policy_compliance JOIN company_policies — compliance rate per policy_category"

Horizontal bar chart (height: 200px):
  Each bar = one policy category, showing compliance rate %
  Categories with sample rates:
    Greeting Protocol:     94%  #10B981
    Empathy Language:      87%  #3B82F6
    Refund Procedure:      79%  #8B5CF6
    Escalation Handling:   61%  #F59E0B
    Billing Explanation:   48%  #EF4444

  Bar height: 24px, 6px radius, #F3F4F6 bg track
  Color changes based on value: ≥80 green, ≥65 blue, ≥50 amber, else red
  % label right-aligned at end of bar

SUBSECTION: Agent Performance Chart (full width card)
──────────────────────────────────────────────────────

Title: "Agent Performance Breakdown"
Subtitle: "interaction_scores: empathy_score · policy_score · resolution_score per agent (aggregated from agent_performance_snapshots)"

Grouped bar chart (height: 210px):
  5 agent groups: Neha, Rajesh, Priya, Vikram, Amit
  3 bars per agent:
    Empathy:    #3B82F6  (Neha:92, Rajesh:85, Priya:82, Vikram:75, Amit:71)
    Policy:     #10B981  (Neha:90, Rajesh:91, Priya:88, Vikram:78, Amit:74)
    Resolution: #8B5CF6  (Neha:94, Rajesh:88, Priya:90, Vikram:75, Amit:81)

  Bars: 4px top radius, no bottom radius, Y-axis: 60–100
  Grid lines: #F3F4F6
  Legend: colored squares + labels, 12px

SUBSECTION: Bottom Row (2-column, 2:5 ratio)
─────────────────────────────────────────────

Left card — "Agent Leaderboard"
  Icon: Star (16px, #F59E0B) in title
  Subtitle: "agent_performance_snapshots — avg_overall_score, current period"
  Note: "MANAGER ONLY — not visible in agent view"

  5 rows sorted by score desc:
    Each row (40px tall, flex):
      Rank badge 28×28 circle:
        #1: #FEF3C7 bg, #D97706 text, bold
        #2: #F3F4F6 bg, #6B7280 text
        #3+: #F9FAFB bg, #9CA3AF text
      Agent name: 13px SemiBold #111827
      Progress bar (flex-1, 6px, #F3F4F6 bg):
        Fill: emerald ≥85%, blue ≥75%, amber otherwise
      Score: 14px Black, colored same as bar
      Trend: TrendingUp (#10B981) or TrendingDown (#F59E0B)

    Data:
      1. Neha    91%  ↑
      2. Priya   87%  ↑
      3. Rajesh  88%  ↑
      4. Vikram  74%  ↓
      5. Amit    72%  ↓

Right card — "Recent Interactions"
  Subtitle: "interactions JOIN users JOIN interaction_scores — all agents, sorted by overall_score asc (lowest first for review)"

  Scrollable list (max-height 360px). Each interaction row (card-in-card):
    White bg, 1px #E5E7EB border, 10px radius, 14px padding
    Hover: border shifts to #3B82F6, shadow increases
    VIOLATION state: #FECACA border, #FFF5F5 bg

    Left:
      Agent name: 13px SemiBold #111827
      If violated: "Violation" pill — #FEE2E2 bg, #DC2626 text, 11px
      Timestamp · Duration · Language: 12px #9CA3AF

    Right:
      Score: DM Serif Display 22px
        ≥85: #10B981 · ≥75: #3B82F6 · else: #F59E0B
      "✓ Resolved" or "✗ Unresolved": 11px colored

    Bottom chips (always visible):
      Empathy · Policy · Resolution in #F3F4F6, 11px #6B7280

    Sample data:
      Rajesh Kumar   09:14  5:42  ar-EG  88%  ✓ Resolved   [1 violation]
      Neha Sharma    10:02  6:58  ar-EG  91%  ✓ Resolved
      Priya Patel    11:30  4:49  ar-EG  87%  ✓ Resolved   [has_overlap badge]
      Vikram Singh   12:45  8:30  ar-EG  74%  ✗ Unresolved [violation]

──────────────────────────────────────────────────────────────────
SCREEN M-2: SESSION INSPECTOR — RECORDS LIST
──────────────────────────────────────────────────────────────────

This is a standalone screen (separate from Dashboard) reached from the sidebar nav.
It shows ALL interactions across all agents, sorted by overall_score ASCENDING by default
(lowest scores at top, so problematic calls are reviewed first).

TOP CONTROLS ROW:
  Left: page title "Session Inspector" + subtitle "All interactions · sorted by score"
  Right:
    - Search input (200px): placeholder "Search agent, date, ID…", 10px radius
    - Filter dropdown: "All Agents" with ChevronDown
    - Sort toggle pills: "Score ↑" (active, blue) | "Date ↓" | "Duration"
    - Period filter: "This Week" | "This Month" | "All Time" tabs (11px SemiBold)

INTERACTION TABLE (full width card, white bg):
  Column headers (11px uppercase SemiBold #9CA3AF, border-bottom 1px #E5E7EB):
    Agent · Date & Time · Duration · Score · Empathy · Policy · Resolution · Status · Actions

  Each row (48px tall):
    Agent: avatar circle (28×28, accent bg, initials) + name (13px SemiBold)
    Date: 13px #374151 "27 Feb · 09:14"
    Duration: 13px #6B7280 "5:42"
    Score: score badge pill — ≥85 #ECFDF5/#10B981, ≥75 #EFF6FF/#3B82F6, else #FFFBEB/#F59E0B
    Empathy / Policy / Resolution: small score values (12px #374151)
    Status: "✓ Resolved" (#10B981) or "✗ Unresolved" (#EF4444), 12px SemiBold
    Violation badge: "⚠ Violation" pill, #FEF3C7 bg, #92400E text, only if any is_compliant=FALSE
    Actions: "Inspect →" button, 12px, #3B82F6 text, hover underline

  Clicking a row or "Inspect →" navigates to M-3 (Session Inspector Detail).

  Pagination footer: "Showing 1–20 of 342" + prev/next arrows, 12px #6B7280

──────────────────────────────────────────────────────────────────
SCREEN M-3: SESSION INSPECTOR — CALL DETAIL
──────────────────────────────────────────────────────────────────

Reached by clicking any row in M-2. Shows full analysis of one call.

Back button (top): ArrowLeft + "Back to Session Inspector"  #3B82F6, 13px SemiBold

CARD: Call Header
  Two-column layout (info left, score ring right)

  Left:
    Eyebrow: "SESSION INSPECTOR"  10px SemiBold uppercase #9CA3AF, 1px letter-spacing
    Agent name: DM Sans 22px Bold — "Rajesh Kumar"
    Line 2: 13px #6B7280 — "27 Feb 2025  ·  09:14  ·  5:42  ·  ar-EG"
    If has_overlap: amber badge "⚠ Overlap detected"
    ID: 11px #9CA3AF monospace — interaction ID + "completed"

  Right:
    Circular score ring (90px diameter):
      Outer ring: #E5E7EB, 7px stroke
      Score ring: colored by value (≥85 #10B981, ≥75 #3B82F6, else #F59E0B), 7px, rounded linecap
      Center: score % in DM Serif Display 20px

  Bottom 4-column mini-grid (below thin divider):
    Each cell: colored bg, centered text
    Empathy 85%    — #EFF6FF bg, #1D4ED8 value
    Policy 91%     — #ECFDF5 bg, #065F46 value
    Resolution 88% — #F5F3FF bg, #6D28D9 value
    Resp. Time 2.1s— #FFFBEB bg, #92400E value
    Source: interaction_scores row for this interaction

CARD: Transcript
  Title: "Transcript"
  Subtitle: "utterances ordered by sequence_index"

  Scrollable area (max-height 280px):
    AGENT utterance (left-aligned bubble):
      Avatar 28px: #2563EB bg, "A" white bold
      Bubble: #EFF6FF bg, radius 0 12px 12px 12px
      Header: agent name (13px SemiBold #6B7280) + timestamp (12px #9CA3AF) + emotion badge
      Text: 14px #1E3A5F

    CUSTOMER utterance (right-aligned):
      Avatar 28px: #059669 bg, "C" white bold (right side)
      Bubble: #ECFDF5 bg, radius 12px 0 12px 12px

    Emotion badge: pill, 11px SemiBold, icon + label + confidence %
      neutral:    #F1F5F9 bg, #475569 text, Meh icon
      happy:      #ECFDF5 bg, #065F46 text, Smile icon
      angry:      #FEF2F2 bg, #991B1B text, Frown icon
      frustrated: #FFFBEB bg, #92400E text, Frown icon

    Sample data (Arabic text, RTL direction):
      seq 1 | Agent   | "مرحباً، شكراً لاتصالك. كيف يمكنني مساعدتك اليوم؟" | neutral  91%
      seq 2 | Customer| "فاتورتي أعلى بكثير هذا الشهر! لا أفهم هذه الرسوم." | angry    87%
      seq 3 | Agent   | "أفهم قلقك تماماً. دعني أراجع حسابك الآن."           | neutral  88%
      seq 4 | Customer| "حسناً، شكراً جزيلاً."                                | happy    79%

CARD: Emotion Events
  Title: "Emotion Events"
  Subtitle: "emotion_events — AI-detected emotional shifts with LLM justification"
  Note: only events where a significant shift occurred (emotion_delta above threshold)

  Each event card (white bg, 1px #E5E7EB border, 12px radius, 16px padding):

    Row 1:
      Timestamp chip: monospace, #F3F4F6 bg, border #E5E7EB — "4.8s"
      From badge (e.g. neutral pill) → To badge (e.g. angry pill, red)
      Delta: "Δ 0.74"  12px #9CA3AF
      Speaker chip: "customer"  #ECFDF5 bg, #065F46 text

      JUMP BUTTON (right side):
        "▶ Jump to 4.8s"  button — #EFF6FF bg, #2563EB text, #BFDBFE border, Play icon 12px
        Clicking this seeks the audio player to jump_to_seconds
        This is the CORE Session Inspector feature — make it visually prominent

    Row 2 (justification):
      Italic quote in #F9FAFB bg, #6B7280 text, left border 3px #3B82F6
      "Customer escalated over unexpected billing charges. Tone shifted sharply."
      Source: emotion_events.llm_justification

    RLHF feedback row (below thin divider):
      Label: "Was this detection accurate?"  11px #9CA3AF
      Precondition: only active if is_flagged = TRUE on the event, else show flag button first:
        "🚩 Flag as incorrect" — #F9FAFB bg, #6B7280 text, 12px — sets is_flagged = TRUE
      After flagging, show feedback form:
        "✓ Accurate" — #ECFDF5 bg, #065F46 text, ThumbsUp icon
        "✗ Incorrect" — #FEF2F2 bg, #991B1B text, ThumbsDown icon
        If "Incorrect" selected: textarea appears "Describe the correct reason..."
        Submit creates a row in emotion_feedback
      Submitted state: "✓ Feedback recorded — queued for model retraining"  #059669 13px

    Event 2: angry → happy, Δ 0.68, timestamp 19.8s
      Justification: "Customer de-escalated after empathetic acknowledgment from the agent."

CARD: Policy Violations
  Title: "Policy Violations"
  Subtitle: "policy_compliance WHERE is_compliant = FALSE — only violated policies are shown here"

  IMPORTANT: This card ONLY shows violated policies (is_compliant = FALSE).
  If all policies passed, show an empty state: CheckCircle icon + "All policies passed for this call" in #10B981.

  Each violation card (#FEF2F2 bg, #FECACA border, 12px radius, 16px padding):
    Top row:
      XCircle icon (15px, #EF4444) + policy title (14px SemiBold) + category badge (#FEE2E2 bg, #DC2626 text)
      Score bar: #FCA5A5 fill on #F3F4F6 bg, 8px height, with % value right-aligned
    LLM reasoning: 12px #4B5563 — why the agent was non-compliant
    Evidence quote (if present): 12px italic #6B7280 — the transcript quote
    Retrieved policy text (collapsible "View policy text" link): monospace 11px #9CA3AF in dark bg

    RLHF feedback section (below thin divider):
      Label: "Was this verdict correct?"  11px #9CA3AF
      Same flag + feedback button flow as emotion events
      Submits to compliance_feedback table

  Sample violations:
    ✗ Billing Escalation Path  [escalation]  41%
      "Agent did not offer an escalation path when customer raised a billing complaint."
      Evidence: (none recorded)

──────────────────────────────────────────────────────────────────
SCREEN M-4: MANAGER ASSISTANT
──────────────────────────────────────────────────────────────────

Full-height flex column, no content padding. Chat fills the frame.

HEADER:
  Gradient: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%)
  Height: 72px, padding: 0 24px
  Left:
    MessageSquare icon (20px white) in white/20 rounded-xl (40×40)
    Title: "Manager Assistant"  16px Bold white
    Subtitle: "Ask anything about your call centre · voice or text"  12px rgba(white,0.7)
  Right:
    Mic button (voice input toggle): 40×40, white/20 bg, Mic icon white
    Active state: pulsing ring animation, red dot indicator

MESSAGES AREA:
  Background: #F9FAFB
  Flex-1, overflow-y scroll, padding: 20px, gap: 12px

  User message (right-aligned):
    Bubble: #3B82F6 bg, white text, radius 18px 18px 4px 18px, max-width 480px, padding 12px 16px
    If voice message: show waveform visualization bar (small, white bars) above text
    Source: assistant_queries — query_mode can be 'voice' or 'chat'

  AI message (left-aligned):
    Bubble: white bg, #E5E7EB border, shadow, radius 18px 18px 18px 4px, max-width 520px, padding 12px 16px
    Text: 14px #374151
    SQL block (if generated_sql present):
      #0D1117 bg, #A7F3D0 text, radius 8px, padding 10px 14px, Mono 11px, overflow-x scroll
      "SELECT u.name, AVG(s.overall_score*10) as pct FROM users u
       JOIN interactions i ON i.agent_id = u.id
       JOIN interaction_scores s ON s.interaction_id = i.id
       WHERE u.organization_id = $1
       GROUP BY u.name ORDER BY pct DESC"
    Execution time chip: "#F3F4F6 bg, 10px #9CA3AF — "Executed in 142ms"

  EMPTY STATE (no messages yet):
    Centered: MessageSquare in 56×56 #EFF6FF rounded-2xl
    Title: "Ask anything about your call centre"  18px Bold #374151
    Subtitle: "Voice or text — queries are logged to assistant_queries"  13px #9CA3AF

SUGGESTED QUERIES (above input):
  White bg, border-top 1px #E5E7EB, padding: 12px 20px
  Label: "Suggested queries"  11px SemiBold uppercase #9CA3AF
  2×2 grid of chips (8px gap, #EFF6FF bg, #1D4ED8 text, #BFDBFE border, 8px radius, 12px):
    "Show top performing agents this week"
    "List all policy violations today"
    "Which agent has the lowest resolution rate?"
    "Show emotion trends across all calls"

INPUT AREA:
  White bg, border-top 1px #E5E7EB, padding: 16px 20px
  Flex row, gap: 8px

  Mic button: 44×44, #F3F4F6 bg, #E5E7EB border, Mic icon #6B7280
    Active (recording): #FEF2F2 bg, #EF4444 icon, pulsing animation

  Text input: flex-1, #F9FAFB bg, #E5E7EB border, radius 12px, height 44px
    Placeholder: "Ask about scores, violations, agent trends…"
    Focus: #3B82F6 ring

  Send button: 44×44, #2563EB bg, white Send icon, radius 12px

──────────────────────────────────────────────────────────────────
SCREEN M-5: KNOWLEDGE BASE
──────────────────────────────────────────────────────────────────

The admin uses this screen to activate and deactivate policies and FAQ articles used by the RAG system.
Editing policy/FAQ content is NOT shown here (that requires re-embedding; the toggle is the primary action).

INFO BANNER (full width):
  #EFF6FF bg, #BFDBFE border, #1D4ED8 text, Info icon
  "Manage which policies and FAQ articles are active for your organization's RAG evaluation system."
  Subtext: "Deactivating a policy removes it from future call evaluations. Changes take effect on the next call processed."

TWO-COLUMN LAYOUT (equal width):

Left card — "Company Policies"
  Title: BookOpen icon + "Company Policies"
  Subtitle: "company_policies JOIN organization_policies — toggle per-org activation"
  
  Filter row: search input + category filter dropdown (free text, e.g. "greeting", "escalation", "refund")

  Policy list (scrollable, max-height 400px):
    Each policy row (white bg, 1px #E5E7EB border, 10px radius, 16px padding, 12px gap):
      Left:
        Category badge: #F5F3FF bg, #7C3AED text, pill — free text e.g. "greeting", "refund"
        Policy title: 14px SemiBold #111827
        Policy text preview: 12px #6B7280, truncated to 2 lines, "Show more" link
      Right:
        Toggle switch (ON/OFF):
          ON:  #10B981 track, white thumb — organization_policies.is_active = TRUE
          OFF: #D1D5DB track — is_active = FALSE
        Status label below toggle: "Active" (#10B981) or "Inactive" (#9CA3AF), 11px

    Sample policies:
      [greeting]   Professional Greeting Protocol     ON
      [empathy]    Empathy Acknowledgment Language     ON
      [escalation] Billing Escalation Path             ON
      [refund]     Refund Request Handling             OFF
      [compliance] Data Privacy Disclosure             ON

Right card — "FAQ Articles"
  Title: HelpCircle icon + "FAQ Articles"
  Subtitle: "faq_articles JOIN organization_faq_articles — toggle per-org activation"

  Filter row: search input + category filter

  FAQ list (same scrollable card style):
    Each FAQ row:
      Left:
        Category badge: #DBEAFE bg, #1D4ED8 text — free text e.g. "billing", "connectivity"
        Question: 14px SemiBold #111827
        Answer preview: 12px #6B7280, 2 lines truncated
      Right:
        Toggle switch (same ON/OFF style)

    Sample FAQs:
      [billing]      How do I request a refund?                 ON
      [cancellation] How do I cancel my subscription?           ON
      [billing]      Why is my bill higher this month?          ON
      [technical]    How do I reset my account password?        OFF
      [connectivity] My internet is not connecting, what do I do? ON


═══════════════════════════════════════════════════════════════════
SECTION 5 — AGENT VIEW SCREENS
═══════════════════════════════════════════════════════════════════

All agent screens use emerald/green accent (#10B981). Dark sidebar with green active states.
Content is ALWAYS scoped to the logged-in agent's own data only.
No leaderboard. No org-wide KPIs. No other agents visible anywhere.

──────────────────────────────────────────────────────────────────
SCREEN A-1: MY PERFORMANCE (Agent Dashboard)
──────────────────────────────────────────────────────────────────

HERO CARD (full width):
  Gradient: linear-gradient(135deg, #065F46 0%, #0D9488 100%)
  Padding: 28px, border-radius: 16px

  Top row (space-between):
    Left:
      Eyebrow: "MY PERFORMANCE"  11px SemiBold uppercase rgba(white,0.6)
      Name: agent name  DM Serif Display 28px white
      Role: "Agent · VocalMind Corp"  13px rgba(white,0.6)
    Right:
      Score: DM Serif Display 56px white — avg overall score
      Label: "Overall Score"  11px rgba(white,0.5)

  Stats row (below white/20 divider, 3 columns):
    Calls This Week | Team Rank | Resolution Rate
    Each: DM Serif Display 36px white + 11px label rgba(white,0.5)
    Source: agent_performance_snapshots for this agent

4 KPI CARDS (emerald accents):
  - Avg Score:    Star icon,   #ECFDF5 bg, value from avg_overall_score
  - Calls Today:  Phone icon,  #ECFDF5 bg, count from interactions WHERE agent_id = me
  - Resolution:   Target icon, #F5F3FF bg, value from resolution_rate
  - Avg Response: Zap icon,    #FFFBEB bg, value from avg_response_time_seconds

TWO-COLUMN ROW:

Left card — "My Score Breakdown"
  Subtitle: "interaction_scores averaged for my calls — empathy, policy, resolution"

  3 labeled progress bars:
    Empathy Score:    [bar, value]  #3B82F6 fill
    Policy Adherence: [bar, value]  #10B981 fill
    Resolution:       [bar, value]  #8B5CF6 fill

    Bar: 10px height, #F3F4F6 bg, rounded, 999px radius
    Label left (13px) + % right (13px SemiBold)
    Source: agent_performance_snapshots.avg_empathy_score / avg_policy_score / avg_resolution_score

Right card — "My Weekly Trend"
  Subtitle: "interaction_scores for my calls this week — overall score trend"

  Line chart (height 190px):
    Single line: #10B981, 3px, dots (6px, white fill + green stroke)
    X-axis: Mon–Sun, Y-axis: 7.0–10.0
    Data: Mon 8.4, Tue 8.9, Wed 8.2, Thu 9.1, Fri 8.8, Sat 8.6, Sun 8.3

CARD: My Recent Calls
  Subtitle: "interactions WHERE agent_id = [me] — personal calls only, sorted by date desc"
  Note: agents CANNOT see other agents' calls

  Same card-list style as manager dashboard Recent Interactions BUT:
    No agent name column (all rows are the logged-in agent)
    FLAGGED state: amber (#FFFBEB bg, #FDE68A border) labelled "Review needed" (not "Violation" — agent-friendly framing)

  3 call rows:
    09:14  5:42  ar-EG  88%  ✓ Resolved  [Review needed]
    14:10  4:27  ar-EG  90%  ✓ Resolved
    15:55  3:18  ar-EG  83%  ✓ Resolved

──────────────────────────────────────────────────────────────────
SCREEN A-2: MY CALLS — CALL DETAIL
──────────────────────────────────────────────────────────────────

Reached by clicking a call row in My Performance or My Calls nav item.
This is the agent's equivalent of the manager's Session Inspector — personal scope only.

Back button: ArrowLeft + "Back to My Calls"  #10B981

CALL HEADER CARD:
  Same structure as M-3 Call Header but:
    Eyebrow: "CALL DETAIL"
    Title: date + time instead of agent name — "27 Feb 2025 · 09:14"
    Score ring: emerald accent
    Same 4-column mini-grid (empathy, policy, resolution, response time)

COACHING POINTS CARD (only shown if any policy was violated):
  Background: #FFFBEB, border: #FDE68A, radius: 14px, padding: 20px
  Title: Target icon (15px #92400E) + "Coaching Points"  14px SemiBold #92400E
  Subtitle: "Areas to focus on — sourced from policy_compliance WHERE is_compliant = FALSE"

  Each coaching item (white bg, #FDE68A border, 10px radius, 14px padding):
    Policy title: 14px SemiBold #111827
    LLM reasoning: 12px #4B5563
    Score note: "Score: 41% — target 80%+"  12px #D97706 SemiBold

  This is the AGENT-FACING equivalent of the Policy Violations card.
  It is framed constructively (coaching, not violations).

TRANSCRIPT CARD: Same as M-3 transcript card but:
  Agent bubble: #ECFDF5 bg (green, not blue)
  Agent avatar: #10B981
  Label "Me" in bubble header instead of agent name
  Customer avatar: #E5E7EB bg, #6B7280 text "C"
  Arabic utterances, RTL direction

CUSTOMER EMOTION JOURNEY CARD:
  Title: "Customer Emotion Journey"
  Subtitle: "emotion_events — how customer sentiment changed during this call"
  Agent-friendly framing: "your impact on the customer experience"

  2 event cards:
    Event 1 (neutral→angry): #FEF2F2 bg, #FECACA border
    Event 2 (angry→happy):   #ECFDF5 bg, #A7F3D0 border

  Each: timestamp chip + "Customer mood:" label + from/to emotion badges + justification quote
  JUMP BUTTON is present here too (agents can also jump to the moment)
  NO RLHF feedback buttons — agents do not correct model outputs in this view


═══════════════════════════════════════════════════════════════════
SECTION 6 — INTERACTIVE STATES & MICRO-DETAILS
═══════════════════════════════════════════════════════════════════

HOVER STATES:
  - Interaction rows: border → accent color, shadow increases
  - Nav items: text → white, bg → #1F2937
  - Buttons: bg darkens 10%, slight shadow
  - Chart points: tooltip with exact value
  - Knowledge Base toggle: thumb animates smoothly

ACTIVE / SELECTED:
  - Nav item: accent bg, white text
  - Jump button: background darkens, audio player seeks

FOCUS STATES:
  - Inputs: 2px ring in role accent color
  - Buttons: visible focus ring

EMPTY STATES:
  - No violations in Session Inspector: CheckCircle + "All policies passed" green text
  - No calls in agent view: Phone icon + "No calls processed yet"
  - Empty assistant chat: centered icon + headline + suggested queries

BADGE STYLES (all pills, 20px radius, 11px SemiBold):
  Emotion: neutral / happy / angry / frustrated — 5 defined color states
  Status: resolved (green) / unresolved (red)
  Violation: amber warning
  Processing: blue pending / green completed / red failed
  Category (policies/FAQs): free-text label, purple or blue background

TOGGLE SWITCHES (Knowledge Base):
  Width: 44px, height: 24px
  ON: role-accent color track, white 20px circle thumb, smooth 0.2s slide
  OFF: #D1D5DB track


═══════════════════════════════════════════════════════════════════
SECTION 7 — SCREEN INVENTORY
═══════════════════════════════════════════════════════════════════

MANAGER SCREENS (blue accent):
  M-0  Sidebar + Top Bar (shared shell component — show as master frame)
  M-1  Dashboard
  M-2  Session Inspector — Records List
  M-3  Session Inspector — Call Detail (call: Rajesh Kumar, 27 Feb)
  M-4  Manager Assistant (empty state + active conversation state)
  M-5  Knowledge Base

AGENT SCREENS (emerald accent):
  A-0  Sidebar + Top Bar (agent shell)
  A-1  My Performance (dashboard)
  A-2  My Calls — Call Detail (same call as M-3 but agent-scoped view)

TOTAL SCREENS: 9 (plus component page)

Recommended Figma file structure:
  Page 1: Design System (colors, typography, components)
  Page 2: Manager Flows (M-0 through M-5)
  Page 3: Agent Flows (A-0 through A-2)
  Page 4: Component Library


═══════════════════════════════════════════════════════════════════
SECTION 8 — DATA SOURCE ANNOTATIONS
═══════════════════════════════════════════════════════════════════

Annotate each screen element with its source table from the schema.
Key mappings:

  Dashboard KPI cards:
    Avg Score          → interaction_scores.overall_score (org avg)
    Calls Processed    → interactions.processing_status = 'completed' count
    Resolution Rate    → interaction_scores.was_resolved proportion
    Policy Violations  → policy_compliance.is_compliant = FALSE distinct interactions

  Dashboard charts:
    Weekly Score Trend → interaction_scores grouped by interactions.interaction_date
    Emotion Donut      → utterances.emotion grouped count
    Policy Compliance  → policy_compliance JOIN company_policies, rate per category
    Agent Breakdown    → agent_performance_snapshots (avg scores per agent)

  Leaderboard:
    → agent_performance_snapshots (MANAGER ONLY — never shown to agents)

  Session Inspector list:
    → interactions JOIN users JOIN interaction_scores ORDER BY overall_score ASC

  Session Inspector detail:
    Header             → interactions + interaction_scores
    Transcript         → utterances ORDER BY sequence_index
    Emotion Events     → emotion_events WHERE interaction_id = X
    RLHF feedback      → emotion_feedback (manager creates row here)
    Policy Violations  → policy_compliance WHERE is_compliant = FALSE (only violations shown)
    Compliance RLHF    → compliance_feedback (manager creates row here)

  Manager Assistant:
    Queries logged     → assistant_queries (query_mode: voice | chat, audio_input_path for voice)
    Response SQL       → assistant_queries.generated_sql

  Knowledge Base:
    Policies list      → company_policies JOIN organization_policies (is_active toggle)
    FAQ list           → faq_articles JOIN organization_faq_articles (is_active toggle)

  Agent Dashboard:
    All KPIs + charts  → agent_performance_snapshots WHERE agent_id = logged-in user
    Call list          → interactions WHERE agent_id = logged-in user

  Agent Call Detail:
    Same as M-3 but scoped to agent's own interaction only
    Coaching Points    → policy_compliance WHERE is_compliant = FALSE (agent framing)
    Emotion Journey    → emotion_events (no RLHF buttons in agent view)

ACCESS CONTROL RULES (annotate on each element):
  ● Leaderboard              → MANAGER ONLY
  ● Org-wide KPI cards       → MANAGER ONLY
  ● All agents' interactions → MANAGER ONLY
  ● Manager Assistant        → MANAGER ONLY
  ● Knowledge Base           → MANAGER ONLY
  ● Policy Violations card   → MANAGER ONLY (agent sees "Coaching Points" instead)
  ● RLHF feedback buttons    → MANAGER ONLY (agent cannot correct model outputs)
  ● Jump-to-audio button     → BOTH roles
  ● Agent's own call detail  → BOTH roles (different framing per role)


═══════════════════════════════════════════════════════════════════
SECTION 9 — FIGMA-SPECIFIC INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

1. Canvas frame size: 1440 × 900 for all desktop screens.

2. Use Auto Layout on ALL components — no fixed-position elements inside cards.

3. Shared components to build in the component library:
   - KPICard (variants: blue / green / amber / red / purple)
   - NavItem (variants: default / hover / active; sizes: expanded / collapsed)
   - EmotionBadge (variants: neutral / happy / angry / frustrated / sad)
   - PolicyViolationRow (states: default / rlhf-submitted)
   - EmotionEventRow (states: default / flagged / feedback-submitted)
   - InteractionRow (variants: normal / violation; roles: manager / agent)
   - ScoreRing (90px, color driven by score value)
   - KBToggleRow (types: policy / faq; states: active / inactive)
   - JumpToAudioButton (states: idle / active)
   - FeedbackButtons (states: idle / flagged / submitted)
   - VoiceMicButton (states: idle / recording)

4. Color variables (not hardcoded hex). Variable collections:
   - Global/Colors
   - Manager/Accent
   - Agent/Accent
   Enables full role-theme swap.

5. Prototype connections:
   - Clicking interaction row (Dashboard or Session Inspector list) → Session Inspector detail
   - Back button → previous screen
   - "Inspect →" in table row → M-3 detail
   - Send / Mic → AI response state in Manager Assistant
   - Flag button → shows feedback form
   - Feedback submit → submitted confirmation state
   - Knowledge Base toggle → ON/OFF state swap
   - Jump button → (annotate: seeks audio player to jump_to_seconds)
   - Nav items → navigate to correct screen

6. Chart components: use realistic data distributions with labeled axes. No placeholder boxes.

7. Arabic transcript text: set text direction RTL, right-aligned within bubbles.

8. The Jump-to-Audio button in Session Inspector is a KEY product feature.
   Make it visually distinct, easy to click, and clearly linked to the emotion event timestamp.
   It should feel like a "go there now" action — not a subtle link.

9. The Session Inspector records list (M-2) sorts by overall_score ASC by default.
   Lowest-scoring calls appear first — this is intentional so managers review problems first.
   The sort controls must be clearly visible and the current sort state clearly indicated.

10. Responsive breakpoints (optional bonus):
    Desktop: 1440px (primary)
    Tablet:  768px (sidebar collapses to icons)
    Mobile:  390px (sidebar hidden, bottom nav bar)