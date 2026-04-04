import { useState, useEffect } from "react";
import { BookOpen, HelpCircle, Info, Search, Loader2, AlertTriangle } from "lucide-react";
import { Switch } from "../../components/ui/switch";
import { getPolicies, getFaqs, type PolicyData, type FAQData } from "../../services/api";

export function KnowledgeBase() {
  const [policies, setPolicies] = useState<PolicyData[]>([]);
  const [faqs, setFaqs] = useState<FAQData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [policySearch, setPolicySearch] = useState("");
  const [faqSearch, setFaqSearch] = useState("");

  useEffect(() => {
    Promise.all([getPolicies(), getFaqs()])
      .then(([p, f]) => {
        setPolicies(p);
        setFaqs(f);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
        <span className="ml-3 text-muted-foreground text-sm">Loading knowledge base...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-warning mx-auto mb-3" />
          <p className="text-foreground text-sm">Failed to load knowledge base</p>
          <p className="text-muted-foreground/80 text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const togglePolicy = (id: string) => {
    setPolicies(policies.map((p) => (p.id === id ? { ...p, isActive: !p.isActive } : p)));
  };

  const toggleFaq = (id: string) => {
    setFaqs(faqs.map((f) => (f.id === id ? { ...f, isActive: !f.isActive } : f)));
  };

  const filteredPolicies = policies.filter(
    (p) =>
      p.title.toLowerCase().includes(policySearch.toLowerCase()) ||
      p.category.toLowerCase().includes(policySearch.toLowerCase())
  );

  const filteredFaqs = faqs.filter(
    (f) =>
      f.question.toLowerCase().includes(faqSearch.toLowerCase()) ||
      f.category.toLowerCase().includes(faqSearch.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-1">
        <h2 className="text-[24px] font-bold text-foreground">Knowledge Base</h2>
        <p className="text-[13px] text-muted-foreground font-medium">Manage call evaluation policies and FAQ criteria</p>
      </div>

      <div className="bg-primary/5 border border-primary/10 rounded-xl p-4 flex items-start gap-3">
        <Info className="w-5 h-5 text-primary shrink-0 mt-0.5" />
        <div className="text-[13px] text-primary/80 leading-relaxed font-medium">
          Manage which policies and FAQ articles are active for your organization's RAG evaluation system. Deactivating a policy removes it from future call evaluations.
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Policies Section */}
        <div className="bg-card rounded-[14px] border border-border p-5">
          <div className="flex items-center gap-2 mb-1">
            <BookOpen className="w-[18px] h-[18px] text-primary" />
            <h3 className="text-[20px] font-bold text-foreground">Company Policies</h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">company_policies JOIN organization_policies</p>
          
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search policies..."
              value={policySearch}
              onChange={(e) => setPolicySearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-muted/20 border border-border rounded-xl text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
          </div>

          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
            {filteredPolicies.map((p) => (
              <div key={p.id} className="p-4 border border-border rounded-xl hover:bg-muted/5 transition-colors">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <span className="inline-block px-2 py-0.5 bg-primary/10 text-primary rounded-full text-[10px] font-bold uppercase tracking-wider mb-2">
                      {p.category}
                    </span>
                    <h4 className="text-[14px] font-bold text-foreground mb-1 truncate">{p.title}</h4>
                    <p className="text-[12px] text-muted-foreground line-clamp-2">{p.preview}</p>
                  </div>
                  <div className="flex flex-col items-center gap-2 shrink-0">
                    <Switch
                      checked={p.isActive}
                      data-cy={`policy-toggle-${p.id}`}
                      onCheckedChange={() => togglePolicy(p.id)}
                    />
                    <span className={`text-[10px] font-bold ${p.isActive ? "text-success" : "text-muted-foreground"}`}>
                      {p.isActive ? "ACTIVE" : "INACTIVE"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* FAQs Section */}
        <div className="bg-card rounded-[14px] border border-border p-5">
          <div className="flex items-center gap-2 mb-1">
            <HelpCircle className="w-[18px] h-[18px] text-primary" />
            <h3 className="text-[20px] font-bold text-foreground">FAQ Articles</h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">faq_articles JOIN organization_faq_articles</p>

          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search FAQs..."
              value={faqSearch}
              onChange={(e) => setFaqSearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-muted/20 border border-border rounded-xl text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
          </div>

          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
            {filteredFaqs.map((f) => (
              <div key={f.id} className="p-4 border border-border rounded-xl hover:bg-muted/5 transition-colors">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <span className="inline-block px-2 py-0.5 bg-success/10 text-success rounded-full text-[10px] font-bold uppercase tracking-wider mb-2">
                      {f.category}
                    </span>
                    <h4 className="text-[14px] font-bold text-foreground mb-1 truncate">{f.question}</h4>
                    <p className="text-[12px] text-muted-foreground line-clamp-2">{f.preview}</p>
                  </div>
                  <div className="flex flex-col items-center gap-2 shrink-0">
                    <Switch
                      checked={f.isActive}
                      data-cy={`faq-toggle-${f.id}`}
                      onCheckedChange={() => toggleFaq(f.id)}
                    />
                    <span className={`text-[10px] font-bold ${f.isActive ? "text-success" : "text-muted-foreground"}`}>
                      {f.isActive ? "ACTIVE" : "INACTIVE"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
