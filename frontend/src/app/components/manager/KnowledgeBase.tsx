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
        <span className="ml-3 text-[#6B7280] text-sm">Loading knowledge base...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertTriangle className="w-10 h-10 text-[#F59E0B] mx-auto mb-3" />
          <p className="text-[#6B7280] text-sm">Failed to load knowledge base</p>
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
      {/* Info Banner */}
      <div className="bg-primary/10 border border-primary/20 rounded-xl p-4 flex items-start gap-3">
        <Info className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-[14px] text-primary font-medium mb-1">
            Manage which policies and FAQ articles are active for your organization's RAG evaluation system.
          </p>
          <p className="text-[13px] text-primary/80">
            Deactivating a policy removes it from future call evaluations. Changes take effect on the next call processed.
          </p>
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-6">
        {/* Company Policies */}
        <div className="bg-card rounded-[14px] border border-border p-5">
          <div className="flex items-center gap-2 mb-1">
            <BookOpen className="w-[18px] h-[18px] text-primary" />
            <h3 className="text-[16px] font-semibold text-foreground">
              Company Policies
            </h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            company_policies JOIN organization_policies — toggle per-org activation
          </p>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-label-foreground" />
            <input
              type="text"
              placeholder="Search..."
              value={policySearch}
              onChange={(e) => setPolicySearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-input border border-border rounded-[10px] text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all shadow-inner"
            />
          </div>

          {/* Policy List */}
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {filteredPolicies.map((policy) => (
              <div
                key={policy.id}
                className="border border-border rounded-[10px] p-4 hover:border-primary transition-colors hover:bg-muted/50"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0 mr-3">
                    <span className="inline-block px-2 py-0.5 bg-primary/10 text-primary rounded-full text-[11px] font-medium mb-2">
                      {policy.category}
                    </span>
            <h4 className="text-[14px] font-semibold text-foreground mb-1">
                      {policy.title}
                    </h4>
                    <p className="text-[12px] text-muted-foreground line-clamp-2">
                      {policy.preview}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2 flex-shrink-0">
                    <Switch checked={policy.isActive} onCheckedChange={() => togglePolicy(policy.id)} />
                    <span
                      className={`text-[11px] font-medium ${
                        policy.isActive ? "text-success" : "text-muted-foreground"
                      }`}
                    >
                      {policy.isActive ? "Active" : "Inactive"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* FAQ Articles */}
        <div className="bg-card rounded-[14px] border border-border p-5">
          <div className="flex items-center gap-2 mb-1">
            <HelpCircle className="w-[18px] h-[18px] text-primary" />
            <h3 className="text-[16px] font-semibold text-foreground">
              FAQ Articles
            </h3>
          </div>
          <p className="text-[11px] italic text-muted-foreground mb-4">
            faq_articles JOIN organization_faq_articles — toggle per-org activation
          </p>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-label-foreground" />
            <input
              type="text"
              placeholder="Search..."
              value={faqSearch}
              onChange={(e) => setFaqSearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-input border border-border rounded-[10px] text-[13px] focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all shadow-inner"
            />
          </div>

          {/* FAQ List */}
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {filteredFaqs.map((faq) => (
              <div
                key={faq.id}
                className="border border-border rounded-[10px] p-4 hover:border-primary transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0 mr-3">
                    <span className="inline-block px-2 py-0.5 bg-success/10 text-success rounded-full text-[11px] font-medium mb-2">
                      {faq.category}
                    </span>
            <h4 className="text-[14px] font-semibold text-foreground mb-1">
                      {faq.question}
                    </h4>
                    <p className="text-[12px] text-muted-foreground line-clamp-2">
                      {faq.preview}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2 flex-shrink-0">
                    <Switch checked={faq.isActive} onCheckedChange={() => toggleFaq(faq.id)} />
                    <span
                      className={`text-[11px] font-medium ${
                        faq.isActive ? "text-success" : "text-muted-foreground"
                      }`}
                    >
                      {faq.isActive ? "Active" : "Inactive"}
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
