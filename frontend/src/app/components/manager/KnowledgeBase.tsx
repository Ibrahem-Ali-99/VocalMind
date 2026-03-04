import { useState } from "react";
import { BookOpen, HelpCircle, Info, Search } from "lucide-react";
import { Switch } from "../../components/ui/switch";
import { mockPolicies, mockFAQs } from "../../data/mockData";

export function KnowledgeBase() {
  const [policies, setPolicies] = useState(mockPolicies);
  const [faqs, setFaqs] = useState(mockFAQs);
  const [policySearch, setPolicySearch] = useState("");
  const [faqSearch, setFaqSearch] = useState("");

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
      <div className="bg-[#EFF6FF] border border-[#BFDBFE] rounded-xl p-4 flex items-start gap-3">
        <Info className="w-5 h-5 text-[#1D4ED8] flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-[14px] text-[#1D4ED8] font-medium mb-1">
            Manage which policies and FAQ articles are active for your organization's RAG evaluation system.
          </p>
          <p className="text-[13px] text-[#3B82F6]">
            Deactivating a policy removes it from future call evaluations. Changes take effect on the next call processed.
          </p>
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-2 gap-6">
        {/* Company Policies */}
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <BookOpen className="w-[18px] h-[18px] text-[#3B82F6]" />
            <h3 className="text-[16px] font-semibold text-[#111827]">
              Company Policies
            </h3>
          </div>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            company_policies JOIN organization_policies — toggle per-org activation
          </p>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9CA3AF]" />
            <input
              type="text"
              placeholder="Search policies..."
              value={policySearch}
              onChange={(e) => setPolicySearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded-[10px] text-[13px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]"
            />
          </div>

          {/* Policy List */}
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {filteredPolicies.map((policy) => (
              <div
                key={policy.id}
                className="border border-[#E5E7EB] rounded-[10px] p-4 hover:border-[#3B82F6] transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0 mr-3">
                    <span className="inline-block px-2 py-0.5 bg-[#F5F3FF] text-[#7C3AED] rounded-full text-[11px] font-medium mb-2">
                      {policy.category}
                    </span>
                    <h4 className="text-[14px] font-semibold text-[#111827] mb-1">
                      {policy.title}
                    </h4>
                    <p className="text-[12px] text-[#6B7280] line-clamp-2">
                      {policy.preview}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2 flex-shrink-0">
                    <Switch checked={policy.isActive} onCheckedChange={() => togglePolicy(policy.id)} />
                    <span
                      className={`text-[11px] font-medium ${
                        policy.isActive ? "text-[#10B981]" : "text-[#9CA3AF]"
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
        <div className="bg-white rounded-[14px] border border-[#E5E7EB] p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <HelpCircle className="w-[18px] h-[18px] text-[#3B82F6]" />
            <h3 className="text-[16px] font-semibold text-[#111827]">
              FAQ Articles
            </h3>
          </div>
          <p className="text-[11px] italic text-[#9CA3AF] mb-4">
            faq_articles JOIN organization_faq_articles — toggle per-org activation
          </p>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#9CA3AF]" />
            <input
              type="text"
              placeholder="Search FAQs..."
              value={faqSearch}
              onChange={(e) => setFaqSearch(e.target.value)}
              className="w-full h-10 pl-9 pr-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded-[10px] text-[13px] focus:outline-none focus:ring-2 focus:ring-[#3B82F6]"
            />
          </div>

          {/* FAQ List */}
          <div className="space-y-3 max-h-[400px] overflow-y-auto">
            {filteredFaqs.map((faq) => (
              <div
                key={faq.id}
                className="border border-[#E5E7EB] rounded-[10px] p-4 hover:border-[#3B82F6] transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0 mr-3">
                    <span className="inline-block px-2 py-0.5 bg-[#DBEAFE] text-[#1D4ED8] rounded-full text-[11px] font-medium mb-2">
                      {faq.category}
                    </span>
                    <h4 className="text-[14px] font-semibold text-[#111827] mb-1">
                      {faq.question}
                    </h4>
                    <p className="text-[12px] text-[#6B7280] line-clamp-2">
                      {faq.preview}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2 flex-shrink-0">
                    <Switch checked={faq.isActive} onCheckedChange={() => toggleFaq(faq.id)} />
                    <span
                      className={`text-[11px] font-medium ${
                        faq.isActive ? "text-[#10B981]" : "text-[#9CA3AF]"
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
