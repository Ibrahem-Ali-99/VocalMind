import { useState, useEffect } from "react";
import { 
  BookOpen, 
  HelpCircle, 
  Search, 
  Loader2, 
  Plus, 
  BarChart3, 
  ArrowRight,
  ShieldCheck,
  Zap,
  Trash2,
  AlertTriangle,
  FileText,
  Upload
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Switch } from "../../components/ui/switch";
import { Badge } from "../../components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter,
  DialogDescription
} from "../../components/ui/dialog";
import { 
  getPolicies, 
  getFaqs, 
  togglePolicy,
  toggleFaq,
  deletePolicy,
  deleteFaq,
  uploadPolicyDocument,
  replacePolicyDocument,
  uploadFaqDocument,
  replaceFaqDocument,
  type PolicyData, 
  type FAQData 
} from "../../services/api";
import { toast } from "sonner";

export function KnowledgeBase() {
  const [policies, setPolicies] = useState<PolicyData[]>([]);
  const [faqs, setFaqs] = useState<FAQData[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("policies");

  // Search States
  const [policySearch, setPolicySearch] = useState("");
  const [faqSearch, setFaqSearch] = useState("");

  // Modal States
  const [isPolicyModalOpen, setIsPolicyModalOpen] = useState(false);
  const [isFaqModalOpen, setIsFaqModalOpen] = useState(false);

  // Form States
  const [policyForm, setPolicyForm] = useState({ title: "", category: "" });
  const [faqForm, setFaqForm] = useState({ question: "", category: "" });
  const [policyFile, setPolicyFile] = useState<File | null>(null);
  const [faqFile, setFaqFile] = useState<File | null>(null);
  const [policyTargetId, setPolicyTargetId] = useState<string | null>(null);
  const [faqTargetId, setFaqTargetId] = useState<string | null>(null);

  // Detail & Delete states
  const [selectedDoc, setSelectedDoc] = useState<PolicyData | FAQData | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [docToDelete, setDocToDelete] = useState<{ id: string, type: 'policy' | 'faq' } | null>(null);
  const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);

  const openPolicyModal = (doc?: PolicyData) => {
    setPolicyTargetId(doc?.id ?? null);
    setPolicyForm({
      title: doc?.title ?? "",
      category: doc?.category ?? "",
    });
    setPolicyFile(null);
    setIsPolicyModalOpen(true);
  };

  const openFaqModal = (doc?: FAQData) => {
    setFaqTargetId(doc?.id ?? null);
    setFaqForm({
      question: doc?.question ?? "",
      category: doc?.category ?? "",
    });
    setFaqFile(null);
    setIsFaqModalOpen(true);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const isPolicyDoc = (doc: PolicyData | FAQData | null): doc is PolicyData => {
    return doc?.documentType === "policy";
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      const [p, f] = await Promise.all([getPolicies(), getFaqs()]);
      setPolicies(p);
      setFaqs(f);
    } catch (err: any) {
      toast.error("Failed to load knowledge base: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- Handlers ---
  
  const handleTogglePolicy = async (id: string) => {
    try {
      const { isActive } = await togglePolicy(id);
      setPolicies(policies.map(p => p.id === id ? { ...p, isActive } : p));
      toast.success(`Policy ${isActive ? 'activated' : 'deactivated'}`);
    } catch (err: any) {
      toast.error("Failed to toggle policy");
    }
  };

  const handleToggleFaq = async (id: string) => {
    try {
      const { isActive } = await toggleFaq(id);
      setFaqs(faqs.map(f => f.id === id ? { ...f, isActive } : f));
      toast.success(`FAQ ${isActive ? 'activated' : 'deactivated'}`);
    } catch (err: any) {
      toast.error("Failed to toggle FAQ");
    }
  };

  const savePolicy = async () => {
    if (!policyFile) {
      toast.error("Choose a PDF for the guideline upload");
      return;
    }

    try {
      if (policyTargetId) {
        await replacePolicyDocument(policyTargetId, {
          title: policyForm.title,
          category: policyForm.category,
          file: policyFile,
        });
        toast.success("Policy replaced with newer PDF");
      } else {
        await uploadPolicyDocument({
          title: policyForm.title,
          category: policyForm.category,
          file: policyFile,
        });
        toast.success("Policy uploaded");
      }
      setIsPolicyModalOpen(false);
      fetchData();
    } catch (err: any) {
      toast.error("Failed to save policy");
    }
  };

  const saveFaq = async () => {
    if (!faqFile) {
      toast.error("Choose a PDF for the FAQ upload");
      return;
    }

    try {
      if (faqTargetId) {
        await replaceFaqDocument(faqTargetId, {
          question: faqForm.question,
          category: faqForm.category,
          file: faqFile,
        });
        toast.success("FAQ replaced with newer PDF");
      } else {
        await uploadFaqDocument({
          question: faqForm.question,
          category: faqForm.category,
          file: faqFile,
        });
        toast.success("FAQ uploaded");
      }
      setIsFaqModalOpen(false);
      fetchData();
    } catch (err: any) {
      toast.error("Failed to save FAQ");
    }
  };

  const handleDelete = async () => {
    if (!docToDelete) return;
    try {
      if (docToDelete.type === 'policy') {
        await deletePolicy(docToDelete.id);
      } else {
        await deleteFaq(docToDelete.id);
      }
      toast.success(`${docToDelete.type === 'policy' ? 'Policy' : 'FAQ'} deleted successfully`);
      setIsDeleteConfirmOpen(false);
      fetchData();
    } catch (err: any) {
      toast.error("Failed to delete item");
    }
  };

  const filteredPolicies = policies.filter(p => 
    p.title.toLowerCase().includes(policySearch.toLowerCase()) || 
    p.category.toLowerCase().includes(policySearch.toLowerCase())
  );

  const filteredFaqs = faqs.filter(f => 
    f.question.toLowerCase().includes(faqSearch.toLowerCase()) || 
    f.category.toLowerCase().includes(faqSearch.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[500px]">
        <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
        <p className="text-muted-foreground animate-pulse font-medium">Synchronizing knowledge base...</p>
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-3xl font-black tracking-tight text-foreground bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">
            Knowledge Engine
          </h2>
          <p className="text-sm text-muted-foreground font-medium max-w-md">
            Define the criteria and behavioral guardrails for your AI evaluation pipeline.
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button 
            variant="outline" 
            className="rounded-xl border-primary/20 bg-primary/5 hover:bg-primary/10 text-primary transition-all duration-300"
            onClick={() => fetchData()}
          >
            Refresh Sync
          </Button>
          <Button 
            className="rounded-xl shadow-lg shadow-primary/20 gap-2 font-bold px-6"
            onClick={() => {
              if (activeTab === "policies") {
                openPolicyModal();
              } else {
                openFaqModal();
              }
            }}
          >
            <Plus className="w-4 h-4" />
            Upload {activeTab === "policies" ? "Guideline PDF" : "FAQ PDF"}
          </Button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card/40 backdrop-blur-sm border border-border p-4 rounded-2xl flex items-center justify-between group hover:border-primary/30 transition-all duration-300 shadow-sm">
          <div className="space-y-1">
            <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Active Policies</p>
            <p className="text-2xl font-black">{policies.filter(p => p.isActive).length}</p>
          </div>
          <div className="p-3 bg-primary/10 rounded-xl text-primary group-hover:scale-110 transition-transform">
            <ShieldCheck className="w-5 h-5" />
          </div>
        </div>
        <div className="bg-card/40 backdrop-blur-sm border border-border p-4 rounded-2xl flex items-center justify-between group hover:border-success/30 transition-all duration-300 shadow-sm">
          <div className="space-y-1">
            <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">SOP Coverage</p>
            <p className="text-2xl font-black">{faqs.length} Documents</p>
          </div>
          <div className="p-3 bg-success/10 rounded-xl text-success group-hover:scale-110 transition-transform">
            <HelpCircle className="w-5 h-5" />
          </div>
        </div>
        <div className="bg-card/40 backdrop-blur-sm border border-border p-4 rounded-2xl flex items-center justify-between group hover:border-warning/30 transition-all duration-300 shadow-sm">
          <div className="space-y-1">
            <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Evaluation Hits</p>
            <p className="text-2xl font-black">
              {policies.reduce((acc, curr) => acc + (curr.usageCount || 0), 0)}
            </p>
          </div>
          <div className="p-3 bg-warning/10 rounded-xl text-warning group-hover:scale-110 transition-transform">
            <Zap className="w-5 h-5" />
          </div>
        </div>
      </div>

      <Tabs defaultValue="policies" onValueChange={setActiveTab} className="space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <TabsList className="bg-muted/50 border border-border p-1 h-12 rounded-2xl backdrop-blur-md">
            <TabsTrigger value="policies" className="rounded-xl px-8 font-bold data-[state=active]:shadow-lg">
              <BookOpen className="w-4 h-4 mr-2" />
              Guidelines
            </TabsTrigger>
            <TabsTrigger value="faqs" className="rounded-xl px-8 font-bold data-[state=active]:shadow-lg">
              <HelpCircle className="w-4 h-4 mr-2" />
              SOP & Knowledge
            </TabsTrigger>
          </TabsList>

          <div className="relative w-full md:w-80 group">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
            <Input
              placeholder={`Search ${activeTab === "policies" ? "guidelines" : "SOP & knowledge"}...`}
              value={activeTab === "policies" ? policySearch : faqSearch}
              onChange={(e) => activeTab === "policies" ? setPolicySearch(e.target.value) : setFaqSearch(e.target.value)}
              className="pl-10 h-12 rounded-2xl border-border bg-card/50 backdrop-blur-sm focus-visible:ring-primary/40 focus-visible:border-primary transition-all shadow-sm"
            />
          </div>
        </div>

        <TabsContent value="policies" className="animate-in fade-in slide-in-from-bottom-2 duration-400 outline-none">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPolicies.map((p) => (
              <div 
                key={p.id} 
                className="group relative bg-card hover:bg-card/80 border border-border hover:border-primary/30 rounded-[24px] p-6 transition-all duration-300 shadow-sm hover:shadow-xl hover:shadow-primary/5 flex flex-col justify-between overflow-hidden"
              >
                {/* Visual Accent */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full -mr-16 -mt-16 blur-3xl transition-all group-hover:bg-primary/10" />
                
                <div className="relative space-y-4">
                  <div className="flex items-start justify-between">
                    <Badge variant="secondary" className="bg-primary/5 text-primary border-primary/10 rounded-lg px-2.5 py-1 text-[10px] font-black uppercase tracking-wider">
                      {p.category}
                    </Badge>
                    <Switch checked={p.isActive} onCheckedChange={() => handleTogglePolicy(p.id)} />
                  </div>
                  
                  <div>
                    <h4 className="text-[17px] font-black text-foreground group-hover:text-primary transition-colors leading-tight mb-2 text-ellipsis overflow-hidden">
                      {p.title}
                    </h4>
                    <p className="text-[13px] text-muted-foreground/80 leading-relaxed line-clamp-3 font-medium h-[60px]">
                      {p.preview}
                    </p>
                  </div>
                </div>

                <div className="relative pt-6 mt-6 border-t border-border flex items-center justify-between">
                  <div className="flex items-center gap-2 text-muted-foreground/60">
                    <BarChart3 className="w-3.5 h-3.5" />
                    <span className="text-[11px] font-bold uppercase tracking-widest">{p.usageCount || 0} HITS</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <Button 
                      variant="ghost" 
                      className="text-[11px] font-bold text-primary hover:bg-primary/5 group/btn"
                      onClick={() => openPolicyModal(p)}
                    >
                      Replace
                      <Upload className="w-3.5 h-3.5 ml-1" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="rounded-xl hover:bg-destructive/10 hover:text-destructive transition-colors h-8 w-8"
                      onClick={() => {
                        setDocToDelete({ id: p.id, type: 'policy' });
                        setIsDeleteConfirmOpen(true);
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      className="text-[11px] font-bold text-primary hover:bg-primary/5 group/btn"
                      onClick={() => {
                        setSelectedDoc(p);
                        setIsDetailOpen(true);
                      }}
                    >
                      Details
                      <ArrowRight className="w-3.5 h-3.5 ml-1 transition-transform group-hover/btn:translate-x-1" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {filteredPolicies.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center space-y-4 bg-muted/20 border border-dashed border-border rounded-[32px]">
              <div className="bg-card w-16 h-16 rounded-3xl flex items-center justify-center shadow-lg">
                <BookOpen className="w-8 h-8 text-muted-foreground/40" />
              </div>
              <div className="space-y-1">
                <p className="text-xl font-bold text-foreground">No guidelines found</p>
                <p className="text-sm text-muted-foreground max-w-xs">Try adjusting your search or add a new policy to the engine.</p>
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="faqs" className="animate-in fade-in slide-in-from-bottom-2 duration-400 outline-none">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredFaqs.map((f) => (
              <div 
                key={f.id} 
                className="group relative bg-card hover:bg-card/80 border border-border hover:border-success/30 rounded-[24px] p-6 transition-all duration-300 shadow-sm hover:shadow-xl hover:shadow-success/5 flex flex-col justify-between overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-success/5 rounded-full -mr-16 -mt-16 blur-3xl transition-all group-hover:bg-success/10" />
                
                <div className="relative space-y-4">
                  <div className="flex items-start justify-between">
                    <Badge variant="secondary" className="bg-success/5 text-success border-success/10 rounded-lg px-2.5 py-1 text-[10px] font-black uppercase tracking-wider">
                      {f.category}
                    </Badge>
                    <Switch checked={f.isActive} onCheckedChange={() => handleToggleFaq(f.id)} />
                  </div>
                  
                  <div>
                    <h4 className="text-[17px] font-black text-foreground group-hover:text-success transition-colors leading-tight mb-2">
                      {f.question}
                    </h4>
                    <p className="text-[13px] text-muted-foreground/80 leading-relaxed line-clamp-3 font-medium h-[60px]">
                      {f.preview}
                    </p>
                  </div>
                </div>

                <div className="relative pt-6 mt-6 border-t border-border flex items-center justify-between">
                  <div className="flex items-center gap-2 text-muted-foreground/60">
                    <BarChart3 className="w-3.5 h-3.5" />
                    <span className="text-[11px] font-bold uppercase tracking-widest">{f.usageCount || 0} HITS</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <Button 
                      variant="ghost" 
                      className="text-[11px] font-bold text-success hover:bg-success/5 group/btn"
                      onClick={() => openFaqModal(f)}
                    >
                      Replace
                      <Upload className="w-3.5 h-3.5 ml-1" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="rounded-xl hover:bg-destructive/10 hover:text-destructive transition-colors h-8 w-8"
                      onClick={() => {
                        setDocToDelete({ id: f.id, type: 'faq' });
                        setIsDeleteConfirmOpen(true);
                      }}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                    <Button 
                      variant="ghost" 
                      className="text-[11px] font-bold text-success hover:bg-success/5 group/btn"
                      onClick={() => {
                        setSelectedDoc(f);
                        setIsDetailOpen(true);
                      }}
                    >
                      Details
                      <ArrowRight className="w-3.5 h-3.5 ml-1 transition-transform group-hover/btn:translate-x-1" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {filteredFaqs.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 text-center space-y-4 bg-muted/20 border border-dashed border-border rounded-[32px]">
              <div className="bg-card w-16 h-16 rounded-3xl flex items-center justify-center shadow-lg">
                <HelpCircle className="w-8 h-8 text-muted-foreground/40" />
              </div>
              <div className="space-y-1">
                <p className="text-xl font-bold text-foreground">No SOP or knowledge documents found</p>
                <p className="text-sm text-muted-foreground max-w-xs">Standard operating procedures will help the AI evaluate customer resolutions.</p>
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Policy Modal */}
      <Dialog open={isPolicyModalOpen} onOpenChange={setIsPolicyModalOpen}>
        <DialogContent className="sm:max-w-[640px] rounded-[28px] border-none shadow-2xl p-0 overflow-hidden">
          <div className="bg-primary px-8 py-8 text-primary-foreground relative">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-32 -mt-32 blur-3xl" />
            <DialogHeader>
              <DialogTitle className="text-3xl font-black">
                {policyTargetId ? "Replace Guideline PDF" : "Upload Guideline PDF"}
              </DialogTitle>
              <DialogDescription className="text-primary-foreground/70 font-medium pt-2">
                {policyTargetId
                  ? "Upload a newer PDF version for the existing guideline."
                  : "Upload a PDF and let the ingestion pipeline extract and index it."}
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="p-8 space-y-6">
            <div className="space-y-2">
              <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">Guideline PDF</p>
              <Input
                type="file"
                accept="application/pdf"
                onChange={(event) => setPolicyFile(event.target.files?.[0] ?? null)}
                className="rounded-xl border-border bg-muted/30 focus-visible:ring-primary/20 h-11 file:mr-4 file:rounded-lg file:border-0 file:bg-primary file:px-4 file:py-2 file:text-primary-foreground file:font-bold"
              />
              <p className="text-[10px] text-muted-foreground pl-1">PDF only. Leave title or category blank to use defaults.</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">Title</p>
                <Input
                  value={policyForm.title}
                  onChange={(event) => setPolicyForm({ ...policyForm, title: event.target.value })}
                  className="rounded-xl border-border bg-muted/30 focus-visible:ring-primary/20 h-11"
                  placeholder="Optional custom title"
                />
              </div>
              <div className="space-y-2">
                <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">Category</p>
                <Input
                  value={policyForm.category}
                  onChange={(event) => setPolicyForm({ ...policyForm, category: event.target.value })}
                  className="rounded-xl border-border bg-muted/30 focus-visible:ring-primary/20 h-11"
                  placeholder="Optional category"
                />
              </div>
            </div>

            <DialogFooter className="pt-2">
              <Button variant="ghost" className="font-bold underline text-muted-foreground" onClick={() => setIsPolicyModalOpen(false)}>
                Cancel
              </Button>
              <Button onClick={savePolicy} className="rounded-xl px-10 font-bold shadow-lg shadow-primary/20">
                {policyTargetId ? "Replace Guideline" : "Upload Guideline"}
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* FAQ Modal (Markdown Enhanced) */}
      <Dialog open={isFaqModalOpen} onOpenChange={setIsFaqModalOpen}>
        <DialogContent className="sm:max-w-[640px] rounded-[28px] border-none shadow-2xl p-0 overflow-hidden">
          <div className="bg-success px-8 py-8 text-success-foreground relative">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-32 -mt-32 blur-3xl" />
            <DialogHeader>
              <DialogTitle className="text-3xl font-black">
                {faqTargetId ? "Replace SOP/Knowledge PDF" : "Upload SOP/Knowledge PDF"}
              </DialogTitle>
              <DialogDescription className="text-success-foreground/70 font-medium pt-2">
                {faqTargetId
                  ? "Upload a newer PDF version for the existing FAQ item."
                  : "Upload a PDF and let the system extract the answer text."}
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="p-8 space-y-6">
            <div className="space-y-2">
              <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">FAQ PDF</p>
              <Input
                type="file"
                accept="application/pdf"
                onChange={(event) => setFaqFile(event.target.files?.[0] ?? null)}
                className="rounded-xl border-border bg-muted/30 focus-visible:ring-success/20 h-11 file:mr-4 file:rounded-lg file:border-0 file:bg-success file:px-4 file:py-2 file:text-white file:font-bold"
              />
              <p className="text-[10px] text-muted-foreground pl-1">PDF only. The extracted text becomes the answer content.</p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">Question</p>
                <Input
                  value={faqForm.question}
                  onChange={(event) => setFaqForm({ ...faqForm, question: event.target.value })}
                  className="rounded-xl border-border bg-muted/30 focus-visible:ring-success/20 h-11"
                  placeholder="Optional question or topic"
                />
              </div>
              <div className="space-y-2">
                <p className="text-[11px] font-black uppercase text-muted-foreground tracking-widest pl-1">Category</p>
                <Input
                  value={faqForm.category}
                  onChange={(event) => setFaqForm({ ...faqForm, category: event.target.value })}
                  className="rounded-xl border-border bg-muted/30 focus-visible:ring-success/20 h-11"
                  placeholder="Optional category"
                />
              </div>
            </div>

            <DialogFooter className="pt-2">
              <Button variant="ghost" className="font-bold underline text-muted-foreground" onClick={() => setIsFaqModalOpen(false)}>
                Cancel
              </Button>
              <Button onClick={saveFaq} className="rounded-xl px-10 font-bold bg-success hover:bg-success/90 text-white shadow-lg shadow-success/20">
                {faqTargetId ? "Replace FAQ" : "Upload FAQ"}
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={isDeleteConfirmOpen} onOpenChange={setIsDeleteConfirmOpen}>
        <DialogContent className="sm:max-w-[400px] rounded-[32px] p-8 border-none shadow-2xl">
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="w-16 h-16 bg-destructive/10 rounded-3xl flex items-center justify-center text-destructive mb-2">
              <AlertTriangle className="w-8 h-8" />
            </div>
            <DialogHeader>
              <DialogTitle className="text-2xl font-black">Hold on!</DialogTitle>
              <DialogDescription className="text-muted-foreground font-medium pt-2">
                Are you sure you want to delete this {docToDelete?.type === 'policy' ? 'guideline' : 'SOP/knowledge document'}? 
                This action will remove it from the knowledge engine immediately.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="w-full flex-col sm:flex-row gap-3 pt-4">
              <Button 
                variant="ghost" 
                className="rounded-xl font-bold flex-1 h-12" 
                onClick={() => setIsDeleteConfirmOpen(false)}
              >
                Go Back
              </Button>
              <Button 
                variant="destructive" 
                className="rounded-xl font-black flex-1 h-12 shadow-lg shadow-destructive/20"
                onClick={handleDelete}
              >
                Yes, Delete
              </Button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Document Detail Modal */}
      <Dialog open={isDetailOpen} onOpenChange={setIsDetailOpen}>
        <DialogContent className="sm:max-w-[850px] max-h-[85vh] p-0 rounded-[32px] border-none shadow-3xl overflow-hidden flex flex-col">
          <div className={`p-8 pb-6 flex items-start justify-between ${isPolicyDoc(selectedDoc) ? 'bg-primary/5' : 'bg-success/5'}`}>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Badge className={`rounded-lg px-2 py-0.5 text-[10px] font-black uppercase tracking-wider ${isPolicyDoc(selectedDoc) ? 'bg-primary/10 text-primary border-none' : 'bg-success/10 text-success border-none'}`}>
                  {selectedDoc?.category}
                </Badge>
                {selectedDoc && !selectedDoc.isActive && (
                  <Badge variant="outline" className="rounded-lg text-[10px] font-bold border-dashed opacity-60">Inactive</Badge>
                )}
              </div>
              <h3 className="text-2xl font-black tracking-tight text-foreground line-clamp-1">
                {isPolicyDoc(selectedDoc) ? selectedDoc.title : (selectedDoc as FAQData | null)?.question}
              </h3>
            </div>
            <Button variant="ghost" size="icon" className="rounded-xl h-10 w-10 -mr-2 shadow-none border-none hover:bg-muted" onClick={() => setIsDetailOpen(false)}>
              <Loader2 className="w-5 h-5" />
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto px-10 py-8 custom-scrollbar bg-background shadow-inner">
            <div className="max-w-3xl mx-auto">
              <article className="prose dark:prose-invert prose-slate max-w-none 
                prose-headings:font-black prose-headings:tracking-tight 
                prose-h1:text-3xl prose-h1:mb-10
                prose-h2:text-2xl prose-h2:border-b prose-h2:border-border prose-h2:pb-2 prose-h2:mt-10
                prose-h3:text-xl prose-h3:mt-8
                prose-p:text-[15px] prose-p:leading-relaxed prose-p:text-foreground/80
                prose-li:text-[15px] prose-li:text-foreground/80
                prose-strong:text-foreground prose-strong:font-bold
                prose-table:border prose-table:border-border prose-table:rounded-xl prose-table:overflow-hidden
                prose-th:bg-muted/50 prose-th:p-4 prose-th:text-[13px] prose-th:font-black prose-th:uppercase prose-th:tracking-wider
                prose-td:p-4 prose-td:border-t prose-td:border-border prose-td:text-[14px]">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {selectedDoc ? (isPolicyDoc(selectedDoc) ? selectedDoc.content : selectedDoc.answer) : ''}
                </ReactMarkdown>
              </article>
            </div>
          </div>
          
          <div className="p-6 border-t border-border bg-card flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest leading-none mb-1">Impact</span>
                <div className="flex items-center gap-2 text-foreground font-bold text-sm">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  {selectedDoc?.usageCount || 0} Agent Evaluations
                </div>
              </div>
              <div className="w-px h-8 bg-border" />
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest leading-none mb-1">Source</span>
                <div className="flex items-center gap-2 text-foreground font-bold text-sm">
                  <FileText className="w-4 h-4 text-muted-foreground" />
                  Nexalink Knowledge Base
                </div>
              </div>
            </div>
            <Button 
              className="rounded-xl px-10 font-black shadow-lg"
              onClick={() => setIsDetailOpen(false)}
            >
              Done Viewing
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
