import { Wrench, ArrowLeft } from "lucide-react";
import { Link, useNavigate } from "react-router";
import { useAuth } from "../../contexts/AuthContext";

export function UnderDevelopment() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const homePath = user?.role === "agent" ? "/agent" : user?.role === "manager" ? "/manager" : "/login";

  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-4 animate-in fade-in zoom-in-95 duration-500">
      <div className="w-16 h-16 bg-primary/10 rounded-2xl flex items-center justify-center mb-6 shadow-sm border border-primary/20">
        <Wrench className="w-8 h-8 text-primary" />
      </div>
      <h1 className="text-2xl font-bold text-foreground mb-2" style={{ fontFamily: 'var(--font-serif)' }}>
        Under Development
      </h1>
      <p className="text-muted-foreground text-[14px] max-w-sm mb-8">
        We're currently building out this section of VocalMind. Check back soon for exciting new updates!
      </p>
      
      <div className="flex items-center gap-3">
        <button 
          onClick={() => navigate(-1)}
          data-cy="under-development-back"
          className="flex items-center gap-2 h-10 px-5 rounded-xl border border-border bg-card text-[13px] font-semibold text-foreground hover:bg-muted transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Go Back
        </button>
        <Link 
          to={homePath}
          data-cy="under-development-home"
          className="flex items-center gap-2 h-10 px-5 rounded-xl bg-primary text-[13px] font-semibold text-primary-foreground hover:bg-primary/90 transition-colors shadow-sm shadow-primary/20"
        >
          Return Home
        </Link>
      </div>
    </div>
  );
}
