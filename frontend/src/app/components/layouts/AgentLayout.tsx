import { Outlet, Link, useLocation } from "react-router";
import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { UserNav } from "./UserNav";
import {
  Mic,
  Activity,
  Phone,
  Settings,
  ChevronLeft,
  ChevronRight,
  Bell,
} from "lucide-react";

export function AgentLayout() {
  const { user } = useAuth();
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const getInitials = (name: string) => {
    if (!name) return "??";
    const parts = name.split(" ");
    if (parts.length >= 2) return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    return name.substring(0, 2).toUpperCase();
  };

  const navItems = [
    { icon: Activity, label: "My Performance", path: "/agent" },
    { icon: Phone, label: "My Calls", path: "/agent/calls" },
  ];

  const getPageTitle = () => {
    if (location.pathname === "/agent") return "My Performance";
    if (location.pathname.includes("calls")) return "Call Detail";
    return "My Performance";
  };

  return (
    <div className="flex h-screen bg-background text-foreground transition-colors duration-300">
      {/* Sidebar */}
      <div
        className={`${
          collapsed ? "w-[72px]" : "w-[240px]"
        } bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-300`}
      >
        {/* Logo Area */}
        <div className="h-16 flex items-center px-4 border-b border-sidebar-border">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-primary rounded-xl flex items-center justify-center flex-shrink-0">
              <Mic className="w-[18px] h-[18px] text-primary-foreground" />
            </div>
            {!collapsed && (
              <span className="text-sidebar-foreground font-bold text-lg" style={{ fontFamily: 'var(--font-sans)' }}>
                VocalMind
              </span>
            )}
          </div>
        </div>

        {/* Navigation & Content Area (Flex Grow) */}
        <div className="flex-1 flex flex-col justify-between py-4">
          {/* Top Group: Role Badge & Nav */}
          <div className="space-y-6">
            {!collapsed && (
              <div className="px-4">
                <div className="bg-success/10 border border-success/20 rounded-xl p-3.5 transition-all shadow-sm">
                  <div className="text-[10px] font-bold text-success uppercase tracking-widest mb-1 opacity-80">
                    Agent Portal
                  </div>
                  <div className="text-[12px] text-foreground font-semibold leading-tight">
                    Personal view only
                  </div>
                </div>
              </div>
            )}

            <nav className="px-2 space-y-1.5">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path || 
                  (item.path === "/agent/calls" && location.pathname.includes("/agent/calls"));
                
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center gap-3 px-3.5 h-11 rounded-xl transition-all ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                        : "text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent"
                    }`}
                  >
                    <Icon className="w-[18px] h-[18px] flex-shrink-0" />
                    {!collapsed && (
                      <span className="text-[14px] font-semibold">
                        {item.label}
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Bottom Group (Distributed) */}
          <div className="px-2 space-y-4">
            {/* Additional items could go here if needed to fill space */}
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-sidebar-border p-4">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="w-full flex items-center justify-center h-8 text-sidebar-foreground/60 hover:text-sidebar-foreground transition-colors mb-3"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="w-5 h-5" />
            ) : (
              <ChevronLeft className="w-5 h-5" />
            )}
          </button>

          {!collapsed && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-[11px] font-bold ring-2 ring-white/10">
                {getInitials(user?.name || "Agent")}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sidebar-foreground text-[13px] font-semibold truncate leading-tight">
                  {user?.name || "Robert King"}
                </div>
                <div className="text-sidebar-foreground/60 text-[11px]">
                  Agent
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <div className="h-14 bg-card border-b border-border px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-[16px] font-bold text-foreground">
              {getPageTitle()}
            </h1>
            <span className="px-2.5 py-1 bg-primary/10 text-primary border border-primary/20 rounded-full text-[11px] font-semibold uppercase tracking-wide">
              Agent
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button className="w-8 h-8 flex items-center justify-center bg-accent/30 border border-border rounded-lg hover:bg-accent transition-colors">
              <Bell className="w-4 h-4 text-muted-foreground" />
            </button>
            <UserNav />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
