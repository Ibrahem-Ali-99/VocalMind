import { User, Bell, Shield, Key } from "lucide-react";

export function ManagerSettings() {
  return (
    <div className="p-4 md:p-8 bg-[#F8FAFC] min-h-screen">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-[#0F172A]">Settings & Preferences</h2>
          <p className="text-[#64748B] text-sm mt-1">Manage your account settings and application preferences.</p>
        </div>

        <div className="bg-white rounded-2xl border border-[#E2E8F0] shadow-sm overflow-hidden flex">
          {/* Sidebar Tabs */}
          <div className="w-64 border-r border-[#E2E8F0] bg-[#F8FAFC]/50 p-4 space-y-1">
            <button className="w-full flex items-center gap-3 px-3 py-2.5 bg-[#EFF6FF] text-[#3B82F6] font-semibold rounded-xl text-sm transition-colors">
              <User className="w-4 h-4" /> Profile
            </button>
            <button className="w-full flex items-center gap-3 px-3 py-2.5 text-[#64748B] hover:bg-[#F1F5F9] font-medium rounded-xl text-sm transition-colors">
              <Bell className="w-4 h-4" /> Notifications
            </button>
            <button className="w-full flex items-center gap-3 px-3 py-2.5 text-[#64748B] hover:bg-[#F1F5F9] font-medium rounded-xl text-sm transition-colors">
              <Shield className="w-4 h-4" /> Privacy & Security
            </button>
            <button className="w-full flex items-center gap-3 px-3 py-2.5 text-[#64748B] hover:bg-[#F1F5F9] font-medium rounded-xl text-sm transition-colors">
              <Key className="w-4 h-4" /> API Keys
            </button>
          </div>

          {/* Content Area */}
          <div className="flex-1 p-8">
            <h3 className="text-lg font-bold text-[#0F172A] border-b border-[#E2E8F0] pb-4 mb-6">Profile Information</h3>
            
            <div className="space-y-6 max-w-lg">
              <div className="flex items-center gap-6">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#3B82F6] to-[#1D4ED8] flex items-center justify-center text-white text-2xl font-bold shadow-sm">
                  MK
                </div>
                <div>
                  <button className="px-4 py-2 bg-white border border-[#E2E8F0] rounded-lg text-sm font-semibold text-[#334155] hover:bg-[#F8FAFC] shadow-sm transition-colors">
                    Change Avatar
                  </button>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-[#334155] mb-1.5">Full Name</label>
                  <input type="text" defaultValue="Manager User" className="w-full h-11 px-3 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-[#3B82F6]/50 focus:border-[#3B82F6] transition-colors" />
                </div>
                
                <div>
                  <label className="block text-sm font-semibold text-[#334155] mb-1.5">Email Address</label>
                  <input type="email" defaultValue="manager@vocalmind.io" className="w-full h-11 px-3 bg-[#F8FAFC] border border-[#E2E8F0] rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-[#3B82F6]/50 focus:border-[#3B82F6] transition-colors" />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-[#334155] mb-1.5">Organization</label>
                  <input type="text" defaultValue="VocalMind Corp" disabled className="w-full h-11 px-3 bg-[#E2E8F0]/50 border border-[#E2E8F0] rounded-xl text-sm text-[#94A3B8] cursor-not-allowed" />
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <button className="px-6 py-2.5 bg-[#3B82F6] hover:bg-[#2563EB] text-white rounded-xl text-sm font-bold shadow-sm shadow-[#3B82F6]/20 transition-all active:scale-95">
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
