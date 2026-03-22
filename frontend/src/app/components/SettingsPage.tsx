import React from "react";
import { useAuth } from "../contexts/AuthContext";

export function SettingsPage() {
  const { user } = useAuth();
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground">Account Settings</h1>
        <p className="text-muted-foreground mt-2">Manage your profile, preferences, and security settings.</p>
      </div>

      <div className="space-y-6">
        <div id="profile" className="glass-card bg-card border border-border rounded-xl p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Profile Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">Full Name</label>
              <div className="p-2.5 bg-background border border-border rounded-lg text-foreground">
                {user?.name || "Loading..."}
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">Email Address</label>
              <div className="p-2.5 bg-background border border-border rounded-lg text-foreground">
                {user?.email || "Loading..."}
              </div>
            </div>
          </div>
        </div>

        <div id="security" className="glass-card bg-card border border-border rounded-xl p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Security</h2>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-opacity">
            Change Password
          </button>
        </div>

        <div id="notifications" className="glass-card bg-card border border-border rounded-xl p-6">
          <h2 className="text-xl font-semibold text-foreground mb-4">Notification Preferences</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-background border border-border rounded-lg">
              <span className="text-foreground">Email Notifications</span>
              <div className="w-10 h-5 bg-primary rounded-full relative">
                <div className="absolute right-1 top-1 w-3 h-3 bg-primary-foreground rounded-full"></div>
              </div>
            </div>
            <div className="flex items-center justify-between p-3 bg-background border border-border rounded-lg">
              <span className="text-foreground">Browser Push Notifications</span>
              <div className="w-10 h-5 bg-muted rounded-full relative">
                <div className="absolute left-1 top-1 w-3 h-3 bg-muted-foreground/30 rounded-full"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
