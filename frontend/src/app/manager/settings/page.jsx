'use client'

import { useState } from 'react'

const tabs = [
    { id: 'profile', label: 'Profile Settings', icon: 'user' },
    { id: 'notifications', label: 'Notifications', icon: 'bell' },
    { id: 'security', label: 'Privacy & Security', icon: 'shield' },
]

const Toggle = ({ enabled, onChange, label, description }) => (
    <div className="flex items-center justify-between py-3">
        <div>
            <p className="font-medium text-gray-900">{label}</p>
            {description && <p className="text-sm text-gray-500">{description}</p>}
        </div>
        <button
            type="button"
            onClick={() => onChange(!enabled)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${enabled ? 'bg-cyan' : 'bg-gray-200'
                }`}
        >
            <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${enabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
            />
        </button>
    </div>
)

const InputField = ({ label, type = 'text', value, placeholder, disabled }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
        <input
            type={type}
            defaultValue={value}
            placeholder={placeholder}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan/50 disabled:bg-gray-50 disabled:text-gray-500"
        />
    </div>
)

function ProfileSettings() {
    return (
        <div className="space-y-6">
            <div className="flex items-center gap-4 mb-6">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan to-purple flex items-center justify-center text-white text-2xl font-medium">
                    SM
                </div>
                <div>
                    <button className="px-4 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-all">
                        Change Photo
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <InputField label="First Name" value="Sarah" />
                <InputField label="Last Name" value="Miller" />
            </div>
            <InputField label="Email" type="email" value="sarah.miller@vocalmind.com" />
            <InputField label="Phone" value="+1 (555) 123-4567" />
            <InputField label="Role" value="Manager" disabled />

            <div className="flex justify-end gap-3 pt-4">
                <button className="px-4 py-2 text-gray-600 bg-white border border-gray-200 rounded-lg hover:bg-gray-50">
                    Cancel
                </button>
                <button className="px-4 py-2 bg-cyan text-navy rounded-lg hover:bg-cyan-light">
                    Save Changes
                </button>
            </div>
        </div>
    )
}

function NotificationSettings() {
    const [notifications, setNotifications] = useState({
        emailAlerts: true,
        pushNotifications: true,
        weeklyDigest: false,
        flaggedCalls: true,
    })

    const updateNotification = (key) => {
        setNotifications(prev => ({ ...prev, [key]: !prev[key] }))
    }

    return (
        <div className="space-y-4">
            <Toggle
                label="Email Alerts"
                description="Receive email notifications for important events"
                enabled={notifications.emailAlerts}
                onChange={() => updateNotification('emailAlerts')}
            />
            <Toggle
                label="Push Notifications"
                description="Receive browser push notifications"
                enabled={notifications.pushNotifications}
                onChange={() => updateNotification('pushNotifications')}
            />
            <Toggle
                label="Weekly Digest"
                description="Receive a weekly summary email"
                enabled={notifications.weeklyDigest}
                onChange={() => updateNotification('weeklyDigest')}
            />
            <Toggle
                label="Flagged Call Alerts"
                description="Get notified when calls are flagged for review"
                enabled={notifications.flaggedCalls}
                onChange={() => updateNotification('flaggedCalls')}
            />
        </div>
    )
}

function SecuritySettings() {
    const [security, setSecurity] = useState({
        twoFactor: true,
        sessionTimeout: false,
    })

    const updateSecurity = (key) => {
        setSecurity(prev => ({ ...prev, [key]: !prev[key] }))
    }

    return (
        <div className="space-y-6">
            <div className="space-y-4">
                <Toggle
                    label="Two-Factor Authentication"
                    description="Add an extra layer of security to your account"
                    enabled={security.twoFactor}
                    onChange={() => updateSecurity('twoFactor')}
                />
                <Toggle
                    label="Auto Session Timeout"
                    description="Automatically log out after 30 minutes of inactivity"
                    enabled={security.sessionTimeout}
                    onChange={() => updateSecurity('sessionTimeout')}
                />
            </div>

            <div className="pt-4 border-t border-gray-100">
                <h3 className="font-medium text-gray-900 mb-4">Password</h3>
                <InputField label="Current Password" type="password" />
                <InputField label="New Password" type="password" />
                <InputField label="Confirm Password" type="password" />
                <button className="px-4 py-2 bg-navy text-white rounded-lg hover:bg-navy-light">
                    Update Password
                </button>
            </div>
        </div>
    )
}

export default function SettingsPage() {
    const [activeTab, setActiveTab] = useState('profile')

    const renderContent = () => {
        switch (activeTab) {
            case 'profile':
                return <ProfileSettings />
            case 'notifications':
                return <NotificationSettings />
            case 'security':
                return <SecuritySettings />
            default:
                return <ProfileSettings />
        }
    }

    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-2xl font-semibold text-gray-900 mb-6">Settings</h1>

            <div className="bg-white rounded-xl shadow-card overflow-hidden">
                <div className="flex flex-col md:flex-row">
                    {/* Tabs */}
                    <div className="md:w-64 border-b md:border-b-0 md:border-r border-gray-100">
                        <nav className="p-4 space-y-1">
                            {tabs.map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`w-full text-left px-4 py-3 rounded-lg transition-all ${activeTab === tab.id
                                            ? 'bg-cyan/10 text-cyan font-medium'
                                            : 'text-gray-600 hover:bg-gray-50'
                                        }`}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </nav>
                    </div>

                    {/* Content */}
                    <div className="flex-1 p-6">
                        {renderContent()}
                    </div>
                </div>
            </div>
        </div>
    )
}
