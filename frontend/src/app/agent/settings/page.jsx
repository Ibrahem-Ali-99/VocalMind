'use client'

import { useState } from 'react'

const mockAgent = {
    firstName: 'Sarah',
    lastName: 'Miller',
    email: 'sarah.miller@vocalmind.com',
    phone: '+1 (555) 123-4567',
    department: 'Customer Support',
    employeeId: 'EMP-2024-1547',
}

const tabs = [
    { id: 'profile', label: 'Profile Settings' },
    { id: 'notifications', label: 'Notifications' },
    { id: 'preferences', label: 'Preferences' },
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

const InputField = ({ label, type = 'text', value, disabled }) => (
    <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
        <input
            type={type}
            defaultValue={value}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan/50 disabled:bg-gray-50 disabled:text-gray-500"
        />
    </div>
)

export default function AgentSettingsPage() {
    const [activeTab, setActiveTab] = useState('profile')
    const [notifications, setNotifications] = useState({
        newAssignments: true,
        feedbackReceived: true,
        shiftReminders: false,
    })
    const [preferences, setPreferences] = useState({
        darkMode: false,
        soundAlerts: true,
    })

    const renderContent = () => {
        switch (activeTab) {
            case 'profile':
                return (
                    <div className="space-y-6">
                        <div className="flex items-center gap-4 mb-6">
                            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan to-purple flex items-center justify-center text-white text-2xl font-medium">
                                SM
                            </div>
                            <div>
                                <button className="px-4 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50">
                                    Change Photo
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <InputField label="First Name" value={mockAgent.firstName} />
                            <InputField label="Last Name" value={mockAgent.lastName} />
                        </div>
                        <InputField label="Email" type="email" value={mockAgent.email} />
                        <InputField label="Phone" value={mockAgent.phone} />
                        <InputField label="Department" value={mockAgent.department} disabled />
                        <InputField label="Employee ID" value={mockAgent.employeeId} disabled />

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

            case 'notifications':
                return (
                    <div className="space-y-4">
                        <Toggle
                            label="New Assignments"
                            description="Get notified when new calls are assigned to you"
                            enabled={notifications.newAssignments}
                            onChange={(v) => setNotifications(p => ({ ...p, newAssignments: v }))}
                        />
                        <Toggle
                            label="Feedback Received"
                            description="Notifications when manager provides feedback"
                            enabled={notifications.feedbackReceived}
                            onChange={(v) => setNotifications(p => ({ ...p, feedbackReceived: v }))}
                        />
                        <Toggle
                            label="Shift Reminders"
                            description="Get reminded before your shift starts"
                            enabled={notifications.shiftReminders}
                            onChange={(v) => setNotifications(p => ({ ...p, shiftReminders: v }))}
                        />
                    </div>
                )

            case 'preferences':
                return (
                    <div className="space-y-4">
                        <Toggle
                            label="Dark Mode"
                            description="Use dark theme across the app"
                            enabled={preferences.darkMode}
                            onChange={(v) => setPreferences(p => ({ ...p, darkMode: v }))}
                        />
                        <Toggle
                            label="Sound Alerts"
                            description="Play sound for incoming notifications"
                            enabled={preferences.soundAlerts}
                            onChange={(v) => setPreferences(p => ({ ...p, soundAlerts: v }))}
                        />
                    </div>
                )

            default:
                return null
        }
    }

    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-2xl font-semibold text-gray-900 mb-6">Settings</h1>

            <div className="bg-white rounded-xl shadow-card overflow-hidden">
                <div className="flex flex-col md:flex-row">
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

                    <div className="flex-1 p-6">
                        {renderContent()}
                    </div>
                </div>
            </div>
        </div>
    )
}
