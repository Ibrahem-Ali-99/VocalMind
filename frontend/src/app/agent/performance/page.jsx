'use client'

export default function AgentPerformancePage() {
    const metrics = [
        { label: 'Overall Score', value: '4.2', max: '5', color: 'text-emerald-600', progress: 84 },
        { label: 'Customer Satisfaction', value: '92%', color: 'text-cyan', progress: 92 },
        { label: 'First Call Resolution', value: '87%', color: 'text-purple', progress: 87 },
        { label: 'Average Handle Time', value: '5:23', color: 'text-amber-600', progress: 75 },
    ]

    const skills = [
        { name: 'Empathy', score: 4.5 },
        { name: 'Clarity', score: 4.2 },
        { name: 'Problem Solving', score: 4.0 },
        { name: 'Product Knowledge', score: 3.8 },
        { name: 'Policy Adherence', score: 4.6 },
    ]

    return (
        <div className="max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">My Performance</h1>
                    <p className="text-gray-500 mt-1">Last 30 days</p>
                </div>
                <select className="px-3 py-2 border border-gray-200 rounded-lg text-sm">
                    <option>Last 30 days</option>
                    <option>Last 90 days</option>
                    <option>This year</option>
                </select>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                {metrics.map((metric, i) => (
                    <div key={i} className="bg-white rounded-xl shadow-card p-5">
                        <p className="text-sm text-gray-500 mb-2">{metric.label}</p>
                        <div className="flex items-baseline gap-1 mb-3">
                            <span className={`text-3xl font-bold ${metric.color}`}>{metric.value}</span>
                            {metric.max && <span className="text-lg text-gray-400">/{metric.max}</span>}
                        </div>
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-cyan to-cyan-dark rounded-full transition-all"
                                style={{ width: `${metric.progress}%` }}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Skills Breakdown */}
            <div className="bg-white rounded-xl shadow-card p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-6">Skills Breakdown</h2>
                <div className="space-y-4">
                    {skills.map((skill, i) => (
                        <div key={i} className="flex items-center gap-4">
                            <span className="w-32 text-sm text-gray-600">{skill.name}</span>
                            <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-purple to-purple-dark rounded-full"
                                    style={{ width: `${(skill.score / 5) * 100}%` }}
                                />
                            </div>
                            <span className={`text-sm font-medium ${skill.score >= 4 ? 'text-emerald-600' : 'text-amber-600'}`}>
                                {skill.score.toFixed(1)}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
