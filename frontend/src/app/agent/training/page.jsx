export default function AgentTrainingPage() {
    const resources = [
        { id: 1, title: 'Customer Empathy Training', type: 'Video', duration: '15 min', completed: true },
        { id: 2, title: 'New Product Features Q1 2026', type: 'Document', duration: '10 min', completed: true },
        { id: 3, title: 'Handling Difficult Customers', type: 'Video', duration: '25 min', completed: false },
        { id: 4, title: 'Privacy Policy Updates', type: 'Quiz', duration: '5 min', completed: false },
        { id: 5, title: 'Advanced Troubleshooting', type: 'Interactive', duration: '30 min', completed: false },
    ]

    return (
        <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-900">Training Resources</h1>
                    <p className="text-gray-500 mt-1">2 of {resources.length} completed</p>
                </div>
            </div>

            <div className="space-y-4">
                {resources.map((resource) => (
                    <div key={resource.id} className="bg-white rounded-xl shadow-card p-4 sm:p-5 hover:shadow-card-hover transition-all">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                            <div className="flex items-center gap-3 sm:gap-4">
                                <div className={`w-10 h-10 rounded-lg flex-shrink-0 flex items-center justify-center ${resource.completed ? 'bg-emerald-100 text-emerald-600' : 'bg-gray-100 text-gray-500'
                                    }`}>
                                    {resource.completed ? (
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                    ) : (
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                    )}
                                </div>
                                <div>
                                    <h3 className="font-medium text-gray-900">{resource.title}</h3>
                                    <div className="flex items-center gap-2 text-sm text-gray-500">
                                        <span className="px-2 py-0.5 bg-gray-100 rounded text-xs">{resource.type}</span>
                                        <span>{resource.duration}</span>
                                    </div>
                                </div>
                            </div>
                            <button className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${resource.completed
                                    ? 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    : 'bg-cyan text-navy hover:bg-cyan-light'
                                }`}>
                                {resource.completed ? 'Review' : 'Start'}
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
