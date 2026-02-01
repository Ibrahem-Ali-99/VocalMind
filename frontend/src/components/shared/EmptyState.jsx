export default function EmptyState({ title, description, icon, action }) {
    return (
        <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            {icon && <div className="mb-4">{icon}</div>}
            <h3 className="text-lg font-medium text-gray-900 mb-1">{title}</h3>
            <p className="text-sm text-gray-500 max-w-sm">{description}</p>
            {action && <div className="mt-4">{action}</div>}
        </div>
    )
}
