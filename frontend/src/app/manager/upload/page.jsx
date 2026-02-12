'use client'

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'

export default function UploadPage() {
    const router = useRouter()
    const [files, setFiles] = useState([])
    const [uploading, setUploading] = useState(false)
    const [dragActive, setDragActive] = useState(false)

    const handleDrag = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true)
        } else if (e.type === 'dragleave') {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)

        const droppedFiles = [...e.dataTransfer.files]
        const audioFiles = droppedFiles.filter(f => f.type.startsWith('audio/') || f.name.endsWith('.wav') || f.name.endsWith('.mp3'))
        setFiles(prev => [...prev, ...audioFiles])
    }, [])

    const handleFileInput = (e) => {
        const selectedFiles = [...e.target.files]
        setFiles(prev => [...prev, ...selectedFiles])
    }

    const removeFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index))
    }

    const handleUpload = async () => {
        setUploading(true)
        // Simulate upload
        await new Promise(resolve => setTimeout(resolve, 2000))
        setUploading(false)
        setFiles([])
        router.push('/manager/calls')
    }

    return (
        <div className="max-w-3xl mx-auto">
            <h1 className="text-2xl font-semibold text-gray-900 mb-6">Upload Calls</h1>

            <div className="bg-white rounded-xl shadow-card p-4 sm:p-6">
                {/* Drop Zone */}
                <div
                    className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition-all ${dragActive
                        ? 'border-cyan bg-cyan/5'
                        : 'border-gray-200 hover:border-gray-300'
                        }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                >
                    <svg className="w-12 h-12 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-gray-600 mb-2">
                        <span className="font-medium">Drop audio files here</span> or click to browse
                    </p>
                    <p className="text-sm text-gray-400">
                        Supports MP3, WAV, M4A up to 100MB each
                    </p>
                    <label className="mt-4 inline-block">
                        <input
                            type="file"
                            multiple
                            accept="audio/*,.wav,.mp3,.m4a"
                            onChange={handleFileInput}
                            className="hidden"
                        />
                        <span className="px-4 py-2 bg-white border border-gray-200 rounded-lg text-gray-700 cursor-pointer hover:bg-gray-50">
                            Choose Files
                        </span>
                    </label>
                </div>

                {/* File List */}
                {files.length > 0 && (
                    <div className="mt-6 space-y-3">
                        <h3 className="font-medium text-gray-900">Selected Files ({files.length})</h3>
                        {files.map((file, index) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div className="flex items-center gap-3">
                                    <svg className="w-5 h-5 text-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                                    </svg>
                                    <div>
                                        <p className="text-sm font-medium text-gray-900">{file.name}</p>
                                        <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => removeFile(index)}
                                    className="p-1 text-gray-400 hover:text-red-500"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Upload Button */}
                {files.length > 0 && (
                    <button
                        onClick={handleUpload}
                        disabled={uploading}
                        className="mt-6 w-full py-3 bg-cyan text-navy rounded-lg font-medium hover:bg-cyan-light transition-all disabled:opacity-50"
                    >
                        {uploading ? 'Uploading...' : `Upload ${files.length} File${files.length > 1 ? 's' : ''}`}
                    </button>
                )}
            </div>
        </div>
    )
}
