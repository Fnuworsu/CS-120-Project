document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const caseId = urlParams.get('caseId');
    const mainContent = document.querySelector('main');

    // Display case ID
    const caseIdDisplay = document.getElementById('caseIdDisplay');
    if (caseIdDisplay) {
        caseIdDisplay.textContent = caseId ? `Case ID: ${caseId}` : 'Sample Data';
    }

    // Helper function to format and display results
    function displayResult(id, value) {
        const element = document.getElementById(id);
        if (!element) return;

        // Format the value if it's a number (like C/D ratio)
        const formattedValue = id.includes('cd-ratio') && !isNaN(value) 
            ? Number(value).toFixed(2) 
            : value;

        element.textContent = formattedValue || 'N/A';

        // Add status indicators if needed
        if (value && typeof value === 'object' && value.status) {
            const statusIcon = document.createElement('span');
            statusIcon.className = `ml-2 ${value.status === 'normal' ? 'text-green-500' : 'text-yellow-500'}`;
            statusIcon.innerHTML = value.status === 'normal' 
                ? '✓'
                : '⚠️';
            element.appendChild(statusIcon);
        }
    }

    try {
        let data;
        
        if (!caseId) {
            // Sample data for development
            data = {
                od: {
                    opticDisc: "Normal appearance",
                    cdRatio: 0.4,
                    macula: "No abnormalities detected",
                    bloodVessels: "Normal caliber and distribution",
                    posteriorPole: "Within normal limits"
                },
                os: {
                    opticDisc: "Normal appearance",
                    cdRatio: 0.3,
                    macula: "No abnormalities detected",
                    bloodVessels: "Normal caliber and distribution",
                    posteriorPole: "Within normal limits"
                }
            };
        } else {
            // Fetch real data if case ID exists
            const response = await fetch(`/api/get-diagnosis?caseId=${caseId}`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            data = await response.json();
        }

        // Update the DOM with the diagnosis results
        displayResult('od-optic-disc', data.od.opticDisc);
        displayResult('os-optic-disc', data.os.opticDisc);
        
        displayResult('od-cd-ratio', data.od.cdRatio);
        displayResult('os-cd-ratio', data.os.cdRatio);
        
        displayResult('od-macula', data.od.macula);
        displayResult('os-macula', data.os.macula);
        
        displayResult('od-blood-vessels', data.od.bloodVessels);
        displayResult('os-blood-vessels', data.os.bloodVessels);
        
        displayResult('od-posterior-pole', data.od.posteriorPole);
        displayResult('os-posterior-pole', data.os.posteriorPole);

    } catch (error) {
        console.error('Error:', error);
        mainContent.innerHTML = `
            <div class="min-h-screen flex items-center justify-center p-4">
                <div class="max-w-md w-full bg-white rounded-xl shadow-lg p-8 text-center">
                    <svg class="mx-auto h-12 w-12 text-red-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 class="text-xl font-semibold text-gray-900 mb-2">Error Loading Results</h2>
                    <p class="text-gray-600 mb-6">An error occurred while fetching the diagnosis results. Please try again.</p>
                    <div class="space-x-4">
                        <button onclick="location.reload()" class="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                            Try Again
                        </button>
                        <a href="index.html" class="inline-flex items-center px-4 py-2 border border-gray-300 text-base font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                            Upload New Scans
                        </a>
                    </div>
                </div>
            </div>`;
    }
}); 