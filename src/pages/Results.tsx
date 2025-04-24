
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { AlertCircle } from "lucide-react";

// Interfaces for the analysis results
interface AnalysisResult {
  leftEye: EyeAnalysis;
  rightEye: EyeAnalysis;
  systemicFindings: {
    diabetes: {
      risk: "low" | "moderate" | "high";
      confidence: number;
    };
    cardiovascular: {
      risk: "low" | "moderate" | "high";
      confidence: number;
    };
  };
}

interface EyeAnalysis {
  condition: "glaucoma" | "cataract" | "scarring" | "healthy" | "unknown";
  confidence: number;
  icd_code?: string;
  cpt_code?: string;
  prescription?: string;
  severity?: string;
  status?: string;
  symptoms?: string[];
  error?: string;
}

const Results = () => {
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    // Try to get results from sessionStorage
    const storedResults = sessionStorage.getItem('analysisResults');
    
    if (storedResults) {
      try {
        const parsedResults = JSON.parse(storedResults);
        console.log("Retrieved analysis results:", parsedResults);
        setResults(parsedResults);
        
        // Check if we got an error response from the backend
        if (parsedResults.error) {
          setError(parsedResults.error);
          toast({
            title: "Analysis Error",
            description: parsedResults.error,
            variant: "destructive",
          });
        }
        setLoading(false);
      } catch (error) {
        console.error("Error parsing results:", error);
        setError("Failed to parse analysis results");
        setLoading(false);
      }
    } else {
      console.log("No results found in session storage");
      setError("No analysis data found. Please upload eye scans first.");
      setLoading(false);
    }
  }, [toast]);

  const handlePrint = () => {
    toast({
      title: "Preparing report",
      description: "Your report is being prepared for printing",
    });
    window.print();
  };

  const uploadNewScans = () => {
    navigate("/");
  };

  const getConditionColor = (condition: string) => {
    switch (condition) {
      case "glaucoma":
        return "text-amber-600";
      case "cataract":
        return "text-amber-600";
      case "scarring":
        return "text-red-600";
      case "healthy":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high":
        return "text-red-600";
      case "moderate":
        return "text-amber-600";
      case "low":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const getConfidenceText = (confidence: number) => {
    return `${Math.round(confidence * 100)}% confidence`;
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
      <Card className="w-full max-w-4xl bg-white rounded-lg shadow-lg overflow-hidden p-8 print:shadow-none">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-gray-800">Retinal Analysis Results</h1>
          <p className="text-gray-600 mt-2">
            AI-Powered Eye Disorder Detection
          </p>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="mt-4 text-gray-600">Analyzing retinal scans...</p>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-500 text-lg mb-6">{error}</p>
            <Button
              onClick={uploadNewScans}
              className="bg-blue-500 hover:bg-blue-600 text-white"
            >
              Upload New Scans
            </Button>
          </div>
        ) : results ? (
          <div className="space-y-8">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="border rounded-lg p-4 bg-gray-50">
                <h2 className="text-lg font-semibold mb-3">Left Eye (OS)</h2>
                <p className="text-lg">
                  Condition:{" "}
                  <span className={`font-bold ${getConditionColor(results.leftEye.condition)}`}>
                    {results.leftEye.condition.charAt(0).toUpperCase() + results.leftEye.condition.slice(1)}
                  </span>
                </p>
                <p className="text-sm text-gray-500 mb-3">
                  {getConfidenceText(results.leftEye.confidence)}
                </p>
                
                {results.leftEye.condition !== "healthy" && results.leftEye.condition !== "unknown" && results.leftEye.symptoms && (
                  <div className="mt-2">
                    <p className="font-medium">Symptoms:</p>
                    <ul className="list-disc pl-5 text-sm">
                      {results.leftEye.symptoms.map((symptom, index) => (
                        <li key={index}>{symptom}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {results.leftEye.prescription && (
                  <p className="mt-2 text-sm">
                    <span className="font-medium">Recommended treatment:</span> {results.leftEye.prescription}
                  </p>
                )}
                
                {results.leftEye.icd_code && (
                  <p className="mt-2 text-xs text-gray-500">ICD Code: {results.leftEye.icd_code}</p>
                )}

                {results.leftEye.error && (
                  <p className="mt-2 text-xs text-red-500">Error: {results.leftEye.error}</p>
                )}
              </div>

              <div className="border rounded-lg p-4 bg-gray-50">
                <h2 className="text-lg font-semibold mb-3">Right Eye (OD)</h2>
                <p className="text-lg">
                  Condition:{" "}
                  <span className={`font-bold ${getConditionColor(results.rightEye.condition)}`}>
                    {results.rightEye.condition.charAt(0).toUpperCase() + results.rightEye.condition.slice(1)}
                  </span>
                </p>
                <p className="text-sm text-gray-500 mb-3">
                  {getConfidenceText(results.rightEye.confidence)}
                </p>
                
                {results.rightEye.condition !== "healthy" && results.rightEye.condition !== "unknown" && results.rightEye.symptoms && (
                  <div className="mt-2">
                    <p className="font-medium">Symptoms:</p>
                    <ul className="list-disc pl-5 text-sm">
                      {results.rightEye.symptoms.map((symptom, index) => (
                        <li key={index}>{symptom}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {results.rightEye.prescription && (
                  <p className="mt-2 text-sm">
                    <span className="font-medium">Recommended treatment:</span> {results.rightEye.prescription}
                  </p>
                )}
                
                {results.rightEye.icd_code && (
                  <p className="mt-2 text-xs text-gray-500">ICD Code: {results.rightEye.icd_code}</p>
                )}

                {results.rightEye.error && (
                  <p className="mt-2 text-xs text-red-500">Error: {results.rightEye.error}</p>
                )}
              </div>
            </div>

            <div className="border-t pt-6">
              <h2 className="text-xl font-semibold mb-4">Systemic Health Indicators</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h3 className="font-medium mb-2">Diabetes Risk</h3>
                  <p className="text-lg">
                    Risk Level:{" "}
                    <span className={`font-bold ${getRiskColor(results.systemicFindings.diabetes.risk)}`}>
                      {results.systemicFindings.diabetes.risk.charAt(0).toUpperCase() + 
                       results.systemicFindings.diabetes.risk.slice(1)}
                    </span>
                  </p>
                  <p className="text-sm text-gray-500">
                    {getConfidenceText(results.systemicFindings.diabetes.confidence)}
                  </p>
                </div>

                <div className="border rounded-lg p-4 bg-gray-50">
                  <h3 className="font-medium mb-2">Cardiovascular Risk</h3>
                  <p className="text-lg">
                    Risk Level:{" "}
                    <span className={`font-bold ${getRiskColor(results.systemicFindings.cardiovascular.risk)}`}>
                      {results.systemicFindings.cardiovascular.risk.charAt(0).toUpperCase() + 
                       results.systemicFindings.cardiovascular.risk.slice(1)}
                    </span>
                  </p>
                  <p className="text-sm text-gray-500">
                    {getConfidenceText(results.systemicFindings.cardiovascular.confidence)}
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-8 flex flex-col md:flex-row gap-4 justify-center print:hidden">
              <Button
                onClick={handlePrint}
                variant="outline"
                className="flex items-center gap-2"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="6 9 6 2 18 2 18 9"></polyline>
                  <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                  <rect x="6" y="14" width="12" height="8"></rect>
                </svg>
                Print Report
              </Button>
              <Button
                onClick={uploadNewScans}
                className="bg-blue-500 hover:bg-blue-600 text-white"
              >
                Upload New Scans
              </Button>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <p className="text-red-500 text-xl">No analysis data found. Please upload eye scans first.</p>
            <Button
              onClick={uploadNewScans}
              className="bg-blue-500 hover:bg-blue-600 text-white mt-6"
            >
              Upload New Scans
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Results;
