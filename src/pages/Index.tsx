
import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useNavigate } from "react-router-dom";
import { Upload, Loader, AlertCircle } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import ImageUpload from "@/components/ImageUpload";

const Index = () => {
  const [leftEyeImage, setLeftEyeImage] = useState<File | null>(null);
  const [rightEyeImage, setRightEyeImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleProcessScans = async () => {
    if (!leftEyeImage || !rightEyeImage) {
      toast({
        title: "Missing images",
        description: "Please upload both left and right eye retina scans",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setApiError(null);

    try {
      // Convert the image files to base64
      const leftEyeBase64 = await fileToBase64(leftEyeImage);
      const rightEyeBase64 = await fileToBase64(rightEyeImage);

      console.log("Making API call to backend...");

      // Make API call to the backend
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          leftEyeImage: leftEyeBase64,
          rightEyeImage: rightEyeBase64
        })
      });

      const analysisData = await response.json();
      console.log("Received response from backend:", analysisData);

      if (!response.ok) {
        throw new Error(analysisData.error || `API error: ${response.statusText}`);
      }
      
      // Clear any previous results
      sessionStorage.removeItem('analysisResults');
      
      // Store analysis data in sessionStorage to pass to results page
      sessionStorage.setItem('analysisResults', JSON.stringify(analysisData));
      
      // Navigate to results page
      navigate("/results");
    } catch (error) {
      console.error("Error processing scans:", error);
      
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setApiError(errorMessage);
      
      toast({
        title: "Analysis failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Helper function to convert File to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
      <Card className="w-full max-w-4xl bg-white rounded-lg shadow-lg overflow-hidden p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Retina Scan Analysis</h1>
          <p className="text-gray-600 mt-2">
            Upload retinal scans for AI-powered diagnostic analysis
          </p>
        </div>

        <div className="flex items-center justify-center mb-8">
          <div className="h-0.5 w-full bg-gray-200 relative">
            <div className="absolute left-1/3 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                <div className="w-10 h-10 bg-blue-500 rounded-full border-2 border-white"></div>
              </div>
            </div>
            <div className="absolute right-1/3 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                <div className="w-10 h-10 bg-blue-500 rounded-full border-2 border-white"></div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="text-lg font-medium text-gray-700 mb-2">Left Eye Retina (OS)</h2>
            <ImageUpload 
              onImageSelected={(file) => setLeftEyeImage(file)} 
              id="left-eye"
            />
          </div>

          <div>
            <h2 className="text-lg font-medium text-gray-700 mb-2">Right Eye Retina (OD)</h2>
            <ImageUpload 
              onImageSelected={(file) => setRightEyeImage(file)} 
              id="right-eye"
            />
          </div>
        </div>

        {apiError && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md flex items-start">
            <AlertCircle className="text-red-500 mr-2 h-5 w-5 mt-0.5 flex-shrink-0" />
            <p className="text-red-700 text-sm">{apiError}</p>
          </div>
        )}

        <div className="mt-8 flex justify-center">
          <Button
            onClick={handleProcessScans}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-md flex items-center gap-2 w-48 justify-center"
            disabled={isProcessing}
          >
            {isProcessing ? (
              <Loader className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Upload className="mr-2 h-5 w-5" />
            )}
            {isProcessing ? "Processing..." : "Process Scans"}
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default Index;
