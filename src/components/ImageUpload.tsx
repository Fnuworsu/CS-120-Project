
import React, { useState, useRef } from "react";
import { Card } from "@/components/ui/card";
import { FileImage, X } from "lucide-react";

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  id: string;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelected, id }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    // Check if file is an image
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file");
      return;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert("File size should be less than 10MB");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
      onImageSelected(file);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const removeImage = () => {
    setPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <Card
      className={`border-2 border-dashed rounded-md p-4 text-center ${
        isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300"
      } ${preview ? "border-solid" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        id={id}
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/png,image/jpeg,image/jpg"
      />

      {preview ? (
        <div className="relative">
          <img
            src={preview}
            alt="Retina scan preview"
            className="w-full h-48 object-contain"
          />
          <button
            onClick={removeImage}
            className="absolute top-0 right-0 bg-red-500 text-white rounded-full p-1"
            type="button"
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        <label htmlFor={id} className="cursor-pointer block p-6">
          <div className="flex flex-col items-center justify-center">
            <FileImage className="h-12 w-12 text-gray-400 mb-3" />
            <p className="text-blue-500 font-medium">Upload a file or drag and drop</p>
            <p className="text-sm text-gray-500 mt-1">PNG, JPG, JPEG up to 10MB</p>
          </div>
        </label>
      )}
      {preview && (
        <p className="mt-2 text-sm text-gray-500">
          {preview.substring(preview.lastIndexOf("/") + 1).length > 20
            ? "Selected file"
            : preview.substring(preview.lastIndexOf("/") + 1)}
        </p>
      )}
    </Card>
  );
};

export default ImageUpload;
