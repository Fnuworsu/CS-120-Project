import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Patient {
  id: number;
  first_name: string;
  last_name: string;
  date_of_birth: string;
  address: string;
  email: string;
  created_at: string;
}

export interface Examination {
  id: number;
  patient_id: number;
  date: string;
  has_glaucoma: boolean;
  has_cataract: boolean;
  has_scarring: boolean;
  has_cardiovascular_disease: boolean;
  has_diabetes: boolean;
  is_healthy: boolean;
  ai_confidence_score: number;
  path: string;
  name: string;
}

export const createPatient = async (patientData: Omit<Patient, 'id' | 'created_at'>) => {
  const response = await api.post<Patient>('/patients/', patientData);
  return response.data;
};

export const createExamination = async (
  patientId: number,
  odImage: File,
  osImage: File
) => {
  const formData = new FormData();
  formData.append('od_image', odImage);
  formData.append('os_image', osImage);

  const response = await api.post<{
    message: string;
    examination_id: number;
    results: {
      glaucoma: boolean;
      cataract: boolean;
      scarring: boolean;
      cardiovascular_disease: boolean;
      diabetes: boolean;
      is_healthy: boolean;
      confidence_scores: {
        glaucoma: number;
        cataract: number;
        scarring: number;
        cardiovascular: number;
        diabetes: number;
        healthy: number;
      };
    };
  }>(`/examinations/${patientId}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const getPatientExaminations = async (patientId: number) => {
  const response = await api.get<Examination[]>(`/patients/${patientId}/examinations`);
  return response.data;
}; 