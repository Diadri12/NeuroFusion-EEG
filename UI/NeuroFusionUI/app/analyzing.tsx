import { useRouter, useLocalSearchParams } from 'expo-router';
import AnalyzingScreen from '../src/screens/AnalyzingScreen';
import { Alert } from "react-native";
import { useEffect } from "react";


export default function Analyzing() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const fileUri = params.fileUri as string | undefined;
  const fileType = params.fileType as string | undefined;

  useEffect(() => {
    // If user refreshes page or params missing
    if (!fileUri || !fileType) {
      Alert.alert("Error", "Missing file information.");
      router.replace("/");
    }
  }, []);

  const handleAnalysisComplete = (result: {
    hasSeizure: boolean;
    label: string;
    confidence: number;
    fileName: string;
    timeTaken?: string;
    numSamples?: number;
    error?: boolean;
  }) => {
    console.log("Analysis Complete:", result);
    // FIX: DO NOT NAVIGATE if error
    if (result.error) {
      Alert.alert("Analysis Failed", "Could not analyze the file.");
      return;
    }

    const target = result.hasSeizure
      ? '/seizure-detected'
      : '/no-seizure-detected';

    router.replace({
      pathname: target,
      params: {
        label: result.label,
        confidence: result.confidence.toString() || '',
        fileName: result.fileName,
        timeTaken: result.timeTaken || '',
        numSamples: result.numSamples?.toString() || '',
      },
    });
  };

  if (!fileUri || !fileType) {
    return null;
  }

  return <AnalyzingScreen
      fileUri={fileUri as string}
      fileType={fileType as string}
      onComplete={handleAnalysisComplete}
    />
}