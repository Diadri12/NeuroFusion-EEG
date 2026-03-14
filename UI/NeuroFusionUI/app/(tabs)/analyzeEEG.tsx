import { useRouter } from 'expo-router';
import AnalyzeEEGScreen from '../../src/screens/AnalyzeEEGScreen';

export default function AnalyzeEEG() {
  const router = useRouter();

  const handleStartAnalysis = () => {
    router.push('/analyzing');
  };

  const handleGoBack = () => {
    router.back();
  };

  return (
    <AnalyzeEEGScreen
      onStartAnalysis={handleStartAnalysis}
      onGoBack={handleGoBack}
    />
  );
}