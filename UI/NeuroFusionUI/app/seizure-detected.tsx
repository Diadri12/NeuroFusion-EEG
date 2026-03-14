import { useRouter, useLocalSearchParams } from 'expo-router';
import SeizureDetectedScreen from '../src/screens/SeizureDetectedScreen';

export default function SeizureDetected() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const handleGoBack = () => {
    router.replace('/(tabs)/dashboard');
  };

  return<SeizureDetectedScreen
      onGoBack={handleGoBack}
      confidence={params.confidence ?? '0'}
      numSamples={params.numSamples ?? '0'}
      timeTaken={params.timeTaken ?? '0'}
      fileName={params.fileName ?? 'unknown'}
    />;
}