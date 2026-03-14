import { useRouter, useLocalSearchParams } from 'expo-router';
import NoSeizureDetectedScreen from '../src/screens/NoSeizureDetectedScreen';

export default function NoSeizureDetected() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const handleGoBack = () => {
    router.replace('/(tabs)/dashboard');
  };

  return (
    <NoSeizureDetectedScreen
      onGoBack={handleGoBack}
      confidence={params.confidence ?? "0"}
      numSamples={params.numSamples ?? "0"}
      timeTaken={params.timeTaken ?? "0"}
      fileName={params.fileName ?? "unknown"}
    />
  );
}